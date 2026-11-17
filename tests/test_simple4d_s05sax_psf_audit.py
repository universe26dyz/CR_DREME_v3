import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler
from cardioresp4d.experiments.simple4d_unified_evaluation import S05SAXAdapter, S1CoreMVAdapter
from cardioresp4d.experiments.simple4d_native_sax_collapse import temporal_stats
from cardioresp4d.adapters.nesvor_psf import NeSVoRPSFAdapter

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('psf_audit',ROOT/'scripts/audit_simple4d_s1_psf_stochasticity.py')
audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def obs(view='SAX', sid='s1', frame=1):
    return DynamicObservation(image=torch.zeros(1,8,8),view=view,slice_id=sid,dynamic_frame_id=frame,center_mm=torch.zeros(3),row_direction=torch.tensor([1.,0,0]),column_direction=torch.tensor([0.,1,0]),normal=torch.tensor([0.,0,1.]),pixel_spacing_mm=torch.ones(2),slice_thickness_mm=1.,qc_valid=True,qc_reason='',normalized_slice_position=0.)


def test_s05sax_admission_and_config(tmp_path):
    a=yaml.safe_load((ROOT/'configs/simple4d_core_mv_thickslice_v1.yaml').read_text()); b=yaml.safe_load((ROOT/'configs/simple4d_core_sax_thickslice_v1.yaml').read_text())
    assert {k:v for k,v in a.items() if k!='experiment'}=={k:v for k,v in b.items() if k!='experiment'}
    sampler=ViewLocationBalancedSampler([obs('SAX','x'),obs('SAX','y')],seed=12,required_views=('SAX',))
    state=sampler.state_dict(); first=sampler.sample_step(observations_per_update=8); sampler.load_state_dict(state)
    assert all(x.view=='SAX' for x in first) and first==sampler.sample_step(observations_per_update=8)
    path=tmp_path/'cp'; path.write_bytes(b'untouched'); adapter=S05SAXAdapter(None,path)
    assert adapter.supports_target(None,obs(),None,'v1') and not adapter.supports_target(None,obs('2CH'),None,'v1')
    adapter.verify_unchanged()


class FakeModel(torch.nn.Module):
    def __init__(self): super().__init__(); self.psf=NeSVoRPSFAdapter(8); self.seen=[]
    def render_with_encoded(self, observation, pixels, encoded):
        self.seen.append((self.psf.n_samples,encoded['code'].clone()))
        n=len(pixels); centers=torch.zeros(n,3); basis=torch.eye(3); resolution=torch.ones(n,3)
        samples=self.psf.sample(centers,resolution,row_direction=basis[0].expand(n,-1),column_direction=basis[1].expand(n,-1),normal=basis[2].expand(n,-1))
        return {'predicted_intensity':samples.square().sum(-1).mean(-1)+encoded['code'][0,0]}


def test_psf_modes_deterministic_and_read_only(tmp_path):
    cp=tmp_path/'checkpoint'; cp.write_bytes(b'unchanged'); model=FakeModel(); adapter=S1CoreMVAdapter(model,cp)
    x=obs(); pixels=torch.stack((torch.arange(64)%8,torch.arange(64)//8),-1); code=torch.ones(1,32)
    r=lambda mode,frame: audit.render(adapter,x,pixels,code,mode=mode,frame_index=frame,chunk_size=16)
    assert np.array_equal(r('crn8',1),r('crn8',2))
    assert not np.array_equal(r('independent8',1),r('independent8',2))
    assert not np.array_equal(r('repeat8',1),r('repeat8',2))
    high=r('high128',1); assert high.shape==(64,) and any(n==128 for n,_ in model.seen)
    assert all(torch.equal(c,code) for _,c in model.seen) and model.psf.n_samples==8
    adapter.verify_unchanged(); assert cp.read_bytes()==b'unchanged'


def test_temporal_math():
    actual=np.array([[[0.,1.]],[[2.,3.]]]); predicted=actual*.5
    result=temporal_stats(actual,predicted)
    assert result['rendered_temporal_std_ratio']==.5 and result['adjacent_difference_ratio']==.5
    assert result['dynamic_only_mse']==.25 and np.isclose(result['mean_temporal_correlation'],1.)


def test_psf_audit_outputs_synthetic(tmp_path,monkeypatch):
    import json
    from dataclasses import replace
    (tmp_path/'inputs').mkdir(); (tmp_path/'run').mkdir()
    paths={name:tmp_path/'inputs'/name for name in ('manifest','qc','domain','static','model','volume','poses')}
    for name,path in paths.items(): path.write_text('{}')
    paths['static'].write_text(json.dumps({'outputs':{'model':{'path':str(paths['model'])},'volume':{'path':str(paths['volume'])}},'corrected_pose_manifest':{'path':str(paths['poses'])}}))
    hashes={name:'sha_'+name for name in ('manifest','qc_table','canonical_domain','static_manifest','static_volume','corrected_pose_manifest','simple4d_vendor_manifest')}
    source={k:hashes[v] for k,v in [('manifest','manifest'),('qc_table','qc_table'),('canonical_domain','canonical_domain'),('static_manifest','static_manifest'),('static_volume','static_volume'),('corrected_poses','corrected_pose_manifest'),('vendor_manifest','simple4d_vendor_manifest')]}
    source.update({'config':audit.output_sha256(ROOT/'configs/simple4d_core_mv_v1.yaml'),'backend':'pytorch_fallback','effective_domain':{'lower_lps_mm':[-1]*3,'upper_lps_mm':[1]*3},'spacing':{'resolved_reference_spacing_mm':[1]*3}})
    cp=tmp_path/'run'/'cp.pt'; torch.save({'schema':'simple4d_s1_v1','provenance':source,'config':yaml.safe_load((ROOT/'configs/simple4d_core_mv_v1.yaml').read_text()),'runner':{'global_step':2000,'model':{}}},cp)
    class Model(FakeModel):
        def __init__(self,*args,**kwargs): super().__init__()
        def encode_observation(self,observation,sample=False): return {'mu':torch.full((1,32),float(observation.dynamic_frame_id)/1000)}
    cohort=[]
    for view,sid in [('SAX','SAX_s018'),('2CH','two'),('4CH','four')]:
        frames=[replace(obs(view,sid,801 if i==1 else i+810),timestamp_s=float(i)) for i in range(4)]
        cohort.append((view,sid,frames))
    monkeypatch.setattr(audit,'load_inputs',lambda args: ({},{'hashes':hashes},None,cohort))
    monkeypatch.setattr(audit,'Simple4DCoreMultiView',Model)
    monkeypatch.setattr(audit,'representative_locations',lambda entries: [x[0] for x in entries])
    monkeypatch.setattr('sys.argv',['audit','--manifest',str(paths['manifest']),'--qc-table',str(paths['qc']),'--canonical-domain',str(paths['domain']),'--static-manifest',str(paths['static']),'--checkpoint',str(cp),'--output-dir',str(tmp_path/'result'),'--device','cpu','--expected-frames','2','--chunk-size','16'])
    digest=audit.output_sha256(cp); audit.main(); assert audit.output_sha256(cp)==digest
    result=tmp_path/'result'; assert all((result/name).is_file() for name in ('psf_stochasticity_summary.json','psf_stochasticity_by_view.csv','psf_temporal_fidelity.csv','same_frame_repeat_summary.json','same_frame_repeat_std_requested.png','independent8_SAX_s018.gif','crn8_SAX_s018.gif','high128_SAX_s018.gif'))
    summary=json.loads((result/'same_frame_repeat_summary.json').read_text()); assert summary['requested']['frame_id']==801 and summary['requested']['repeats']==32


def test_s05sax_runner_resume(tmp_path):
    from dataclasses import replace
    from cardioresp4d.experiments.simple4d_core_mv import Simple4DCoreMultiViewThickSlice, S1Runner
    options={'n_levels':2,'n_features_per_level':2,'log2_hashmap_size':5,'base_resolution':2,'max_resolution':4,'mlp_hidden_dims':(8,)}
    def model(): return Simple4DCoreMultiViewThickSlice([-100]*3,[100]*3,[1]*3,use_tcnn=False,ngp_options=options)
    observations=[replace(obs('SAX','loc',i),image=torch.rand(1,64,64)) for i in range(3)]
    def runner(): return S1Runner(model(),ViewLocationBalancedSampler(observations,seed=12,required_views=('SAX',)),observations_per_update=3,pixel_samples=4)
    first=runner(); first.update(); path=tmp_path/'cp.pt'; torch.save(first.state_dict(),path); saved=torch.load(path,weights_only=False); first.update()
    resumed=runner(); resumed.load_state_dict(saved); resumed.update()
    assert resumed.global_step==2 and resumed.scheduler.state_dict()==first.scheduler.state_dict()
    assert all(torch.equal(v,resumed.model.state_dict()[k]) for k,v in first.model.state_dict().items())
