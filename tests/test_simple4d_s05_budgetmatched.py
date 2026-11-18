import copy
import csv
import importlib.util
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler
from cardioresp4d.experiments.simple4d_core_mv import S1Runner,Simple4DCoreMultiViewThickSlice
from cardioresp4d.experiments.simple4d_unified_evaluation import S05BMAdapter,evaluate_matched
from cardioresp4d.adapters.simple4d_physical_thick_slice import PhysicalUpstreamThickSliceAdapter

ROOT=Path(__file__).resolve().parents[1]
QUOTAS={'SAX':8,'2CH':4,'4CH':4}

def observations():
    return [DynamicObservation(image=torch.rand(1,64,64),view=v,slice_id=str(j),dynamic_frame_id=i,center_mm=torch.zeros(3),row_direction=torch.tensor([1.,0,0]),column_direction=torch.tensor([0.,1,0]),normal=torch.tensor([0.,0,1.]),pixel_spacing_mm=torch.ones(2),slice_thickness_mm=1.,qc_valid=True,qc_reason='',normalized_slice_position=0.) for v in ('SAX','2CH','4CH') for j in range(2) for i in range(2)]

def identity(items): return [(o.view,o.slice_id,o.dynamic_frame_id) for o in items]

def test_quota_counts_resume_and_legacy():
    items=observations(); sampler=ViewLocationBalancedSampler(items,seed=12)
    rng=random.Random(12); expected=[]
    for v,count in [('SAX',3),('2CH',3),('4CH',2)]:
        for _ in range(count):
            location=rng.choice(['0','1']); frame=rng.choice([x for x in items if x.view==v and x.slice_id==location]); expected.append((v,location,frame.dynamic_frame_id))
    assert identity(sampler.sample_step(observations_per_update=8))==expected
    state=copy.deepcopy(sampler.state_dict()); batch=sampler.sample_step(view_counts=QUOTAS); assert Counter(x.view for x in batch)==QUOTAS
    restored=ViewLocationBalancedSampler(items,seed=999); restored.load_state_dict(state); assert identity(batch)==identity(restored.sample_step(view_counts=QUOTAS))
    for counts in ({'SAX':8,'2CH':4},{**QUOTAS,'4CH':0},{**QUOTAS,'4CH':True},{**QUOTAS,'foo':1}):
        with pytest.raises(ValueError): sampler.sample_step(view_counts=counts)
    with pytest.raises(ValueError): sampler.sample_step(observations_per_update=8,view_counts=QUOTAS)
    with pytest.raises(ValueError): sampler.sample_step(observations_per_update=None)
    assert len(sampler.sample_step())==3

def test_budget_model_runner_mean_and_resume(tmp_path,monkeypatch):
    import cardioresp4d.simple4d.losses as loss_module
    base=yaml.safe_load((ROOT/'configs/simple4d_core_mv_thickslice_v1.yaml').read_text()); bm=yaml.safe_load((ROOT/'configs/simple4d_core_mv_thickslice_budgetmatched_v1.yaml').read_text())
    assert {k:v for k,v in base.items() if k not in ('experiment','data')}=={k:v for k,v in bm.items() if k not in ('experiment','data')}
    assert bm['data']['view_counts']==QUOTAS and bm['data']['pixel_samples_per_observation']==base['data']['pixel_samples_per_observation']
    opts={'n_levels':2,'n_features_per_level':2,'log2_hashmap_size':5,'base_resolution':2,'max_resolution':4,'mlp_hidden_dims':(8,)}
    def model(): return Simple4DCoreMultiViewThickSlice([-100]*3,[100]*3,[1]*3,use_tcnn=False,ngp_options=opts)
    items=observations()
    def runner(): return S1Runner(model(),ViewLocationBalancedSampler(items,seed=12),view_counts=QUOTAS,pixel_samples=3)
    collected=[]; original=loss_module.respiratory_loss
    def tracked(*args,**kwargs):
        result=original(*args,**kwargs); collected.append(float(result['total'].detach())); return result
    monkeypatch.setattr(loss_module,'respiratory_loss',tracked)
    torch.manual_seed(6); r=runner(); assert isinstance(r.model.psf,PhysicalUpstreamThickSliceAdapter)
    result=r.update(); assert len(collected)==16 and result['total']==pytest.approx(np.mean(collected)); assert [result['view_count_'+x] for x in QUOTAS]==[8,4,4]
    saved=copy.deepcopy(r.state_dict()); collected.clear(); r.update(); expected=copy.deepcopy(r.model.state_dict()); optim=copy.deepcopy(r.optimizer.state_dict()); sched=r.scheduler.state_dict()
    r2=runner(); r2.load_state_dict(saved); r2.update()
    assert all(torch.equal(v,r2.model.state_dict()[k]) for k,v in expected.items())
    assert r2.optimizer.state_dict()['param_groups']==optim['param_groups'] and r2.scheduler.state_dict()==sched
    for k,values in optim['state'].items(): assert all(torch.equal(v,r2.optimizer.state_dict()['state'][k][name]) for name,v in values.items())

def test_evaluation_outputs_temporal_s05bm(tmp_path):
    class Model:
        def encode_observation(self,obs,*,sample): return {'mu':torch.tensor([[float(obs.dynamic_frame_id)]])}
        def render_with_encoded(self,obs,pixels,encoded): return {'predicted_intensity':torch.ones(len(pixels))*encoded['code'][0,0]}
    def ob(i,v='SAX'): return SimpleNamespace(view=v,slice_id='x',dynamic_frame_id=i,timestamp_s=float(i),image=torch.ones(1,3,3)*i)
    checkpoint=tmp_path/'cp'; checkpoint.write_bytes(b'fixed'); adapter=S05BMAdapter(Model(),checkpoint)
    assert all(adapter.supports_target(None,ob(1,v),None,'v1') for v in ('SAX','2CH','4CH'))
    out=tmp_path/'out'; evaluate_matched([adapter],[(ob(0),ob(1),ob(2)),(ob(1),ob(2),ob(3))],out,mode='v2')
    with (out/'temporal_fidelity.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==1 and rows[0]['method']=='S05BM' and rows[0]['view']=='SAX'; adapter.verify_unchanged()


def test_budget_evaluator_loads_checkpoint(tmp_path,monkeypatch):
    import json,sys
    spec=importlib.util.spec_from_file_location('evaluate_bm',ROOT/'scripts/evaluate_simple4d_method_comparison.py'); cli=importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    class Model(torch.nn.Module):
        def __init__(self,*args,**kwargs): super().__init__()
        def encode_observation(self,obs,*,sample): return {'mu':torch.tensor([[float(obs.dynamic_frame_id)]])}
        def render_with_encoded(self,obs,pixels,encoded): return {'predicted_intensity':torch.ones(len(pixels))*encoded['code'][0,0]}
    hashes={v:'hash_'+v for v in ('manifest','qc_table','canonical_domain','static_manifest','static_volume','corrected_pose_manifest','simple4d_vendor_manifest')}
    prov={k:hashes[v] for k,v in [('manifest','manifest'),('qc_table','qc_table'),('canonical_domain','canonical_domain'),('static_manifest','static_manifest'),('static_volume','static_volume'),('corrected_poses','corrected_pose_manifest'),('vendor_manifest','simple4d_vendor_manifest')]}
    config=ROOT/'configs/simple4d_core_mv_thickslice_budgetmatched_v1.yaml'; prov.update({'config':cli.output_sha256(config),'effective_domain':{'lower_lps_mm':[-1]*3,'upper_lps_mm':[1]*3},'spacing':{'resolved_reference_spacing_mm':[1]*3},'backend':'pytorch_fallback'})
    run=tmp_path/'run'; run.mkdir(); cp=run/'cp.pt'; torch.save({'schema':'simple4d_s1_v1','config':yaml.safe_load(config.read_text()),'provenance':prov,'runner':{'model':{}}},cp)
    inputs=tmp_path/'inputs'; inputs.mkdir()
    for name in ('manifest','qc','domain','static'): (inputs/name).write_text('{}')
    frames=[SimpleNamespace(view='SAX',slice_id='x',dynamic_frame_id=i,timestamp_s=float(i),normalized_slice_position=0.,image=torch.ones(1,3,3)*i) for i in range(5)]
    monkeypatch.setattr(cli,'load_inputs',lambda args: ({},{'hashes':hashes},None,(('SAX','x',frames),)))
    monkeypatch.setattr(cli,'Simple4DCoreMultiViewThickSlice',Model)
    out=tmp_path/'out'; monkeypatch.setattr(sys,'argv',['evaluate','--manifest',str(inputs/'manifest'),'--qc-table',str(inputs/'qc'),'--canonical-domain',str(inputs/'domain'),'--static-manifest',str(inputs/'static'),'--s05bm-checkpoint',str(cp),'--output-dir',str(out),'--device','cpu'])
    cli.main(); manifest=json.loads((out/'evaluation_manifest.json').read_text())
    assert manifest['methods'][0]['method_id']=='S05BM' and (out/'temporal_fidelity.csv').is_file()
