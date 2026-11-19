import copy
from collections import Counter
from types import SimpleNamespace
from pathlib import Path
import pytest
import yaml
import torch
from cardioresp4d.training.sampler import ViewLocationBalancedSampler,proportional_view_counts
from cardioresp4d.experiments.simple4d_unified_evaluation import S05PROPAdapter


def fixture():
    return [SimpleNamespace(view=view,slice_id=str(loc),dynamic_frame_id=frame,qc_valid=True,qc_reason='',timestamp_s=frame) for view,n in [('SAX',3),('2CH',2),('4CH',4)] for loc in range(n) for frame in range(5)]

def signature(batch): return [(x.view,x.slice_id,x.dynamic_frame_id) for x in batch]

def test_fixed_budget_allocation_locality_and_resume():
    assert proportional_view_counts({'SAX':49,'2CH':52,'4CH':42},24)=={'SAX':8,'2CH':9,'4CH':7}
    assert proportional_view_counts({'SAX':1,'2CH':1,'4CH':1},4)=={'SAX':2,'2CH':1,'4CH':1}
    counts=proportional_view_counts({'SAX':3,'2CH':2,'4CH':4},24)
    sampler=ViewLocationBalancedSampler(fixture(),seed=12); state=copy.deepcopy(sampler.state_dict())
    batches=[sampler.sample_step(view_counts=counts) for _ in range(4)]
    for batch in batches:
        assert len(batch)==24 and Counter(x.view for x in batch)==counts
        assert all(0<=x.dynamic_frame_id<5 for x in batch)
    other=ViewLocationBalancedSampler(fixture(),seed=88); other.load_state_dict(state)
    assert [signature(x) for x in batches]==[signature(other.sample_step(view_counts=counts)) for _ in batches]
    with pytest.raises(ValueError): proportional_view_counts({'SAX':0,'2CH':1,'4CH':1},24)

def test_historical_modes_and_temporal_resume():
    a=ViewLocationBalancedSampler(fixture(),seed=12); b=ViewLocationBalancedSampler(fixture(),seed=12)
    assert signature(a.sample_step(observations_per_update=8))==signature(b.sample_step(observations_per_update=8))
    assert signature(a.sample_step(view_counts={'SAX':3,'2CH':2,'4CH':4}))==signature(b.sample_step(view_counts={'SAX':3,'2CH':2,'4CH':4}))
    state=copy.deepcopy(a.state_dict()); expected=signature(a.temporal_batch(max_items=3)); b.load_state_dict(state)
    assert expected==signature(b.temporal_batch(max_items=3))

def test_prop_config_and_evaluator(tmp_path):
    root=Path(__file__).resolve().parents[1]; base=yaml.safe_load((root/'configs/simple4d_core_mv_thickslice_v1.yaml').read_text()); cfg=yaml.safe_load((root/'configs/simple4d_core_mv_thickslice_sliceproportional_v1.yaml').read_text())
    assert all(base[x]==cfg[x] for x in base if x not in ('data','experiment'))
    assert cfg['data']=={'normalization':'per_series','pixel_samples_per_observation':10000,'sampling_mode':'location_proportional_fixed_budget','observations_per_update':24}
    cp=tmp_path/'cp'; cp.write_bytes(b'unchanged'); adapter=S05PROPAdapter(None,cp)
    assert all(adapter.supports_target(None,SimpleNamespace(view=x),None,'v1') for x in ('SAX','2CH','4CH')); adapter.verify_unchanged()

def test_runner_24_mean_and_resume(monkeypatch):
    import numpy as np
    import cardioresp4d.simple4d.losses as loss_module
    from cardioresp4d.training.sampler import DynamicObservation
    from cardioresp4d.experiments.simple4d_core_mv import S1Runner,Simple4DCoreMultiViewThickSlice
    from cardioresp4d.adapters.simple4d_physical_thick_slice import PhysicalUpstreamThickSliceAdapter
    opts={'n_levels':2,'n_features_per_level':2,'log2_hashmap_size':5,'base_resolution':2,'max_resolution':4,'mlp_hidden_dims':(8,)}
    items=[DynamicObservation(image=torch.rand(1,64,64),view=view,slice_id=str(loc),dynamic_frame_id=frame,center_mm=torch.zeros(3),row_direction=torch.tensor([1.,0,0]),column_direction=torch.tensor([0.,1,0]),normal=torch.tensor([0.,0,1.]),pixel_spacing_mm=torch.ones(2),slice_thickness_mm=1.,qc_valid=True,qc_reason='',normalized_slice_position=0.) for view,n in [('SAX',12),('2CH',15),('4CH',10)] for loc in range(n) for frame in range(2)]
    def runner():
        net=Simple4DCoreMultiViewThickSlice([-100]*3,[100]*3,[1]*3,use_tcnn=False,ngp_options=opts)
        return S1Runner(net,ViewLocationBalancedSampler(items,seed=12),sampling_mode='location_proportional_fixed_budget',view_counts=proportional_view_counts({'SAX':12,'2CH':15,'4CH':10},24),pixel_samples=2)
    totals=[]; orig=loss_module.respiratory_loss
    def tracked(*args,**kwargs):
        result=orig(*args,**kwargs); totals.append(float(result['total'].detach())); return result
    monkeypatch.setattr(loss_module,'respiratory_loss',tracked)
    torch.manual_seed(6); first=runner(); assert isinstance(first.model.psf,PhysicalUpstreamThickSliceAdapter)
    row=first.update(); assert len(totals)==24 and row['total']==pytest.approx(np.mean(totals))
    assert (row['view_count_SAX'],row['view_count_2CH'],row['view_count_4CH'],row['total_observations'])==(8,10,6,24)
    state=copy.deepcopy(first.state_dict()); first.update(); expected=copy.deepcopy(first.model.state_dict())
    resumed=runner(); resumed.load_state_dict(state); resumed.update()
    assert all(torch.equal(v,resumed.model.state_dict()[k]) for k,v in expected.items())

def test_prop_evaluator_cli_checkpoint(tmp_path,monkeypatch):
    import importlib.util,json,sys
    root=Path(__file__).resolve().parents[1]; spec=importlib.util.spec_from_file_location('prop_eval',root/'scripts/evaluate_simple4d_method_comparison.py'); cli=importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    class Model(torch.nn.Module):
        def __init__(self,*args,**kwargs): super().__init__()
        def encode_observation(self,obs,*,sample): return {'mu':torch.tensor([[float(obs.dynamic_frame_id)]])}
        def render_with_encoded(self,obs,pixels,encoded): return {'predicted_intensity':torch.ones(len(pixels))*encoded['code'][0,0]}
    hashes={x:'hash_'+x for x in ('manifest','qc_table','canonical_domain','static_manifest','static_volume','corrected_pose_manifest','simple4d_vendor_manifest')}
    prov={k:hashes[v] for k,v in [('manifest','manifest'),('qc_table','qc_table'),('canonical_domain','canonical_domain'),('static_manifest','static_manifest'),('static_volume','static_volume'),('corrected_poses','corrected_pose_manifest'),('vendor_manifest','simple4d_vendor_manifest')]}
    cfg=root/'configs/simple4d_core_mv_thickslice_sliceproportional_v1.yaml'
    prov.update({'config':cli.output_sha256(cfg),'effective_domain':{'lower_lps_mm':[-1]*3,'upper_lps_mm':[1]*3},'spacing':{'resolved_reference_spacing_mm':[1]*3},'backend':'pytorch_fallback','sampling_mode':'location_proportional_fixed_budget','observations_per_update':24,'location_counts':{'SAX':49,'2CH':52,'4CH':42},'view_counts':{'SAX':8,'2CH':9,'4CH':7},'total_observations_per_update':24,'location_sampling_within_view':'uniform_with_replacement','frame_sampling_within_location':'uniform','pixel_samples_per_observation':10000,'formal_steps':10000})
    run=tmp_path/'run'; run.mkdir(); cp=run/'cp.pt'; torch.save({'schema':'simple4d_s1_v1','config':yaml.safe_load(cfg.read_text()),'provenance':prov,'runner':{'model':{}}},cp)
    inputs=tmp_path/'inputs'; inputs.mkdir()
    for name in ('manifest','qc','domain','static'): (inputs/name).write_text('{}')
    frames=[SimpleNamespace(view='SAX',slice_id='x',dynamic_frame_id=i,timestamp_s=float(i),normalized_slice_position=0.,image=torch.ones(1,3,3)*i) for i in range(5)]
    monkeypatch.setattr(cli,'load_inputs',lambda args: ({},{'hashes':hashes},None,(('SAX','x',frames),))); monkeypatch.setattr(cli,'Simple4DCoreMultiViewThickSlice',Model)
    out=tmp_path/'out'; monkeypatch.setattr(sys,'argv',['evaluate','--manifest',str(inputs/'manifest'),'--qc-table',str(inputs/'qc'),'--canonical-domain',str(inputs/'domain'),'--static-manifest',str(inputs/'static'),'--s05prop-checkpoint',str(cp),'--output-dir',str(out),'--device','cpu'])
    cli.main(); assert json.loads((out/'evaluation_manifest.json').read_text())['methods'][0]['method_id']=='S05PROP' and (out/'temporal_fidelity.csv').is_file()
