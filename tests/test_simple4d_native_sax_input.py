from dataclasses import replace
import numpy as np
import pytest
import torch
from cardioresp4d.training.sampler import DynamicObservation
from cardioresp4d.experiments.simple4d_native_sax import build_native_sax_series,save_native_npz
from third_party.simple4d.simple4d.data import SliceSeries


def obs(view,slice_id,z,frame,t,valid=True):
    return DynamicObservation(image=torch.ones(1,8,8)*.4,view=view,slice_id=slice_id,dynamic_frame_id=frame,center_mm=torch.tensor([0.,0.,z]),row_direction=torch.tensor([1.,0,0]),column_direction=torch.tensor([0.,1,0]),normal=torch.tensor([0.,0,1]),pixel_spacing_mm=torch.ones(2),slice_thickness_mm=2.,qc_valid=valid,qc_reason='',timestamp_s=t)


def test_native_sax_roundtrip(tmp_path):
    items=[obs('SAX','z1',2,4,4),obs('2CH','x',0,7,0),obs('SAX','z0',0,1,1),obs('SAX','z1',2,2,2),obs('SAX','z0',0,3,3),obs('SAX','z0',0,8,5,False)]
    series,metadata=build_native_sax_series(items); path=tmp_path/'input.npz'; save_native_npz(series,path); loaded=SliceSeries.load(path)
    assert len(loaded)==4 and metadata['slice_order']==['z0','z1']
    assert [x['dynamic_frame_id'] for x in metadata['frame_mapping']]==[1,2,3,4]
    assert loaded.slice_index.tolist()==[0,1,0,1] and loaded.spacing==(2,1,1)


def test_reject_irregular_geometry():
    items=[obs('SAX','a',0,1,1),obs('SAX','b',2,2,2),obs('SAX','c',5,3,3)]
    with pytest.raises(ValueError,match='uniformly spaced'): build_native_sax_series(items)
    with pytest.raises(ValueError,match='aligned'): build_native_sax_series([items[0],replace(items[1],center_mm=torch.tensor([1.,0.,2.]))])


def test_native_upstream_step_and_resume(tmp_path,monkeypatch):
    import copy
    from third_party.simple4d.simple4d import networks
    from third_party.simple4d.simple4d.config import build_parser
    from third_party.simple4d.simple4d.model import SIMPLE4D
    from cardioresp4d.experiments.simple4d_native_sax import NativeSAXRunner
    # This is a CPU fallback fixture even on hosts with CUDA and tiny-cuda-nn.
    monkeypatch.setattr(networks,'HAS_TCNN',False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda: False)
    cfg=build_parser().parse_args([]); cfg.num_points=4; cfg.ngp_levels=2; cfg.ngp_features_per_level=2; cfg.ngp_log2_hashmap_size=5; cfg.ngp_base_resolution=2; cfg.ngp_max_resolution=4; cfg.ngp_mlp=[8]; cfg.encoder_channels=4; cfg.deform_layers=[8]*4; cfg.n_holdout=0
    rng=np.random.default_rng(2); series=SliceSeries(rng.random((4,64,64),dtype=np.float32),np.array([0,1,0,1]),(2.,1.,1.),2,(0,0))
    def make(): return SIMPLE4D(cfg,64,64,2,series.spacing,series.depth_crop)
    first=NativeSAXRunner(make(),series,np.arange(4),batch_size=2,seed=12)
    assert first.model.device.type=='cpu' and not first.model.feature_net.use_tcnn
    assert first.update()['step']==1
    checkpoint=tmp_path/'checkpoint.pt'; torch.save(first.state_dict(),checkpoint); saved=torch.load(checkpoint,weights_only=False); first.update(); expected={key:copy.deepcopy(net.state_dict()) for key,net in first.model.networks().items()}
    resumed=NativeSAXRunner(make(),series,np.arange(4),batch_size=2,seed=12); resumed.load_state_dict(saved); resumed.update()
    assert all(torch.equal(value,resumed.model.networks()[key].state_dict()[name]) for key,values in expected.items() for name,value in values.items())


def test_formal_config():
    from pathlib import Path
    from third_party.simple4d.simple4d.config import load_config
    from cardioresp4d.experiments.simple4d_native_sax import validate_native_config
    cfg=load_config(Path(__file__).resolve().parents[1]/'configs/simple4d_native_sax_v1.yaml')
    validate_native_config(cfg); cfg.lr_hash=.002
    with pytest.raises(ValueError,match='configuration mismatch'): validate_native_config(cfg)


def test_backend_and_resume_admission():
    from cardioresp4d.experiments.simple4d_native_sax import validate_native_backend,validate_native_resume
    validate_native_backend('pytorch_fallback',False,False)
    with pytest.raises(RuntimeError,match='BLOCKED_TCNN_UNAVAILABLE'): validate_native_backend('tcnn',False,False)
    validate_native_resume({'schema':'simple4d_s0_v1','provenance':{'backend':'pytorch_fallback'},'runner':{}},{'backend':'pytorch_fallback'})
    with pytest.raises(ValueError): validate_native_resume({'schema':'simple4d_s0_v1','provenance':{'backend':'tcnn'},'runner':{}},{'backend':'pytorch_fallback'})
