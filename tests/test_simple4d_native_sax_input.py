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
    items=[obs('SAX',str(i),z,i,i) for i,z in enumerate((0,2,5,7))]
    with pytest.raises(ValueError,match='uniformly spaced'): build_native_sax_series(items)
    with pytest.raises(ValueError,match='aligned'): build_native_sax_series([items[0],replace(items[1],center_mm=torch.tensor([1.,0.,2.]))])
    with pytest.raises(ValueError,match='orientation'): build_native_sax_series([items[0],replace(items[1],row_direction=torch.tensor([0.,1.,0.]))])


@pytest.mark.parametrize('positions,expected,missing',[
    ([0,2,4,6],[0,1,2,3],[]),
    ([0,2,4,8,10],[0,1,2,4,5],[3]),
    ([0,4,6,10],[0,2,3,5],[1,4]),
])
def test_nominal_lattice_roundtrip(tmp_path,positions,expected,missing):
    from third_party.simple4d.simple4d.coords import make_slice_coords
    items=[obs('SAX',str(i),z,i,i) for i,z in enumerate(positions)]
    series,metadata=build_native_sax_series(items)
    path=tmp_path/'lattice.npz'; save_native_npz(series,path); loaded=SliceSeries.load(path)
    assert loaded.spacing[0]==pytest.approx(2.) and loaded.num_positions==expected[-1]+1
    assert loaded.slice_index.tolist()==expected and metadata['missing_slice_indices']==missing
    assert metadata['observed_slice_indices']==expected and metadata['num_observed_positions']==len(positions)
    assert len(loaded)==len(positions) and not metadata['interpolation_used'] and not metadata['synthetic_slices_created']
    coords=make_slice_coords((len(loaded),8,8),torch.from_numpy(loaded.slice_index),8,loaded.num_positions)
    assert coords.shape==(len(loaded),64,3) and torch.isfinite(coords).all()


def test_builder_uses_nominal_geometry_and_binds_corrected_pose(tmp_path,monkeypatch):
    import hashlib,importlib.util,json,sys
    from pathlib import Path
    from cardioresp4d.experiments import respiratory_s2vdreme_reference_v1 as corrected
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('native_sax_builder',root/'scripts/build_simple4d_native_sax_input.py')
    builder=importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
    files={name:tmp_path/name for name in ('manifest','qc','pose')}
    for name,path in files.items(): path.write_text(name)
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    static=tmp_path/'static.json'; static.write_text(json.dumps({'source_manifest_sha256':digest(files['manifest']),'qc_table_sha256':digest(files['qc']),'corrected_pose_manifest':{'path':str(files['pose']),'sha256':digest(files['pose'])}}))
    items=[obs('SAX',str(i),z,i,i) for i,z in enumerate((0,2,4,8,10))]
    monkeypatch.setattr(builder,'observations_from_manifest',lambda *args,**kwargs:(items,{},[]))
    def forbidden(*args,**kwargs): raise AssertionError('corrected poses must not be applied to S0')
    monkeypatch.setattr(corrected,'load_corrected_pose_map',forbidden)
    monkeypatch.setattr(corrected,'apply_corrected_static_poses',forbidden)
    output=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['builder','--manifest',str(files['manifest']),'--qc-table',str(files['qc']),'--static-manifest',str(static),'--output-dir',str(output)])
    builder.main(); manifest=json.loads((output/'input_adapter_manifest.json').read_text())
    assert manifest['geometry_source']=='nominal_acquisition_geometry' and manifest['corrected_pose_applied'] is False
    assert manifest['corrected_pose_role']=='provenance_only_not_applied_to_s0_geometry'
    assert manifest['source_sha256']['corrected_pose_manifest']==digest(files['pose'])
    assert manifest['missing_slice_indices']==[3] and SliceSeries.load(output/'sax_input.npz').num_positions==6
    files['pose'].write_text('changed')
    monkeypatch.setattr(sys,'argv',['builder','--manifest',str(files['manifest']),'--qc-table',str(files['qc']),'--static-manifest',str(static),'--output-dir',str(tmp_path/'bad')])
    with pytest.raises(ValueError,match='binding mismatch'): builder.main()


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
