import torch
from third_party.simple4d.simple4d.networks import InstantNGPINR, MotionCodeVAE, DeformCodeAdaptive
from cardioresp4d.adapters.nesvor_psf import NeSVoRPSFAdapter
from cardioresp4d.experiments.simple4d_core_mv import Simple4DCoreMultiView,make_s1_optimizer


def test_core_multiview_ownership_and_coordinates(monkeypatch):
    def reject_load(*args,**kwargs): raise AssertionError('from-scratch model must not load a checkpoint')
    monkeypatch.setattr(torch,'load',reject_load)
    torch.manual_seed(6)
    net=Simple4DCoreMultiView([-10]*3,[10]*3,[1]*3,use_tcnn=False,ngp_options={'n_levels':2,'n_features_per_level':2,'log2_hashmap_size':5,'base_resolution':2,'max_resolution':4,'mlp_hidden_dims':(8,)})
    torch.manual_seed(6)
    fresh=Simple4DCoreMultiView([-10]*3,[10]*3,[1]*3,use_tcnn=False,ngp_options={'n_levels':2,'n_features_per_level':2,'log2_hashmap_size':5,'base_resolution':2,'max_resolution':4,'mlp_hidden_dims':(8,)})
    assert all(torch.equal(value,fresh.state_dict()[name]) for name,value in net.state_dict().items())
    assert isinstance(net.canonical.network,InstantNGPINR) and net.canonical.backend=='pytorch_fallback'
    assert type(net.respiratory_encoder) is MotionCodeVAE and not hasattr(net.respiratory_encoder,'view_embedding')
    assert isinstance(net.forward_deformation.network,DeformCodeAdaptive) and isinstance(net.inverse_deformation.network,DeformCodeAdaptive)
    assert isinstance(net.psf,NeSVoRPSFAdapter)
    points=torch.tensor([[3.,-2.,5.]])
    assert torch.allclose(net.canonical.coordinates.normalized_to_world(net.canonical.coordinates.world_to_normalized(points)),points,atol=1e-6)
    opt,scheduler=make_s1_optimizer(net)
    assert [x['lr'] for x in opt.param_groups]==[1e-2,2e-5,5e-5,5e-5,5e-5]
    assert all(p.requires_grad for p in net.parameters()) and scheduler.step_size==5000


def test_joint_update_and_resume(tmp_path):
    import copy
    from cardioresp4d.training.sampler import DynamicObservation,ViewLocationBalancedSampler
    from cardioresp4d.experiments.simple4d_core_mv import S1Runner
    def new(): return Simple4DCoreMultiView([-100]*3,[100]*3,[1]*3,use_tcnn=False,ngp_options={'n_levels':2,'n_features_per_level':2,'log2_hashmap_size':5,'base_resolution':2,'max_resolution':4,'mlp_hidden_dims':(8,)},psf_samples=2)
    observations=[DynamicObservation(image=torch.rand(1,64,64),view=view,slice_id='x',dynamic_frame_id=i,center_mm=torch.zeros(3),row_direction=torch.tensor([1.,0,0]),column_direction=torch.tensor([0.,1,0]),normal=torch.tensor([0.,0,1]),pixel_spacing_mm=torch.ones(2),slice_thickness_mm=1.,qc_valid=True,qc_reason='',normalized_slice_position=0.) for i,view in enumerate(('SAX','2CH','4CH'))]
    runner=S1Runner(new(),ViewLocationBalancedSampler(observations,seed=12),observations_per_update=3,pixel_samples=4)
    row=runner.update(); assert row['global_step']==1 and all(row[key+'_grad_norm']>0 for key in ('canonical','encoder','forward','inverse'))
    from cardioresp4d.experiments.simple4d_resp_v1_r0 import diagnostic_metrics
    diagnostic=diagnostic_metrics(runner.model,runner.sampler,pixel_count=4)
    assert diagnostic['diagnostic_frame_count']==3 and 'jacobian_det_p01' in diagnostic
    checkpoint=tmp_path/'checkpoint.pt'; torch.save(runner.state_dict(),checkpoint); saved=torch.load(checkpoint,weights_only=False); runner.update(); expected=copy.deepcopy(runner.model.state_dict())
    restored=S1Runner(new(),ViewLocationBalancedSampler(observations,seed=12),observations_per_update=3,pixel_samples=4); restored.load_state_dict(saved); restored.update()
    assert all(torch.equal(value,restored.model.state_dict()[name]) for name,value in expected.items())


def test_resume_and_dynamic_anatomy(tmp_path):
    import pytest
    from cardioresp4d.experiments.simple4d_core_mv import validate_s1_resume
    from cardioresp4d.experiments.simple4d_unified_evaluation import S1CoreMVAdapter
    validate_s1_resume({'schema':'simple4d_s1_v1','provenance':{'hash':'a'},'runner':{}},{'hash':'a'})
    with pytest.raises(ValueError): validate_s1_resume({'schema':'simple4d_s1_v1','provenance':{'hash':'b'},'runner':{}},{'hash':'a'})
    net=Simple4DCoreMultiView([-2]*3,[2]*3,[1]*3,use_tcnn=False,ngp_options={'n_levels':2,'n_features_per_level':2,'log2_hashmap_size':5,'base_resolution':2,'max_resolution':4,'mlp_hidden_dims':(8,)})
    checkpoint=tmp_path/'model.pt'; checkpoint.write_bytes(b'unchanged'); adapter=S1CoreMVAdapter(net,checkpoint)
    volume=adapter.render_dynamic_state(torch.zeros(1,32),{'lower_lps_mm':[-2]*3,'spacing_mm':[1]*3,'shape':[3,3,3]},chunk_size=4)
    assert volume.shape==(3,3,3) and torch.isfinite(torch.tensor(volume)).all()
    adapter.verify_unchanged()
