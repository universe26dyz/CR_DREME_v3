from types import SimpleNamespace
import torch
from cardioresp4d.experiments.simple4d_cardiac_residual import StagedCardiacResidual
from cardioresp4d.experiments.simple4d_core_mv import Simple4DCoreMultiViewThickSlice
from cardioresp4d.roi.cardiac_box import CardiacBox


def test_frozen_base_locality_and_gradients():
    opts={'n_levels':2,'n_features_per_level':2,'log2_hashmap_size':5,'base_resolution':2,'max_resolution':4,'mlp_hidden_dims':(8,)}
    base=Simple4DCoreMultiViewThickSlice([-20]*3,[20]*3,[1]*3,use_tcnn=False,ngp_options=opts)
    model=StagedCardiacResidual(base,CardiacBox([0,0,0],[10,10,10]),taper_mm=2,hidden_dim=16)
    assert all(not p.requires_grad for p in base.parameters())
    points=torch.tensor([[[0.,0.,0.],[10.,0.,0.]]]); code=torch.zeros(1,32); card=torch.zeros(1,16)
    base_u=base.forward_deformation(points,code)['displacement_mm']
    residual=model.forward_displacement(points,code,card)-base_u
    torch.testing.assert_close(residual[:,1],torch.zeros_like(residual[:,1]),atol=0,rtol=0)
    model.forward_displacement(points,code,card).square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.forward_residual.parameters())
    assert all(p.grad is None for p in base.parameters())
    assert model.inverse_displacement(points,code,card).shape == points.shape


def test_zero_residual_base_equivalence_and_encoder_gradients():
    opts={'n_levels':2,'n_features_per_level':2,'log2_hashmap_size':5,'base_resolution':2,'max_resolution':4,'mlp_hidden_dims':(8,)}
    base=Simple4DCoreMultiViewThickSlice([-20]*3,[20]*3,[1]*3,use_tcnn=False,ngp_options=opts)
    model=StagedCardiacResidual(base,CardiacBox([0,0,0],[20,20,20]),taper_mm=2,hidden_dim=16)
    model.train(); assert not model.base.training
    obs=SimpleNamespace(image=torch.rand(1,64,64),center_mm=torch.zeros(3),row_direction=torch.tensor([1.,0,0]),column_direction=torch.tensor([0.,1.,0]),pixel_spacing_mm=torch.ones(2)*.25,normalized_slice_position=0.)
    cardiac,mu,logvar=model.cardiac_encoder(obs,model.box,model.taper_mm)
    points=torch.tensor([[[0.,0.,0.],[1.,1.,1.]]]); base_code=torch.zeros(1,32)
    model.forward_displacement(points,base_code,cardiac).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.cardiac_encoder.parameters())
    class Zero(torch.nn.Module):
        def forward(self, points, code, box, taper): return torch.zeros_like(points)
    model.forward_residual=Zero(); model.inverse_residual=Zero()
    torch.testing.assert_close(model.forward_displacement(points,base_code,cardiac),base.forward_deformation(points,base_code)['displacement_mm'],rtol=0,atol=0)
    torch.testing.assert_close(model.inverse_displacement(points,base_code,cardiac),base.inverse_deformation(points,base_code)['displacement_mm'],rtol=0,atol=0)


def test_runner_step_and_resume():
    import copy
    from cardioresp4d.experiments.simple4d_cardiac_residual import CardiacRunner
    from cardioresp4d.training.sampler import ViewLocationBalancedSampler, DynamicObservation
    torch.set_num_threads(1)
    opts={'n_levels':2,'n_features_per_level':2,'log2_hashmap_size':5,'base_resolution':2,'max_resolution':4,'mlp_hidden_dims':(8,)}
    def make():
        base=Simple4DCoreMultiViewThickSlice([-20]*3,[20]*3,[1]*3,use_tcnn=False,ngp_options=opts)
        model=StagedCardiacResidual(base,CardiacBox([0,0,0],[12]*3),taper_mm=2,hidden_dim=16)
        obs=[DynamicObservation(image=torch.ones(1,64,64)*.3,view=v,slice_id=v,dynamic_frame_id=0,center_mm=torch.zeros(3),row_direction=torch.tensor([1.,0,0]),column_direction=torch.tensor([0.,1,0]),normal=torch.tensor([0.,0,1.]),pixel_spacing_mm=torch.tensor([.2,.2]),slice_thickness_mm=1.,qc_valid=True,qc_reason='',normalized_slice_position=0.) for v in ('SAX','2CH','4CH')]
        return CardiacRunner(model,ViewLocationBalancedSampler(obs,seed=12),pixel_samples=4,reg_samples=2)
    torch.manual_seed(4); runner=make(); row=runner.update(); assert row['step']==1
    state=copy.deepcopy(runner.state_dict()); runner.update(); expected=copy.deepcopy(runner.state_dict())
    restored=make(); restored.load_state_dict(state); restored.update()
    assert restored.step==2 and all(torch.equal(v,restored.model.state_dict()[k]) for k,v in expected['model'].items())
