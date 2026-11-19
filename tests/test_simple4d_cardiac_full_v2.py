import numpy as np
import torch
from types import SimpleNamespace
from cardioresp4d.experiments.simple4d_cardiac_full_v2 import FullCardiacModel,crop_rectangle,heart_crop,temporal_energy,regularization_weights,FullCardiacRunner
from cardioresp4d.experiments.simple4d_core_mv import Simple4DCoreMultiViewThickSlice
from cardioresp4d.roi.cardiac_box import CardiacBox
from cardioresp4d.training.sampler import DynamicObservation,ViewLocationBalancedSampler


def build():
    opts={'n_levels':2,'n_features_per_level':2,'log2_hashmap_size':5,'base_resolution':2,'max_resolution':4,'mlp_hidden_dims':(8,)}
    base=Simple4DCoreMultiViewThickSlice([-20]*3,[20]*3,[1]*3,use_tcnn=False,ngp_options=opts)
    return FullCardiacModel(base,CardiacBox([0,0,0],[12]*3),taper_mm=2,crop_size=64)


def obs(v='SAX',i=0):
    return DynamicObservation(image=torch.ones(1,64,64)*(.1+i*.01),view=v,slice_id=v,dynamic_frame_id=i,center_mm=torch.zeros(3),row_direction=torch.tensor([1.,0,0]),column_direction=torch.tensor([0.,1,0]),normal=torch.tensor([0.,0,1.]),pixel_spacing_mm=torch.tensor([.2,.2]),slice_thickness_mm=1.,qc_valid=True,qc_reason='',normalized_slice_position=0.,timestamp_s=float(i))


def test_crop_architecture_freeze_and_locality():
    model=build(); assert model.cardiac_encoder.network.code_dim==32
    assert len(list(model.forward_residual.network.parameters()))==len(list(model.base.forward_deformation.network.parameters()))
    assert not set(map(id,model.base.parameters())) & set(map(id,model.cardiac_encoder.parameters()))
    assert all(not p.requires_grad for p in model.base.parameters())
    for v in ('SAX','2CH','4CH'):
        o=obs(v);r=crop_rectangle(o,model.box);np.testing.assert_allclose(r,crop_rectangle(o,model.box));assert heart_crop(o,r,64).shape==(1,1,64,64)
    points=torch.tensor([[[0.,0.,0.],[15.,0.,0.]]]);code=torch.zeros(1,32); card=model.forward_residual(points,code,model.box,2)
    assert torch.equal(card[:,1],torch.zeros_like(card[:,1]))
    model.set_stage(3); assert all(p.requires_grad for p in model.base.respiratory_encoder.parameters());assert all(not p.requires_grad for p in model.base.canonical.parameters())


def test_energy_and_schedule():
    z=torch.zeros(4,32,requires_grad=True); z.data[:,0]=torch.tensor([-.01,0.,.01,.02]);v,l=temporal_energy(z,.01); l.backward(); assert torch.isfinite(z.grad).all() and z.grad[:,0].abs().sum()>0
    assert regularization_weights(500)['cycle']==.01 and regularization_weights(2000)['cycle']==.1
    assert temporal_energy(torch.tensor([[-1.],[1.]]),.1)[1]==0


def test_full_runner_resume():
    import copy
    torch.set_num_threads(1);torch.manual_seed(4)
    def make():
        model=build();items=[obs(v,i) for v in ('SAX','2CH','4CH') for i in range(3)]
        return FullCardiacRunner(model,ViewLocationBalancedSampler(items,seed=12),pixel_samples=4,reg_samples=2,floor=.001)
    first=make();first.update();state=copy.deepcopy(first.state_dict());first.update();expected=copy.deepcopy(first.model.state_dict());second=make();second.load_state_dict(state);second.update();assert all(torch.equal(v,second.model.state_dict()[k]) for k,v in expected.items())
    stage3=FullCardiacRunner(build(),ViewLocationBalancedSampler([obs(v,i) for v in ('SAX','2CH','4CH') for i in range(3)]),pixel_samples=4,reg_samples=2,floor=.001,stage=3);assert stage3.optimizer.param_groups[1]['lr']==5e-6
