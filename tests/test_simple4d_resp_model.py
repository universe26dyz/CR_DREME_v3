import torch
from torch import nn
from cardioresp4d.simple4d.model import Simple4DRespiratoryModel
from cardioresp4d.training.sampler import DynamicObservation
from third_party.simple4d.simple4d.networks import DeformCodeAdaptive


class Canonical(nn.Module):
    def __init__(self): super().__init__(); self.weight = nn.Parameter(torch.tensor(1.))
    def forward(self, x, return_features=False): return x[...,0] * self.weight, x


def test_model():
    model = Simple4DRespiratoryModel(Canonical(), [-100]*3, [100]*3, [1]*3, psf_samples=2)
    assert isinstance(model.forward_deformation.network, DeformCodeAdaptive)
    assert isinstance(model.inverse_deformation.network, DeformCodeAdaptive)
    model.freeze_r0()
    assert not model.canonical.weight.requires_grad
    assert all(p.requires_grad for module in (model.respiratory_encoder, model.forward_deformation, model.inverse_deformation) for p in module.parameters())
    assert not hasattr(model, 'cardiac_mbc') and not hasattr(model, 'respiratory_mbc')
    obs = DynamicObservation(image=torch.randn(1,64,64), view='SAX', slice_id='a', dynamic_frame_id=0, center_mm=torch.zeros(3), row_direction=torch.tensor([1.,0,0]), column_direction=torch.tensor([0.,1,0]), normal=torch.tensor([0.,0,1]), pixel_spacing_mm=torch.ones(2), slice_thickness_mm=1., qc_valid=True, qc_reason='', normalized_slice_position=0.)
    model.eval()
    out = model.predict(obs, torch.tensor([[20.,20.],[30.,30.]]))
    assert out['predicted_intensity'].shape == (2,)
    assert out['reference_samples_world_mm'].shape == (2,2,3)
    assert model.provenance()['reference_spacing_mm'] == [1.,1.,1.]


def test_psf_zero_motion():
    from cardioresp4d.adapters.nesvor_psf import NeSVoRPSFAdapter
    class Zero(nn.Module):
        def forward(self, x): return {'reference_points_mm': x}
    points = torch.zeros(2,3)
    basis = torch.eye(3)
    out = NeSVoRPSFAdapter(3)(Canonical(), points, torch.ones(2,3), row_direction=basis[0].expand(2,-1), column_direction=basis[1].expand(2,-1), normal=basis[2].expand(2,-1), motion=Zero())
    assert torch.equal(out['reference_samples_world_mm'], out['observation_samples_world_mm'])
    assert out['predicted_intensity'].shape == (2,)
