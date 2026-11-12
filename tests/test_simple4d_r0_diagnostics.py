import torch
from cardioresp4d.experiments.simple4d_resp_v1_r0 import decompose, dynamic_pca, diagnostic_metrics
from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler
from torch import nn

class Canonical(nn.Module):
    def __init__(self): super().__init__(); self.weight = nn.Parameter(torch.tensor(1.))
    def forward(self, x, return_features=False): return x[...,0] * self.weight, x
from cardioresp4d.simple4d.model import Simple4DRespiratoryModel


def test_decomposition():
    groups = [torch.tensor([[1.],[3.]]),torch.tensor([[3.],[5.]])]
    out, dynamic = decompose(groups, 'latent_mu')
    assert abs(out['latent_mu_static_rms'] - (10**.5)) < 1e-6
    assert out['latent_mu_dynamic_rms'] == 1
    assert abs(out['latent_mu_static_dynamic_ratio'] - (10**.5)) < 1e-6
    assert dynamic_pca(dynamic)['latent_dynamic_rank_95pct'] == 1
    assert dynamic_pca(torch.zeros(2,32))['latent_dynamic_rank_95pct'] == 0


def test_diagnostics_repeat_and_rng():
    model = Simple4DRespiratoryModel(Canonical(), [-10]*3,[10]*3,[1]*3,psf_samples=2).train()
    observations = [DynamicObservation(image=torch.randn(1,64,64), view=view, slice_id='a', dynamic_frame_id=i, center_mm=torch.zeros(3), row_direction=torch.tensor([1.,0,0]), column_direction=torch.tensor([0.,1,0]), normal=torch.tensor([0.,0,1]), pixel_spacing_mm=torch.ones(2), slice_thickness_mm=1., qc_valid=True, qc_reason='', normalized_slice_position=0.) for i,view in enumerate(('SAX','2CH','4CH'))]
    sampler = ViewLocationBalancedSampler(observations, seed=12)
    before = torch.get_rng_state().clone()
    a = diagnostic_metrics(model,sampler,pixel_count=4)
    b = diagnostic_metrics(model,sampler,pixel_count=4)
    assert torch.equal(before, torch.get_rng_state()) and a == b
    assert a['diagnostic_frame_count'] == 3 and a['jacobian_det_nonpositive_fraction'] >= 0


def test_real_update_with_fake_canonical():
    from cardioresp4d.experiments.simple4d_resp_v1_r0 import R0Runner
    model = Simple4DRespiratoryModel(Canonical(), [-100]*3,[100]*3,[1]*3,psf_samples=2)
    observations = [DynamicObservation(image=torch.rand(1,64,64), view=view, slice_id='a', dynamic_frame_id=i, center_mm=torch.zeros(3), row_direction=torch.tensor([1.,0,0]), column_direction=torch.tensor([0.,1,0]), normal=torch.tensor([0.,0,1]), pixel_spacing_mm=torch.ones(2), slice_thickness_mm=1., qc_valid=True, qc_reason='', normalized_slice_position=0.) for i,view in enumerate(('SAX','2CH','4CH'))]
    runner = R0Runner(model, ViewLocationBalancedSampler(observations,seed=12), observations_per_update=3, pixel_samples=4)
    row = runner.update()
    assert row['global_step'] == 1 and row['regularizer_ramp'] == 0
    assert all(row[name] > 0 for name in ('encoder_grad_norm','forward_deform_grad_norm','inverse_deform_grad_norm'))


def test_analytic_jacobian_and_dvf_decomposition():
    from third_party.simple4d.simple4d.losses import jacobian_matrix
    x = torch.randn(1,8,3,requires_grad=True)
    for displacement, expected in ((x*0 + 0.1,1.), (x*0.1,1.1**3)):
        det = torch.linalg.det(jacobian_matrix(x,displacement))
        assert torch.allclose(det, torch.full_like(det,expected), atol=1e-6)
        assert (det <= 0).sum() == 0
    group = [torch.tensor([[[1.,1.,1.]],[[3.,3.,3.]]]), torch.tensor([[[3.,3.,3.]],[[5.,5.,5.]]])]
    result, _ = decompose(group, 'respiratory_dvf')
    assert abs(result['respiratory_dvf_static_rms'] - (10**.5)) < 1e-6
    assert result['respiratory_dvf_dynamic_rms'] == 1.
