"""Identity and numerical contracts for source-first adapters.

These tests intentionally import the pinned vendored classes rather than
testing a local imitation with matching output shapes.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.adapters.film import FiLMAdapter  # noqa: E402
from cardioresp4d.adapters.nesvor_inr import NeSVoRCanonicalAdapter  # noqa: E402
from cardioresp4d.adapters.nesvor_psf import NeSVoRPSFAdapter  # noqa: E402
from cardioresp4d.adapters.nesvor_uncertainty import NeSVoRDynamicFrameUncertainty  # noqa: E402


class SourceBackedAdapterTest(unittest.TestCase):
    def test_nesvor_adapter_wraps_the_pinned_upstream_inr_and_preserves_forward(self) -> None:
        adapter = NeSVoRCanonicalAdapter(torch.tensor([[-10., -10., -10.], [10., 10., 10.]]), width=8, depth=1, n_features_z=4, finest_resolution=4.)
        from nesvor.inr.models import INR
        self.assertIsInstance(adapter.inr, INR)
        points = torch.zeros(3, 3)
        direct_density, _, direct_z = adapter.inr(points)
        density, z = adapter(points, return_features=True)
        torch.testing.assert_close(density, direct_density)
        torch.testing.assert_close(z, direct_z[..., 1:].reshape(*density.shape, -1))

    def test_psf_sigma_and_zero_motion_sampling_match_upstream(self) -> None:
        from nesvor.utils import resolution2sigma
        psf = NeSVoRPSFAdapter(n_samples=3)
        resolution = torch.tensor([[1.5, 2.0, 8.0]])
        torch.testing.assert_close(psf.sigma_mm(resolution), resolution2sigma(resolution, isotropic=False))
        points = torch.zeros(2, 3)
        directions = {"row_direction": torch.tensor([[1., 0., 0.]]).expand(2, -1), "column_direction": torch.tensor([[0., 1., 0.]]).expand(2, -1), "normal": torch.tensor([[0., 0., 1.]]).expand(2, -1)}
        torch.manual_seed(7); adapter_samples = psf.sample(points, resolution, **directions)
        torch.manual_seed(7); direct_samples = torch.randn(2, 3, 3) * resolution2sigma(resolution, isotropic=False).view(-1, 1, 3) + points[:, None]
        torch.testing.assert_close(adapter_samples, direct_samples)

    def test_official_psf_sampling_preserves_a_constant_canonical_field(self) -> None:
        class ConstantCanonical(torch.nn.Module):
            def __init__(self, inr): super().__init__(); self.inr = inr
            def forward(self, points, return_features=False):
                intensity = torch.ones(points.shape[:-1], dtype=points.dtype, device=points.device)
                latent = torch.zeros(*points.shape[:-1], 4, dtype=points.dtype, device=points.device)
                return (intensity, latent) if return_features else intensity
        inr = NeSVoRCanonicalAdapter(torch.tensor([[-10., -10., -10.], [10., 10., 10.]]), width=8, depth=1, n_features_z=4)
        output = NeSVoRPSFAdapter(n_samples=3)(ConstantCanonical(inr.inr), torch.zeros(2, 3), torch.tensor([[1., 1., 6.]]), row_direction=torch.tensor([[1., 0., 0.]]).expand(2, -1), column_direction=torch.tensor([[0., 1., 0.]]).expand(2, -1), normal=torch.tensor([[0., 0., 1.]]).expand(2, -1))
        torch.testing.assert_close(output["predicted_intensity"], torch.ones(2))

    def test_uncertainty_uses_dynamic_frame_ids_and_matches_torch_nll(self) -> None:
        module = NeSVoRDynamicFrameUncertainty(latent_dim=4, n_dynamic_frames=2, frame_embedding_dim=3, width=8, depth=1)
        latent = torch.randn(2, 4, requires_grad=True)
        result = module(latent, torch.tensor([0, 1]))
        self.assertTrue(torch.all(result["variance"] > 0))
        target = torch.zeros(2); prediction = torch.ones(2)
        torch.testing.assert_close(module.nll(prediction, target, torch.ones(2)), torch.nn.GaussianNLLLoss()(prediction, target, torch.ones(2)))
        result["variance"].sum().backward(); self.assertIsNotNone(module.log_var_frame.grad)

    def test_image_regularization_delegates_to_nesvor_semantics(self) -> None:
        adapter = NeSVoRCanonicalAdapter(torch.tensor([[-10., -10., -10.], [10., 10., 10.]]), width=8, depth=1, n_features_z=4)
        points = torch.randn(3, 2, 3); density, _ = adapter(points, return_features=True)
        regularization = adapter.image_regularization(density, points, mode="edge")
        self.assertTrue(torch.isfinite(regularization))

    def test_film_adapter_wraps_upstream_primitive(self) -> None:
        module = FiLMAdapter()
        from vr.models.filmed_net import FiLM
        self.assertIsInstance(module.film, FiLM)
        feature = torch.randn(2, 3, 4, 4)
        gamma = torch.randn(2, 3); beta = torch.randn(2, 3)
        torch.testing.assert_close(module(feature, gamma, beta), module.film(feature, gamma, beta))


if __name__ == "__main__":
    unittest.main()
