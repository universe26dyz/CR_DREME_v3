"""Deterministic P0 mathematics checks for Phase-2 foundation modules."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.models.hash_inr import CanonicalINR  # noqa: E402
from cardioresp4d.models.bspline_mbc import CardiacMBC, RespiratoryMBC, dvf_from_scores  # noqa: E402
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField, SequentialPullbackMotion  # noqa: E402
from cardioresp4d.models.uncertainty import SliceUncertainty  # noqa: E402
from cardioresp4d.rendering.thick_slice_renderer import ThickSliceRenderer  # noqa: E402


class _ConstantInr(nn.Module):
    def forward(self, xyz, return_features=False):
        value = xyz.new_ones((*xyz.shape[:-1], 1))
        return (value, xyz) if return_features else value


class Phase2CoreTest(unittest.TestCase):
    def test_hash_inr_queries_continuously_and_backpropagates(self):
        model = CanonicalINR(levels=3, features_per_level=2, hash_size=64, min_resolution=4, max_resolution=16, hidden_dim=16, latent_dim=8)
        xyz = torch.zeros(2, 5, 3, requires_grad=True)
        intensity, latent = model(xyz, return_features=True)
        self.assertEqual((2, 5, 1), tuple(intensity.shape)); self.assertEqual((2, 5, 8), tuple(latent.shape))
        intensity.mean().backward(); self.assertTrue(torch.isfinite(xyz.grad).all())

    def test_bspline_zero_score_and_cardiac_boundary_are_zero_in_mm(self):
        lower, upper = torch.tensor([-10., -10., -10.]), torch.tensor([10., 10., 10.])
        resp = RespiratoryMBC(lower, upper, resolutions=(4, 5, 6))
        card = CardiacMBC(lower, upper, resolution=4)
        points = torch.tensor([[[-10., 0., 0.], [0., 0., 0.]]], requires_grad=True)
        fields = resp(points); self.assertEqual((1, 3, 2, 3), tuple(fields.shape))
        self.assertTrue(torch.equal(dvf_from_scores(fields, torch.zeros(1, 3, 3)), torch.zeros(1, 2, 3)))
        self.assertTrue(torch.equal(card(points)[:, 0], torch.zeros(1, 3)))
        card(points).sum().backward(); self.assertTrue(torch.isfinite(points.grad).all())

    def test_score_weighted_mbc_field_has_mm_component_semantics(self):
        lower, upper = torch.tensor([-10., -10., -10.]), torch.tensor([10., 10., 10.])
        mbc = RespiratoryMBC(lower, upper, resolutions=(4, 4, 4))
        with torch.no_grad():
            mbc.levels[0].controls[0].fill_(2.0)
        field = ScoreWeightedMBCField(mbc, torch.tensor([[[0.5, 0., 0.], [0., 0., 0.], [0., 0., 0.]]]))
        value = field(torch.zeros(1, 2, 3))
        self.assertTrue(torch.allclose(value[..., 0], torch.ones(1, 2), atol=1e-6))
        self.assertTrue(torch.allclose(value[..., 1:], torch.zeros(1, 2, 2), atol=1e-6))

    def test_sequential_pullback_evaluates_respiratory_field_after_cardiac_step(self):
        class Card(nn.Module):
            def forward(self, points): return torch.tensor([0., 1., 0.], device=points.device).expand_as(points)
        class Resp(nn.Module):
            def forward(self, points): return torch.stack((points[..., 1], torch.zeros_like(points[..., 1]), torch.zeros_like(points[..., 1])), -1)
        motion = SequentialPullbackMotion(Resp(), Card())
        result = motion(torch.zeros(1, 1, 3))
        torch.testing.assert_close(result["reference_points_mm"], torch.tensor([[[1., 1., 0.]]]))

    def test_five_point_renderer_and_variance_are_finite_with_gradients(self):
        renderer = ThickSliceRenderer(torch.eye(4))
        output = renderer(_ConstantInr(), center_mm=torch.zeros(1, 3), row_direction=torch.tensor([[1., 0., 0.]]),
                          column_direction=torch.tensor([[0., 1., 0.]]), normal=torch.tensor([[0., 0., 1.]]),
                          pixel_spacing_mm=torch.tensor([[1., 1.]]), thickness_mm=torch.tensor([8.]), height=3, width=4)
        torch.testing.assert_close(output["predicted_slice"], torch.ones(1, 1, 3, 4))
        z = output["latent_samples"].detach().requires_grad_()
        uncertainty = SliceUncertainty(latent_dim=3, num_frames=2, embedding_dim=2)
        variance = uncertainty(z, torch.tensor([1]), output["quadrature_weights"])
        self.assertEqual((1, 3, 4), tuple(variance["total_variance"].shape)); variance["total_variance"].mean().backward()
        self.assertTrue(torch.isfinite(z.grad).all())


if __name__ == "__main__": unittest.main()
