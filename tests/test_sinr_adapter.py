"""CPU contracts for the external pinned SINR adapter."""
from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.adapters.sinr_mbc import CardiacSINRMBCAdapter, SINRFFDBasis  # noqa: E402
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField  # noqa: E402


class SINRAdapterTest(unittest.TestCase):
    def test_score_weighted_mbc_collapses_intentionally_broadcast_point_batch(self) -> None:
        class RecordingMBC(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.calls = []

            def forward(self, points):
                self.calls.append(tuple(points.shape))
                return torch.stack((points, points * 2.), dim=1)

        mbc = RecordingMBC()
        points = torch.randn(37, 8, 3)
        output = ScoreWeightedMBCField(mbc, torch.ones(1, 2, 3))(points)
        self.assertEqual(mbc.calls, [(1, 37 * 8, 3)])
        self.assertEqual(tuple(output.shape), (37, 8, 3))

    def test_score_weighted_mbc_preserves_true_multi_score_batch(self) -> None:
        class RecordingMBC(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.calls = []

            def forward(self, points):
                self.calls.append(tuple(points.shape))
                return points[:, None].expand(-1, 2, -1, -1)

        mbc = RecordingMBC()
        ScoreWeightedMBCField(mbc, torch.ones(37, 2, 3))(torch.randn(37, 8, 3))
        self.assertEqual(mbc.calls, [(37, 8, 3)])

    def test_score_weighted_mbc_broadcast_values_and_gradients_match_explicit_batch(self) -> None:
        from cardioresp4d.adapters.sinr_mbc import RespiratorySINRMBCAdapter

        torch.manual_seed(4)
        lower, upper = torch.tensor([-2., -2., -2.]), torch.tensor([2., 2., 2.])
        optimized_mbc = RespiratorySINRMBCAdapter(lower, upper, logical_control_shapes=((2, 2, 2),) * 3, cps=2, hidden_dim=4)
        optimized_mbc.set_active_levels(3)
        reference_mbc = deepcopy(optimized_mbc)
        points = (torch.randn(3, 4, 3) * .5).requires_grad_()
        scores = torch.randn(1, 3, 3, requires_grad=True)
        reference_points = points.detach().clone().requires_grad_()
        reference_scores = scores.detach().expand(3, -1, -1).clone().requires_grad_()
        optimized = ScoreWeightedMBCField(optimized_mbc, scores)(points)
        reference = ScoreWeightedMBCField(reference_mbc, reference_scores)(reference_points)
        torch.testing.assert_close(optimized, reference, rtol=1e-5, atol=1e-6)
        (optimized.square().mean()).backward()
        (reference.square().mean()).backward()
        torch.testing.assert_close(points.grad, reference_points.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(scores.grad, reference_scores.grad.sum(0, keepdim=True), rtol=1e-5, atol=1e-6)
        for left, right in zip(optimized_mbc.parameters(), reference_mbc.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)

    def test_adapter_holds_external_upstream_classes_and_zero_controls_are_zero_mm(self) -> None:
        basis = SINRFFDBasis(torch.tensor([-8., -12., -16.]), torch.tensor([8., 12., 16.]), grid_shape=(8, 8, 8), cps=(2, 2, 2), hidden_dim=8)
        from networks.networks import BSplineSiren
        from models.transformation import CubicBSplineFFDTransform
        self.assertIsInstance(basis.siren, BSplineSiren)
        self.assertIsInstance(basis.ffd, CubicBSplineFFDTransform)
        controls = torch.zeros(1, 3, *basis.control_shape)
        torch.testing.assert_close(basis.dense_dvf_mm(controls), torch.zeros(1, 3, 16, 16, 16))

    def test_adapter_matches_upstream_ffd_with_explicit_mm_conversion(self) -> None:
        basis = SINRFFDBasis(torch.tensor([-8., -12., -16.]), torch.tensor([8., 12., 16.]), grid_shape=(8, 8, 8), cps=(2, 2, 2), hidden_dim=8)
        controls = torch.randn(1, 3, *basis.control_shape)
        expected = basis.ffd(controls) * basis.grid_spacing_mm.view(1, 3, 1, 1, 1)
        actual = basis.dense_dvf_mm(controls)
        torch.testing.assert_close(actual, expected)

    def test_cardiac_boundary_is_zero_and_siren_and_controls_receive_gradients(self) -> None:
        cardiac = CardiacSINRMBCAdapter(torch.tensor([-10., -10., -10.]), torch.tensor([10., 10., 10.]), grid_shape=(8, 8, 8), cps=2, hidden_dim=8, taper_mm=2.)
        boundary = torch.tensor([[[10., 0., 0.]]])
        torch.testing.assert_close(cardiac(boundary), torch.zeros(1, 1, 3))
        points = torch.tensor([[[0., 0., 0.], [1., 1., 1.]]], requires_grad=True)
        value = cardiac(points)
        value.sum().backward()
        self.assertIsNotNone(points.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in cardiac.basis.siren.parameters()))


if __name__ == "__main__":
    unittest.main()
