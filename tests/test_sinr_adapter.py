"""CPU contracts for the external pinned SINR adapter."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.adapters.sinr_mbc import CardiacSINRMBCAdapter, SINRFFDBasis  # noqa: E402


class SINRAdapterTest(unittest.TestCase):
    def test_adapter_holds_external_upstream_classes_and_zero_controls_are_zero_mm(self) -> None:
        basis = SINRFFDBasis(torch.tensor([-8., -12., -16.]), torch.tensor([8., 12., 16.]), grid_shape=(8, 8, 8), cps=(2, 2, 2), hidden_dim=8)
        from networks.networks import BSplineSiren
        from models.transformation import CubicBSplineFFDTransform
        self.assertIsInstance(basis.siren, BSplineSiren)
        self.assertIsInstance(basis.ffd, CubicBSplineFFDTransform)
        controls = torch.zeros(1, 9, *basis.control_shape)
        torch.testing.assert_close(basis.dense_dvf_mm(controls), torch.zeros(1, 9, 8, 8, 8))

    def test_adapter_matches_upstream_ffd_with_explicit_mm_conversion(self) -> None:
        basis = SINRFFDBasis(torch.tensor([-8., -12., -16.]), torch.tensor([8., 12., 16.]), grid_shape=(8, 8, 8), cps=(2, 2, 2), hidden_dim=8)
        controls = torch.randn(1, 9, *basis.control_shape)
        expected = basis.ffd(controls).reshape(1, 3, 3, 8, 8, 8) * basis.grid_spacing_mm.view(1, 1, 3, 1, 1, 1)
        actual = basis.dense_dvf_mm(controls).reshape(1, 3, 3, 8, 8, 8)
        torch.testing.assert_close(actual, expected)

    def test_cardiac_boundary_is_zero_and_siren_and_controls_receive_gradients(self) -> None:
        cardiac = CardiacSINRMBCAdapter(torch.tensor([-10., -10., -10.]), torch.tensor([10., 10., 10.]), grid_shape=(8, 8, 8), cps=2, hidden_dim=8, taper_mm=2.)
        boundary = torch.tensor([[[10., 0., 0.]]])
        torch.testing.assert_close(cardiac(boundary), torch.zeros(1, 1, 3, 3))
        points = torch.tensor([[[0., 0., 0.], [1., 1., 1.]]], requires_grad=True)
        value = cardiac(points)
        value.sum().backward()
        self.assertIsNotNone(points.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in cardiac.basis.siren.parameters()))


if __name__ == "__main__":
    unittest.main()
