"""CPU math contracts for read-only dynamics visualization helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.visualization.dynamics import jacobian_determinant, pullback_displacement, world_grid  # noqa: E402


class DynamicsVisualizationTest(unittest.TestCase):
    def test_identity_translation_scaling_and_folding_jacobians_use_physical_spacing(self) -> None:
        grid, spacing = world_grid(torch.zeros(3), torch.ones(3), (5, 5, 5))
        identity = grid
        translation = grid + torch.tensor([3., -2., .5])
        scaled = grid * torch.tensor([2., 3., 4.])
        folded = grid * torch.tensor([-1., 1., 1.])
        for field in (identity, translation): self.assertTrue(torch.allclose(jacobian_determinant(field, spacing), torch.ones(5, 5, 5), atol=1e-5))
        self.assertTrue(torch.allclose(jacobian_determinant(scaled, spacing), torch.full((5, 5, 5), 24.), atol=1e-5))
        self.assertLess(float(jacobian_determinant(folded, spacing).max()), 0.)

    def test_total_displacement_is_reference_minus_observation(self) -> None:
        observation = torch.randn(2, 3, 3)
        reference = observation + .25
        self.assertTrue(torch.equal(reference - observation, pullback_displacement(observation, reference)))


if __name__ == "__main__":
    unittest.main()
