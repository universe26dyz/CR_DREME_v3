"""Optional real CUDA regression for the adapter-only SINR device fix."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.adapters.sinr_mbc import SINRFFDBasis  # noqa: E402


class Change4CudaTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable on this local validation host")
    def test_sinr_grid_spacing_and_upstream_parameters_follow_cuda_bounds(self) -> None:
        device = torch.device("cuda")
        basis = SINRFFDBasis(torch.tensor([-2., -2., -2.], device=device), torch.tensor([2., 2., 2.], device=device), logical_control_shape=(4, 4, 4), hidden_dim=8).to(device)
        self.assertEqual("cuda", basis.grid_spacing_mm.device.type)
        self.assertTrue(all(parameter.device.type == "cuda" for parameter in basis.siren.parameters()))
        with torch.no_grad():
            output = basis(torch.zeros(1, 1, 3, device=device))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
