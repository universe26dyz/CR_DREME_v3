"""Behavioral regression coverage for read-only Change4 diagnostics."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("diagnose_change4_checkpoint", ROOT / "scripts" / "diagnose_change4_checkpoint.py")
assert SPEC and SPEC.loader
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class DiagnoseChange4CheckpointTest(unittest.TestCase):
    def test_dense_nudft_peak_recovers_known_respiratory_and_cardiac_sinusoids(self) -> None:
        times = torch.arange(50, dtype=torch.float64) * .17
        respiratory = torch.sin(2 * torch.pi * .35 * times)[:, None, None]
        cardiac = torch.sin(2 * torch.pi * 1.41 * times)[:, None, None]
        self.assertAlmostEqual(.35, diagnostic.dominant_peak_hz(respiratory, times), delta=.02)
        self.assertAlmostEqual(1.41, diagnostic.dominant_peak_hz(cardiac, times), delta=.02)


if __name__ == "__main__":
    unittest.main()
