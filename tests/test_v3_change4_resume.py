"""Checkpoint compatibility tests for the v3_change4 progressive schedule."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.training.model import SourceFirstDynamicModel  # noqa: E402
from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler  # noqa: E402
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer  # noqa: E402


def _model() -> SourceFirstDynamicModel:
    return SourceFirstDynamicModel(torch.tensor([-4., -4., -4.]), torch.tensor([4., 4., 4.]), cardiac_lower_world_mm=torch.tensor([-2., -2., -2.]), cardiac_upper_world_mm=torch.tensor([2., 2., 2.]), n_dynamic_frames=3, inr_width=8, inr_depth=1, latent_dim=4, motion_hidden_dim=8, respiratory_grid_shapes=((4, 4, 4),) * 3, cardiac_grid_shape=(4, 4, 4), psf_samples=1)


def _rows() -> list[DynamicObservation]:
    result = []
    for index, view in enumerate(("SAX", "2CH", "4CH")):
        result.append(DynamicObservation(torch.rand(1, 4, 4), view, "s", index, torch.zeros(3), torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.]), torch.ones(2), 4., True, "valid", float(index)))
    return result


class Change4ResumeTest(unittest.TestCase):
    def test_stage2c_checkpoint_loads_then_advances_to_stage3a(self) -> None:
        first = UnifiedProgressiveTrainer(_model(), ViewLocationBalancedSampler(_rows()), pixel_samples=1)
        first.run_stage("stage1", steps=1); first.run_stage("stage2a", steps=1); first.run_stage("stage2b", steps=1); first.run_stage("stage2c", steps=1)
        state = first.training_state_dict()
        resumed = UnifiedProgressiveTrainer(_model(), ViewLocationBalancedSampler(_rows()), pixel_samples=1)
        resumed.load_training_state_dict(state)
        report = resumed.run_stage("stage3a", steps=1)
        self.assertEqual(5, report["global_step"])
        self.assertEqual("stage3a", resumed.current_stage)

    def test_legacy_stage3_checkpoint_fails_without_silent_remap(self) -> None:
        trainer = UnifiedProgressiveTrainer(_model(), ViewLocationBalancedSampler(_rows()), pixel_samples=1)
        with self.assertRaisesRegex(ValueError, "legacy v3 Stage3 checkpoint"):
            trainer.load_training_state_dict({"current_stage": "stage3", "global_step": 400, "stage_step": 100, "optimizer": {}, "sampler": {}, "metrics": []})


if __name__ == "__main__":
    unittest.main()
