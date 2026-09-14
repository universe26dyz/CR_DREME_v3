"""Tiny CPU smoke test for the v3 source-first progressive path."""
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


def _observation(view: str, location: str, frame_id: int, *, valid: bool = True, reason: str = "not_evaluated") -> DynamicObservation:
    return DynamicObservation(image=torch.rand(1, 4, 4), view=view, slice_id=location, dynamic_frame_id=frame_id, center_mm=torch.zeros(3), row_direction=torch.tensor([1., 0., 0.]), column_direction=torch.tensor([0., 1., 0.]), normal=torch.tensor([0., 0., 1.]), pixel_spacing_mm=torch.ones(2), slice_thickness_mm=6., qc_valid=valid, qc_reason=reason)


class UnifiedProgressiveSmokeTest(unittest.TestCase):
    def test_hard_invalid_never_enters_balanced_sampler(self) -> None:
        sampler = ViewLocationBalancedSampler([_observation("SAX", "s", 0), _observation("2CH", "c2", 1), _observation("4CH", "c4", 2), _observation("SAX", "bad", 3, valid=False, reason="manual_exclusion")], seed=0)
        selected = sampler.sample_step()
        self.assertEqual({"SAX", "2CH", "4CH"}, {item.view for item in selected})
        self.assertTrue(all(item.qc_valid and item.qc_reason not in {"manual_exclusion", "slice_local_scale_absolute"} for item in selected))

    def test_stage3c_tiny_pipeline_backpropagates_to_all_source_backed_modules(self) -> None:
        observations = [_observation("SAX", "s", 0), _observation("2CH", "c2", 1), _observation("4CH", "c4", 2)]
        sampler = ViewLocationBalancedSampler(observations, seed=1)
        model = SourceFirstDynamicModel(torch.tensor([-10., -10., -10.]), torch.tensor([10., 10., 10.]), cardiac_lower_world_mm=torch.tensor([-5., -5., -5.]), cardiac_upper_world_mm=torch.tensor([5., 5., 5.]), n_dynamic_frames=3, inr_width=8, inr_depth=1, latent_dim=4, motion_hidden_dim=8, respiratory_grid_shapes=((4, 4, 4), (5, 5, 5), (6, 6, 6)), cardiac_grid_shape=(4, 4, 4), psf_samples=2)
        trainer = UnifiedProgressiveTrainer(model, sampler, pixel_samples=4, learning_rate=1e-3)
        report = trainer.run_stage("stage3c", steps=1)
        self.assertTrue(torch.isfinite(torch.tensor(report["loss_last"])))
        self.assertTrue(report["gradient_non_none"]["inr"])
        self.assertTrue(report["gradient_non_none"]["film"])
        self.assertTrue(report["gradient_non_none"]["respiratory_sinr"])
        self.assertTrue(report["gradient_non_none"]["cardiac_sinr"])
        self.assertTrue(report["gradient_non_none"]["uncertainty"])

    def test_stage1_and_respiratory_progression_keep_zero_motion_then_add_levels(self) -> None:
        observations = [_observation("SAX", "s", 0), _observation("2CH", "c2", 1), _observation("4CH", "c4", 2)]
        model = SourceFirstDynamicModel(torch.tensor([-10., -10., -10.]), torch.tensor([10., 10., 10.]), cardiac_lower_world_mm=torch.tensor([-5., -5., -5.]), cardiac_upper_world_mm=torch.tensor([5., 5., 5.]), n_dynamic_frames=3, inr_width=8, inr_depth=1, latent_dim=4, motion_hidden_dim=8, respiratory_grid_shapes=((4, 4, 4), (5, 5, 5), (6, 6, 6)), cardiac_grid_shape=(4, 4, 4), psf_samples=1)
        sample = model.predict(observations[0], torch.tensor([[1., 1.]]), "stage1")
        torch.testing.assert_close(sample["observation_samples_world_mm"], sample["reference_samples_world_mm"])
        trainer = UnifiedProgressiveTrainer(model, ViewLocationBalancedSampler(observations, seed=2), pixel_samples=2, learning_rate=1e-3)
        for stage in ("stage1", "stage2a", "stage2b", "stage2c"):
            report = trainer.run_stage(stage, steps=1)
            self.assertEqual(stage, report["stage"])


if __name__ == "__main__":
    unittest.main()
