"""Training semantics added by v3_change1: no hidden stage or sampling fallback."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.models.cardioresp_motion import SequentialPullbackMotion
from cardioresp4d.training.model import SourceFirstDynamicModel
from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer


def observation(view: str, location: str, frame: int, timestamp: float) -> DynamicObservation:
    return DynamicObservation(torch.rand(1, 8, 8), view, location, frame, torch.zeros(3), torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.]), torch.ones(2), 4., True, "not_evaluated", timestamp)


def tiny_model() -> SourceFirstDynamicModel:
    return SourceFirstDynamicModel(torch.tensor([-10., -10., -10.]), torch.tensor([10., 10., 10.]), cardiac_lower_world_mm=torch.tensor([-2., -2., -2.]), cardiac_upper_world_mm=torch.tensor([2., 2., 2.]), n_dynamic_frames=6, inr_width=8, inr_depth=1, latent_dim=4, motion_hidden_dim=8, respiratory_grid_shapes=((4, 4, 4), (5, 5, 5), (6, 6, 6)), cardiac_grid_shape=(4, 4, 4), psf_samples=1)


class V3Change1TrainingTest(unittest.TestCase):
    def test_true_timestamp_batch_and_stage_freeze_are_real(self) -> None:
        items = [observation("SAX", "same", index, timestamp) for index, timestamp in enumerate((.0, .12, .37, .61))] + [observation("2CH", "two", 4, .0), observation("4CH", "four", 5, .0)]
        trainer = UnifiedProgressiveTrainer(tiny_model(), ViewLocationBalancedSampler(items, seed=3), pixel_samples=2, temporal_batch_size=4)
        self.assertEqual([.0, .12, .37, .61], [item.timestamp_s for item in trainer.sampler.temporal_batch(max_items=4)])
        film_before = [parameter.detach().clone() for parameter in trainer.model.film_encoder.parameters()]
        stage1 = trainer.run_stage("stage1", steps=1)
        self.assertFalse(stage1["gradient_non_none"]["film"])
        self.assertTrue(all(torch.equal(before, after) for before, after in zip(film_before, trainer.model.film_encoder.parameters())))
        stage2a = trainer.run_stage("stage2a", steps=1)
        self.assertTrue(stage2a["gradient_non_none"]["film"])
        self.assertTrue(stage2a["gradient_non_none"]["respiratory_sinr"])

    def test_cardiac_priority_sampling_and_authoritative_pullback(self) -> None:
        model = tiny_model(); trainer = UnifiedProgressiveTrainer(model, ViewLocationBalancedSampler([observation("SAX", "s", 0, 0.), observation("2CH", "t", 1, 0.), observation("4CH", "f", 2, 0.)]), pixel_samples=50, cardiac_sampling_fraction=.8)
        item = observation("SAX", "s", 0, 0.)
        ratios = []
        for _ in range(20):
            uv = trainer.sample_pixels(item); points = model._pixel_world(item, uv)
            ratios.append(float(((points >= model.cardiac_lower_world_mm) & (points <= model.cardiac_upper_world_mm)).all(-1).float().mean()))
        self.assertGreater(sum(ratios) / len(ratios), .78)
        class Field(torch.nn.Module):
            def __init__(self, value): super().__init__(); self.value = torch.tensor(value)
            def forward(self, x): return torch.ones_like(x) * self.value.to(x)
        result = SequentialPullbackMotion(Field([1., 0., 0.]), Field([0., 2., 0.]))(torch.zeros(1, 2, 3))
        torch.testing.assert_close(result["cardiac_points_mm"], torch.tensor([[[0., 2., 0.], [0., 2., 0.]]]))
        torch.testing.assert_close(result["reference_points_mm"], torch.tensor([[[1., 2., 0.], [1., 2., 0.]]]))


if __name__ == "__main__":
    unittest.main()
