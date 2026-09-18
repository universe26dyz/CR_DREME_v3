"""Change5A cardiac target-band concentration contracts."""
from __future__ import annotations

import math
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.losses.frequency_loss import (  # noqa: E402
    cardiac_target_band_concentration,
    non_dc_frequency_grid,
    resolve_target_frequency_mask,
)
from cardioresp4d.training.source_first_config import validate_source_first_config  # noqa: E402
from cardioresp4d.training.model import SourceFirstDynamicModel  # noqa: E402
from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler  # noqa: E402
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer  # noqa: E402


def _scores(frequency_hz: float, timestamps_s: torch.Tensor, *, amplitude: float = 1.) -> torch.Tensor:
    signal = amplitude * torch.sin(2 * torch.pi * frequency_hz * timestamps_s)
    return signal[:, None, None].repeat(1, 1, 3)


class CardiacTargetBandConcentrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.times = torch.arange(50, dtype=torch.float64) * .2
        self.grid = non_dc_frequency_grid(self.times)
        self.target = float(self.grid[5])
        self.target_band = [(self.target - .01, self.target + .01)]

    def test_target_sinusoid_is_preferred_over_off_target_sinusoid(self) -> None:
        target = cardiac_target_band_concentration(_scores(self.target, self.times), self.times, self.target_band)
        off_target = cardiac_target_band_concentration(_scores(float(self.grid[12]), self.times), self.times, self.target_band)
        self.assertGreater(float(target["fraction"]), float(off_target["fraction"]))
        self.assertLess(float(target["loss"]), float(off_target["loss"]))

    def test_common_score_scaling_does_not_change_fraction(self) -> None:
        scores = _scores(self.target, self.times) + _scores(float(self.grid[12]), self.times, amplitude=.35)
        first = cardiac_target_band_concentration(scores, self.times, self.target_band)
        scaled = cardiac_target_band_concentration(scores * 17., self.times, self.target_band)
        self.assertAlmostEqual(float(first["fraction"]), float(scaled["fraction"]), places=6)

    def test_power_is_aggregated_over_channels_before_ratio(self) -> None:
        scores = torch.cat((_scores(self.target, self.times, amplitude=10.)[..., :1], _scores(float(self.grid[12]), self.times)[..., :1]), dim=-1)
        result = cardiac_target_band_concentration(scores, self.times, self.target_band)
        self.assertGreater(float(result["fraction"]), .95)
        self.assertLess(float(result["loss"]), .05)

    def test_zero_scores_are_finite_with_loss_one(self) -> None:
        result = cardiac_target_band_concentration(torch.zeros(50, 1, 3), self.times, self.target_band)
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertAlmostEqual(1., float(result["loss"]), places=7)

    def test_gradient_is_finite(self) -> None:
        scores = _scores(self.target, self.times).float().requires_grad_()
        loss = cardiac_target_band_concentration(scores, self.times.float(), self.target_band)["loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_grid_excludes_dc_and_respects_timestamp_nyquist(self) -> None:
        grid = non_dc_frequency_grid(self.times)
        self.assertGreater(float(grid.min()), 0.)
        self.assertLessEqual(float(grid.max()), .5 / .2 + 1e-8)

    def test_target_mask_uses_nearest_grid_point_for_valid_discretization_mismatch(self) -> None:
        grid = non_dc_frequency_grid(self.times)
        mask = resolve_target_frequency_mask(grid, [(float(grid[5]) + .011, float(grid[5]) + .012)])
        self.assertEqual(1, int(mask.sum()))
        self.assertTrue(bool(mask[5]))

    def test_malformed_or_out_of_range_bands_fail_fast(self) -> None:
        grid = non_dc_frequency_grid(self.times)
        for band in ([(float("nan"), 1.)], [(2., 1.)], [(3., 4.)], [(0., .1)]):
            with self.assertRaises(ValueError):
                resolve_target_frequency_mask(grid, band)

    def test_distinct_location_bands_select_distinct_targets(self) -> None:
        first, second = float(self.grid[4]), float(self.grid[10])
        first_score, second_score = _scores(first, self.times), _scores(second, self.times)
        first_local = cardiac_target_band_concentration(first_score, self.times, [(first - .01, first + .01)])
        second_local = cardiac_target_band_concentration(second_score, self.times, [(second - .01, second + .01)])
        wrong_local = cardiac_target_band_concentration(first_score, self.times, [(second - .01, second + .01)])
        self.assertGreater(float(first_local["fraction"]), float(wrong_local["fraction"]))
        self.assertGreater(float(second_local["fraction"]), float(wrong_local["fraction"]))

    def test_change4_config_missing_weight_remains_valid_and_change5a_validates(self) -> None:
        baseline = yaml.safe_load((ROOT / "configs" / "source_first.yaml").read_text())
        validate_source_first_config(baseline, ROOT)
        change5a = yaml.safe_load((ROOT / "configs" / "source_first_change5a.yaml").read_text())
        validate_source_first_config(change5a, ROOT)
        self.assertEqual(.001, change5a["training"]["loss_weights"]["cardiac_target_concentration"])

    def test_invalid_change5a_weight_is_rejected(self) -> None:
        config = yaml.safe_load((ROOT / "configs" / "source_first.yaml").read_text())
        config["training"]["loss_weights"]["cardiac_target_concentration"] = -1.
        with self.assertRaises(ValueError):
            validate_source_first_config(config, ROOT)
        config["training"]["loss_weights"]["cardiac_target_concentration"] = math.nan
        with self.assertRaises(ValueError):
            validate_source_first_config(config, ROOT)

    def test_only_cardiac_stages_expose_and_apply_concentration(self) -> None:
        model = SourceFirstDynamicModel(torch.tensor([-4., -4., -4.]), torch.tensor([4., 4., 4.]), cardiac_lower_world_mm=torch.tensor([-2., -2., -2.]), cardiac_upper_world_mm=torch.tensor([2., 2., 2.]), n_dynamic_frames=9, inr_width=8, inr_depth=1, latent_dim=4, motion_hidden_dim=8, respiratory_grid_shapes=((4, 4, 4),) * 3, cardiac_grid_shape=(4, 4, 4), psf_samples=1)
        items = []
        for view in ("SAX", "2CH", "4CH"):
            for frame in range(3):
                items.append(DynamicObservation(torch.rand(1, 4, 4), view, "s", len(items), torch.zeros(3), torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.]), torch.ones(2), 4., True, "valid", float(frame)))
        local = SimpleNamespace(cardiac_bands_hz=[(.45, .55)], respiratory_bands_hz=[(.1, .2)], cardiac_baseline_pairs=[{"cardiac_band_hz": [.45, .55], "baseline_band_hz": [.25, .35]}])
        trainer = UnifiedProgressiveTrainer(model, ViewLocationBalancedSampler(items, seed=0), pixel_samples=1, frequency_prior=SimpleNamespace(for_location=lambda view, slice_id: local), loss_weights={"cardiac_target_concentration": .1})
        self.assertEqual(0., UnifiedProgressiveTrainer(model, ViewLocationBalancedSampler(items, seed=1), pixel_samples=1).loss_weights["cardiac_target_concentration"])
        self.assertNotIn("cardiac_target_concentration", trainer._temporal_components("stage2a"))
        self.assertIn("cardiac_target_concentration", trainer._temporal_components("stage3a"))
        report = trainer.run_stage("stage3a", steps=1)
        self.assertIn("cardiac_target_concentration", report["loss_components"])
        self.assertTrue(report["gradient_non_none"]["cardiac_sinr"])


if __name__ == "__main__":
    unittest.main()
