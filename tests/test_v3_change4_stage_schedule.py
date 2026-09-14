"""Regression tests for the v3_change4 Stage3a/3b/3c ownership split."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.training.model import SourceFirstDynamicModel  # noqa: E402
from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler  # noqa: E402
from cardioresp4d.training.stage_contract import stage_contract  # noqa: E402
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer  # noqa: E402


def _model() -> SourceFirstDynamicModel:
    return SourceFirstDynamicModel(
        torch.tensor([-4., -4., -4.]), torch.tensor([4., 4., 4.]),
        cardiac_lower_world_mm=torch.tensor([-2., -2., -2.]), cardiac_upper_world_mm=torch.tensor([2., 2., 2.]),
        n_dynamic_frames=3, inr_width=8, inr_depth=1, latent_dim=4, motion_hidden_dim=8,
        respiratory_grid_shapes=((4, 4, 4), (4, 4, 4), (4, 4, 4)), cardiac_grid_shape=(4, 4, 4), psf_samples=1,
    )


def _item(view: str, frame: int) -> DynamicObservation:
    return DynamicObservation(
        image=torch.rand(1, 4, 4), view=view, slice_id="s", dynamic_frame_id=frame,
        center_mm=torch.zeros(3), row_direction=torch.tensor([1., 0., 0.]),
        column_direction=torch.tensor([0., 1., 0.]), normal=torch.tensor([0., 0., 1.]),
        pixel_spacing_mm=torch.ones(2), slice_thickness_mm=4., qc_valid=True,
        qc_reason="valid", timestamp_s=float(frame),
    )


class Change4StageScheduleTest(unittest.TestCase):
    def test_contract_is_the_single_stage_schedule_source(self) -> None:
        expected = {
            "stage1": (0, False, False, "none", "mse"),
            "stage2a": (1, False, False, "all", "mse"),
            "stage2b": (2, False, False, "all", "mse"),
            "stage2c": (3, False, False, "all", "mse"),
            "stage3a": (3, True, False, "card_head_only", "mse"),
            "stage3b": (3, True, False, "all", "mse"),
            "stage3c": (3, True, True, "all", "gaussian_nll"),
        }
        actual = {
            stage: (contract.active_respiratory_levels, contract.enable_cardiac, contract.enable_uncertainty, contract.film_train_mode, contract.data_term)
            for stage, contract in ((stage, stage_contract(stage)) for stage in expected)
        }
        self.assertEqual(expected, actual)

    def test_stage3a_backward_only_reaches_card_head_and_cardiac_mbc(self) -> None:
        torch.manual_seed(11)
        model = _model()
        trainer = UnifiedProgressiveTrainer(model, ViewLocationBalancedSampler([_item("SAX", 0), _item("2CH", 1), _item("4CH", 2)]), pixel_samples=1)
        trainer.run_stage("stage3a", steps=1)
        self.assertTrue(any(parameter.grad is not None for parameter in model.film_encoder.card.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in model.cardiac_mbc.parameters()))
        for module in (model.canonical, model.respiratory_mbc, model.film_encoder.image, model.film_encoder.film, model.film_encoder.geometry_mlp, model.film_encoder.head, model.film_encoder.resp, model.uncertainty):
            self.assertFalse(any(parameter.grad is not None for parameter in module.parameters()), type(module).__name__)

    def test_stage3_data_term_and_full_ownership_follow_contract(self) -> None:
        items = [_item("SAX", 0), _item("2CH", 1), _item("4CH", 2)]
        for stage, expected_nll in (("stage3a", False), ("stage3b", False), ("stage3c", True)):
            model = _model()
            trainer = UnifiedProgressiveTrainer(model, ViewLocationBalancedSampler(items), pixel_samples=1)
            calls = [0]
            original = model.uncertainty.nll

            def nll(prediction, target, variance):
                calls[0] += 1
                return original(prediction, target, variance)

            model.uncertainty.nll = nll  # type: ignore[method-assign]
            trainer.run_stage(stage, steps=1)
            self.assertEqual(expected_nll, bool(calls[0]), stage)
            self.assertEqual(stage != "stage3a", any(parameter.grad is not None for parameter in model.canonical.parameters()), stage)
            self.assertTrue(any(parameter.grad is not None for parameter in model.cardiac_mbc.parameters()), stage)
            self.assertEqual(stage == "stage3c", any(parameter.grad is not None for parameter in model.uncertainty.parameters()), stage)


if __name__ == "__main__":
    unittest.main()
