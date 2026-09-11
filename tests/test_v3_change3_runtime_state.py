"""Red/green identity, hard-QC and reproducibility contracts."""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.training.runtime_state import capture_rng_state, restore_rng_state, set_reproducibility, validate_dynamic_frame_rows  # noqa: E402
from cardioresp4d.training.model import SourceFirstDynamicModel  # noqa: E402
from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler  # noqa: E402
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer  # noqa: E402


class RuntimeStateTest(unittest.TestCase):
    def test_only_hard_invalid_qc_reason_is_accepted_and_duplicate_tokens_reject(self) -> None:
        valid = [{"source_file_token": "a", "view": "SAX", "slice_id": "s", "frame_index": "0", "timestamp_s": "1.0", "qc_valid": "True", "qc_reason": "low_ncc"}]
        validate_dynamic_frame_rows(valid)
        with self.assertRaisesRegex(ValueError, "hard-invalid"):
            validate_dynamic_frame_rows([{**valid[0], "qc_valid": "False"}])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_dynamic_frame_rows(valid + [{**valid[0], "frame_index": "1"}])
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_dynamic_frame_rows([{**valid[0], "timestamp_s": "nan"}])

    def test_rng_capture_restore_reproduces_python_numpy_and_torch(self) -> None:
        set_reproducibility(17)
        state = capture_rng_state()
        expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
        restore_rng_state(state)
        actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
        self.assertEqual(expected, actual)

    def test_stage_checkpoint_restores_optimizer_and_sampler_state(self) -> None:
        def item(view, index):
            return DynamicObservation(torch.rand(1, 3, 3), view, view, index, torch.zeros(3), torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.]), torch.ones(2), 4., True, "valid", float(index))
        items = [item("SAX", 0), item("2CH", 1), item("4CH", 2)]
        def fresh():
            return SourceFirstDynamicModel(torch.tensor([-4., -4., -4.]), torch.tensor([4., 4., 4.]), cardiac_lower_world_mm=torch.tensor([-2., -2., -2.]), cardiac_upper_world_mm=torch.tensor([2., 2., 2.]), n_dynamic_frames=3, inr_width=8, inr_depth=1, latent_dim=4, motion_hidden_dim=8, respiratory_grid_shapes=((4, 4, 4), (4, 4, 4), (4, 4, 4)), cardiac_grid_shape=(4, 4, 4), psf_samples=1)
        set_reproducibility(3)
        original = fresh(); trainer = UnifiedProgressiveTrainer(original, ViewLocationBalancedSampler(items, seed=3), pixel_samples=1)
        trainer.run_stage("stage1", steps=1)
        checkpoint = {"model": original.state_dict(), "trainer": trainer.training_state_dict(), "rng": capture_rng_state()}
        resumed = fresh(); resumed.load_state_dict(checkpoint["model"])
        second = UnifiedProgressiveTrainer(resumed, ViewLocationBalancedSampler(items, seed=99), pixel_samples=1)
        second.load_training_state_dict(checkpoint["trainer"])
        restore_rng_state(checkpoint["rng"])
        expected = trainer.run_stage("stage1", steps=1)["loss_last"]
        restore_rng_state(checkpoint["rng"])
        actual = second.run_stage("stage1", steps=1)["loss_last"]
        self.assertAlmostEqual(expected, actual, places=6)


if __name__ == "__main__":
    unittest.main()
