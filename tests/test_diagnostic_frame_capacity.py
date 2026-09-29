"""Change5C checkpoint-reconstruction frame-capacity contracts."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from cardioresp4d.training.runtime_state import checkpoint_compatible_dynamic_frame_count, training_dynamic_frame_count, validate_checkpoint_dynamic_frame_capacity


class DiagnosticFrameCapacityTest(unittest.TestCase):
    def test_qc_valid_count_matches_training_contract(self) -> None:
        observations = [SimpleNamespace(qc_valid=index < 7150) for index in range(7200)]
        self.assertEqual(7150, training_dynamic_frame_count(observations))
        self.assertEqual(7150, checkpoint_compatible_dynamic_frame_count(None, observations))

    def test_matching_uncertainty_capacity_passes(self) -> None:
        checkpoint = {"model": {"uncertainty.log_var_frame": torch.zeros(7150), "uncertainty.frame_embedding.weight": torch.zeros(7150, 8)}}
        self.assertEqual(7150, validate_checkpoint_dynamic_frame_capacity(checkpoint, 7150))
        observations = [SimpleNamespace(qc_valid=True) for _ in range(7150)]
        self.assertEqual(7150, checkpoint_compatible_dynamic_frame_count(checkpoint, observations))

    def test_mismatched_capacity_fails_before_model_load(self) -> None:
        checkpoint = {"model": {"uncertainty.log_var_frame": torch.zeros(7150), "uncertainty.frame_embedding.weight": torch.zeros(7150, 8)}}
        with self.assertRaisesRegex(ValueError, "checkpoint dynamic-frame capacity = 7150.*current QC-valid dynamic-frame count = 7200"):
            validate_checkpoint_dynamic_frame_capacity(checkpoint, 7200)

    def test_diagnostic_scripts_use_shared_count_helper(self) -> None:
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        for name in ("audit_all_location_cardiac_ablation.py", "audit_stage3a_loss_gradients.py", "diagnose_change4_checkpoint.py", "visualize_checkpoint_dynamics.py", "visualize_change5c_score_comparison.py"):
            self.assertIn("checkpoint_compatible_dynamic_frame_count(checkpoint, observations)", (root / "scripts" / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
