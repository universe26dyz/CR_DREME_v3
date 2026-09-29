"""Regression contract for loss-first Stage3a gradient audit report rendering."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_change5c_closure import render_closure_report  # noqa: E402


class ClosureReportHotfixTest(unittest.TestCase):
    def test_loss_first_gradient_payload_renders_intended_norms(self) -> None:
        audit = {
            "data": {"cardiac_film_head": {"raw_l2_norm": 2.}, "cardiac_mbc": {"raw_l2_norm": 3.}},
            "semantic_aux": {"cardiac_film_head": {"combined_l2_norm": 4.}},
            "total": {"cardiac_mbc": {"combined_l2_norm": 5.}},
            "cardiac_target_concentration": {"cardiac_mbc": {"raw_l2_norm": 0.}},
            "cardiac_pca_waveform": {"cardiac_mbc": {"raw_l2_norm": 0.}},
            "combined_ratios": {"cardiac_film_head": {"semantic_aux_to_data_norm_ratio": 2.}},
            "cosine_similarity": {"cardiac_film_head": {"data_vs_semantic_aux": -.25}},
        }
        report = render_closure_report({"CTRL_2500": {"gradient_location_matched": {"audits": [audit]}, "gradient_training_step_matched": {"training_step_aggregate": {}}, "semantic_summary": {"overall": {"continuous": {}, "boolean": {}}}, "cardiac_ablation": {"overall": {}, "per_view": {}}}}, {}, {"CTRL_2500": {"current_stage": "stage3a", "global_step": 1, "stage_step": 1, "checkpoint_effective_loss_weights": {}}}, "CTRL_2500")
        self.assertIn("head data=2", report)
        self.assertIn("semantic=4", report)
        self.assertIn("MBC data=3", report)
        self.assertIn("total=5", report)


if __name__ == "__main__":
    unittest.main()
