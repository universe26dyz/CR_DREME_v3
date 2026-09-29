"""Checkpoint-effective configuration contracts for Change5C."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_checkpoint_contract import checkpoint_effective_loss_weights, relevant_weight_agreement  # noqa: E402


class CheckpointContractHotfixTest(unittest.TestCase):
    def test_checkpoint_report_weights_are_authoritative(self) -> None:
        checkpoint = {"report": {"effective_config": {"loss_weights": {"cardiac_pca_waveform": .001}}}}
        self.assertEqual(.001, checkpoint_effective_loss_weights(checkpoint)["cardiac_pca_waveform"])

    def test_missing_checkpoint_effective_weights_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint-effective"):
            checkpoint_effective_loss_weights({"report": {}})

    def test_supplied_yaml_mismatch_is_detected(self) -> None:
        agreement = relevant_weight_agreement({"cardiac_pca_waveform": .001}, {"cardiac_pca_waveform": 0.})
        self.assertFalse(agreement["all_relevant_keys_agree"])
        self.assertFalse(agreement["by_key"]["cardiac_pca_waveform"]["agree"])

    def test_omitted_yaml_pca_weight_defaults_to_checkpoint_zero(self) -> None:
        checkpoint_defaults = {"cardiac_target_concentration": 0., "cardiac_pca_waveform": 0., "respiratory_leakage_in_card": 1e-4, "cardiac_leakage_in_resp": 1e-4, "zero_mean_score": 1e-5, "smooth_card": 1e-5, "mbc_normalization": 1e-5}
        self.assertTrue(relevant_weight_agreement(checkpoint_defaults, {})["all_relevant_keys_agree"])

    def test_omitted_yaml_concentration_weight_defaults_to_checkpoint_zero(self) -> None:
        checkpoint_defaults = {"cardiac_target_concentration": 0., "cardiac_pca_waveform": 0., "respiratory_leakage_in_card": 1e-4, "cardiac_leakage_in_resp": 1e-4, "zero_mean_score": 1e-5, "smooth_card": 1e-5, "mbc_normalization": 1e-5}
        self.assertTrue(relevant_weight_agreement(checkpoint_defaults, {})["all_relevant_keys_agree"])

    def test_non_default_yaml_weight_still_disagrees(self) -> None:
        agreement = relevant_weight_agreement({"cardiac_pca_waveform": 0.}, {"cardiac_pca_waveform": .001})
        self.assertFalse(agreement["all_relevant_keys_agree"])


if __name__ == "__main__":
    unittest.main()
