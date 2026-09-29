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


if __name__ == "__main__":
    unittest.main()
