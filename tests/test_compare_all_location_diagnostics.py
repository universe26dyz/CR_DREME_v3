"""CPU contracts for paired full-location diagnostic summaries."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_all_location_diagnostics import compare_records  # noqa: E402


class PairedDiagnosticComparisonTest(unittest.TestCase):
    def test_compare_uses_exact_intersection_and_tolerance(self) -> None:
        left = [{"view": "SAX", "slice_id": "a", "cardiac_pca_waveform_r2": .2, "cardiac_target_fraction": .4, "card_target_over_wrong": 1., "card_score_std": .3, "same_peak_exact": False, "same_peak_within_0.02_hz": False}, {"view": "SAX", "slice_id": "only_left"}]
        right = [{"view": "SAX", "slice_id": "a", "cardiac_pca_waveform_r2": .3, "cardiac_target_fraction": .4 + 1e-7, "card_target_over_wrong": .8, "card_score_std": .4, "same_peak_exact": True, "same_peak_within_0.02_hz": True}, {"view": "2CH", "slice_id": "only_right"}]
        result = compare_records(left, right, tolerance=1e-5)
        self.assertEqual(["SAX/a"], result["paired_location_keys"])
        self.assertEqual(["SAX/only_left"], result["missing_from_right"])
        self.assertEqual(["2CH/only_right"], result["missing_from_left"])
        self.assertEqual(1, result["metrics"]["cardiac_pca_waveform_r2"]["n_improved"])
        self.assertEqual(1, result["metrics"]["cardiac_target_fraction"]["n_unchanged"])
        self.assertEqual(1, result["transitions"]["same_peak_exact"]["false_to_true"])


if __name__ == "__main__":
    unittest.main()
