"""Test fixed-slice image PCA and frequency candidate safeguards.

These tests use independently constructed spatial patterns and temporal signals so
the expected respiration and cardiac frequencies do not come from the analyzer.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cardioresp4d.frequency.frequency_bands import aggregate_frequency_bands  # noqa: E402
from cardioresp4d.frequency.pca_motion import (  # noqa: E402
    SamplingIrregularError,
    analyze_image_series,
)


def synthetic_series(
    timestamps_s: np.ndarray,
    respiratory_hz: float = 0.24,
    cardiac_hz: float = 1.20,
) -> np.ndarray:
    """Create an image series with spatially independent respiratory and cardiac modes."""
    rows, columns = 12, 10
    yy, xx = np.mgrid[:rows, :columns]
    respiratory_pattern = np.where(xx < columns // 2, 1.0, -1.0)
    cardiac_pattern = np.where(yy < rows // 2, 1.0, -1.0)
    times = timestamps_s[:, None, None]
    return (
        0.75 * np.sin(2.0 * np.pi * respiratory_hz * times) * respiratory_pattern
        + 1.20 * np.sin(2.0 * np.pi * cardiac_hz * times) * cardiac_pattern
    ).astype(np.float32)


class PcaFrequencyTest(unittest.TestCase):
    """Exercise public fixed-slice PCA/PSD outputs and reliability rules."""

    def test_identifies_independently_specified_respiratory_and_cardiac_peaks(self) -> None:
        """SVD temporal PCs and one-sided PSD recover 0.24 Hz and 1.20 Hz modes."""
        timestamps_s = np.arange(180, dtype=float) * 0.1
        result = analyze_image_series(synthetic_series(timestamps_s), timestamps_s)

        self.assertAlmostEqual(0.1, result["median_dt_s"], places=8)
        self.assertAlmostEqual(1.0 / 18.0, result["df_hz"], places=8)
        self.assertAlmostEqual(5.0, result["nyquist_hz"], places=8)
        self.assertAlmostEqual(0.24, result["respiratory_candidate"]["frequency_hz"], delta=result["df_hz"])
        self.assertAlmostEqual(1.20, result["cardiac_candidate"]["frequency_hz"], delta=result["df_hz"])
        self.assertTrue(result["respiratory_candidate"]["reliable"])
        self.assertTrue(result["cardiac_candidate"]["reliable"])
        self.assertGreater(result["explained_variance"][0], 0.5)

    def test_rejects_irregular_acquisition_times_before_periodogram(self) -> None:
        """Ordinary FFT/periodogram refuses timestamps outside the documented tolerance."""
        timestamps_s = np.arange(50, dtype=float) * 0.171
        timestamps_s[20:] += 0.02

        with self.assertRaisesRegex(SamplingIrregularError, "uniform"):
            analyze_image_series(synthetic_series(timestamps_s), timestamps_s)

    def test_accepts_millisecond_quantised_acquisition_times(self) -> None:
        """One millisecond DICOM timestamp quantisation remains FFT-uniform."""
        timestamps_s = np.arange(50, dtype=float) * 0.171
        timestamps_s[20:] -= 0.001

        result = analyze_image_series(synthetic_series(timestamps_s), timestamps_s)

        self.assertLessEqual(result["max_dt_deviation_s"], 0.0010001)

    def test_short_duration_respiration_remains_a_dominance_based_candidate(self) -> None:
        """An 8.55 s series keeps a resolvable respiratory signal with duration caveat."""
        timestamps_s = np.arange(50, dtype=float) * 0.171
        result = analyze_image_series(synthetic_series(timestamps_s), timestamps_s)
        bands = aggregate_frequency_bands([result])

        self.assertAlmostEqual(8.379, result["duration_span_s"], places=8)
        self.assertAlmostEqual(8.55, result["duration_n_dt_s"], places=8)
        self.assertAlmostEqual(1.0 / 8.55, result["df_hz"], places=8)
        respiratory = result["respiratory_candidate"]
        self.assertTrue(respiratory["reliable"])
        self.assertEqual("peak_dominance_at_least_2.2", respiratory["reason"])
        self.assertIsNotNone(respiratory["frequency_hz"])
        self.assertIsNotNone(respiratory["selected_pc"])
        selected_pc = respiratory["selected_pc"] - 1
        self.assertGreater(float(np.abs(result["temporal_pcs"][:, selected_pc]).max()), 0.0)
        self.assertGreater(float(result["pc_psd"][:, selected_pc].max()), 0.0)
        self.assertIsNone(bands["respiratory"]["verified_band_hz"])
        self.assertEqual("no_cross_slice_consensus", bands["respiratory"]["reason"])
        self.assertIn("observation_duration_caveat", bands["respiratory"]["limitations"])
        per_slice = bands["respiratory"]["per_slice_candidates"][0]
        self.assertAlmostEqual(8.379, per_slice["duration_span_s"], places=8)
        self.assertAlmostEqual(8.55, per_slice["duration_n_dt_s"], places=8)
        self.assertAlmostEqual(1.0 / 8.55, per_slice["df_hz"], places=8)
        self.assertAlmostEqual(1.0 / (2.0 * 0.171), per_slice["nyquist_hz"], places=8)
        self.assertEqual(50, len(per_slice["explained_variance"]))

    def test_insufficient_frequency_resolution_returns_null_candidates(self) -> None:
        """Very short series cannot form an auditable respiratory or cardiac candidate."""
        timestamps_s = np.arange(4, dtype=float)
        result = analyze_image_series(synthetic_series(timestamps_s), timestamps_s)

        self.assertIsNone(result["respiratory_candidate"]["frequency_hz"])
        self.assertEqual("insufficient_frequency_resolution", result["respiratory_candidate"]["reason"])
        self.assertIsNone(result["cardiac_candidate"]["frequency_hz"])

    def test_zero_signal_has_no_semantic_frequency_peak(self) -> None:
        """A flat 50-frame slice reports null peak fields rather than argmax bin zero."""
        timestamps_s = np.arange(50, dtype=float) * 0.171
        images = np.zeros((50, 12, 10), dtype=np.float32)
        result = analyze_image_series(images, timestamps_s)

        for candidate_name in ("respiratory_candidate", "cardiac_candidate"):
            candidate = result[candidate_name]
            self.assertIsNone(candidate["frequency_hz"])
            self.assertIsNone(candidate["selected_pc"])
            self.assertIsNone(candidate["peak_power"])
            self.assertIsNone(candidate["dominance"])
            self.assertFalse(candidate["reliable"])
            self.assertEqual("no_positive_peak_power", candidate["reason"])

    def test_verified_band_requires_majority_of_all_eligible_slices(self) -> None:
        """One reliable slice cannot label an aggregate band verified when one peer is null."""
        reliable = analyze_image_series(
            synthetic_series(np.arange(50, dtype=float) * 0.171),
            np.arange(50, dtype=float) * 0.171,
        )
        null = analyze_image_series(
            np.zeros((50, 12, 10), dtype=np.float32),
            np.arange(50, dtype=float) * 0.171,
        )

        aggregate = aggregate_frequency_bands([reliable, null], consensus_min_slice_fraction=0.5)

        respiratory = aggregate["respiratory"]
        self.assertIsNone(respiratory["verified_band_hz"])
        self.assertEqual("no_cross_slice_consensus", respiratory["reason"])
        self.assertEqual(2, respiratory["consensus"]["eligible_slice_count"])
        self.assertEqual(1, max(item["support_count"] for item in respiratory["resolution_bin_support"]))

    def test_verified_band_records_resolution_bin_majority_support(self) -> None:
        """Only a recurrent resolution bin is verified and its support is auditable."""
        timestamps_s = np.arange(50, dtype=float) * 0.171
        first = analyze_image_series(synthetic_series(timestamps_s), timestamps_s)
        second = analyze_image_series(synthetic_series(timestamps_s), timestamps_s)
        off_frequency = analyze_image_series(
            synthetic_series(timestamps_s, respiratory_hz=0.47), timestamps_s
        )

        respiratory = aggregate_frequency_bands(
            [first, second, off_frequency], consensus_min_slice_fraction=0.5
        )["respiratory"]

        self.assertIsNotNone(respiratory["verified_band_hz"])
        self.assertEqual("cross_slice_consensus", respiratory["reason"])
        self.assertEqual(3, respiratory["consensus"]["eligible_slice_count"])
        self.assertTrue(any(item["support_count"] == 2 for item in respiratory["resolution_bin_support"]))

    def test_adjacent_fft_bins_are_not_false_cross_slice_recurrence(self) -> None:
        """Resolution intervals that only touch at one FFT edge remain distinct bins."""
        timestamps_s = np.arange(50, dtype=float) * 0.171
        lower_bin = analyze_image_series(synthetic_series(timestamps_s, respiratory_hz=0.24), timestamps_s)
        adjacent_bin = analyze_image_series(synthetic_series(timestamps_s, respiratory_hz=0.35), timestamps_s)

        respiratory = aggregate_frequency_bands(
            [lower_bin, adjacent_bin], consensus_min_slice_fraction=0.5
        )["respiratory"]

        self.assertIsNone(respiratory["verified_band_hz"])
        self.assertEqual("no_cross_slice_consensus", respiratory["reason"])
        self.assertEqual(2, len(respiratory["resolution_bin_support"]))
        self.assertEqual([1, 1], sorted(item["support_count"] for item in respiratory["resolution_bin_support"]))


if __name__ == "__main__":
    unittest.main()
