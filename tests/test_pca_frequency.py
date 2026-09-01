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

    def test_short_duration_respiration_is_not_a_verified_global_band(self) -> None:
        """A 50-frame 8.55 s acquisition may report but cannot verify respiration."""
        timestamps_s = np.arange(50, dtype=float) * 0.171
        result = analyze_image_series(synthetic_series(timestamps_s), timestamps_s)
        bands = aggregate_frequency_bands([result])

        self.assertAlmostEqual(8.379, result["duration_span_s"], places=8)
        self.assertAlmostEqual(8.55, result["duration_n_dt_s"], places=8)
        self.assertAlmostEqual(1.0 / 8.55, result["df_hz"], places=8)
        self.assertFalse(result["respiratory_candidate"]["reliable"])
        self.assertEqual("limited_duration", result["respiratory_candidate"]["reason"])
        self.assertIsNone(bands["respiratory"]["verified_band_hz"])
        self.assertEqual("limited_duration", bands["respiratory"]["reason"])
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


if __name__ == "__main__":
    unittest.main()
