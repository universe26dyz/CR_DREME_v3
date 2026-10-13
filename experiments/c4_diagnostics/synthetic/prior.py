"""Synthetic ground-truth frequency evidence for the production C4 trainer."""
from __future__ import annotations

from typing import Iterable

from cardioresp4d.frequency.training_prior import LocationFrequencyPrior, TrainingFrequencyPrior


def _band(frequency_hz: float, half_width_hz: float) -> tuple[float, float]:
    if frequency_hz <= 0.0 or half_width_hz <= 0.0 or frequency_hz <= half_width_hz:
        raise ValueError("synthetic frequencies and band widths must be positive with no DC overlap")
    return frequency_hz - half_width_hz, frequency_hz + half_width_hz


def build_synthetic_frequency_prior(respiratory_frequency_hz: float, cardiac_frequency_hz: float, locations: Iterable[tuple[str, str]], *, half_width_hz: float = .20) -> TrainingFrequencyPrior:
    """Create location-local GT bands without consuming patient Phase-1 evidence."""
    respiratory = [_band(respiratory_frequency_hz, min(half_width_hz, respiratory_frequency_hz * .6))]
    cardiac = [_band(cardiac_frequency_hz, half_width_hz)]
    baseline = [(cardiac[0][1] + 2.0 * half_width_hz, cardiac[0][1] + 4.0 * half_width_hz)]
    pairs = [{"cardiac_band_hz": list(cardiac[0]), "baseline_band_hz": list(baseline[0]), "resolution_hz": 2.0 * half_width_hz, "source": "synthetic_gt", "location_key": None, "baseline_rule": "synthetic non-physiological separated band"}]
    location_map = {}
    for view, slice_id in locations:
        key = f"{view}/{slice_id}"
        location_map[key] = LocationFrequencyPrior(respiratory, cardiac, [{**pairs[0], "location_key": key}], "synthetic_gt", "synthetic_gt", "synthetic_gt")
    return TrainingFrequencyPrior(respiratory, cardiac, pairs, "synthetic_gt", "synthetic_gt", "synthetic_gt", "synthetic_gt", "synthetic_gt_v1", {"kind": "known_ground_truth", "respiratory_frequency_hz": respiratory_frequency_hz, "cardiac_frequency_hz": cardiac_frequency_hz}, location_map)
