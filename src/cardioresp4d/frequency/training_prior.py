"""Validated Phase-1 frequency evidence for timestamped DREME regularization."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Band = tuple[float, float]


@dataclass(frozen=True)
class LocationFrequencyPrior:
    """Resolved evidence selected for exactly one fixed acquisition location."""

    respiratory_bands_hz: list[Band]
    cardiac_bands_hz: list[Band]
    cardiac_baseline_pairs: list[dict[str, Any]]
    source: str


@dataclass(frozen=True)
class TrainingFrequencyPrior(LocationFrequencyPrior):
    """Global evidence plus only schema-verified per-location overrides."""

    source_path: str
    source_schema: str
    provenance: dict[str, Any]
    locations: dict[str, LocationFrequencyPrior]

    @property
    def baseline_bands_hz(self) -> list[Band]:
        """Compatibility view; formal trainer consumes ``cardiac_baseline_pairs``."""
        return [tuple(item["baseline_band_hz"]) for item in self.cardiac_baseline_pairs]

    def for_location(self, view: str, slice_id: str) -> LocationFrequencyPrior:
        key = f"{view}/{slice_id}"
        return self.locations.get(key, LocationFrequencyPrior(self.respiratory_bands_hz, self.cardiac_bands_hz, self.cardiac_baseline_pairs, "phase1_global_fallback"))


def _bands(value: object, label: str) -> list[Band]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of [low, high] bins")
    result = [(float(item[0]), float(item[1])) for item in value]
    if any(not (lo >= 0 and hi >= lo) for lo, hi in result):
        raise ValueError(f"{label} has invalid bins")
    return result


def _location_key(value: object) -> str:
    if not isinstance(value, str) or value.count("/") != 1:
        raise ValueError("Phase-1 per-slice candidate must have unambiguous slice_key 'view/slice_id'")
    view, slice_id = value.split("/", 1)
    if not view or not slice_id:
        raise ValueError("Phase-1 slice_key has empty view or slice_id")
    return value


def _overlaps(candidate: Band, bands: list[Band]) -> bool:
    return any(candidate[0] < upper and candidate[1] > lower for lower, upper in bands)


def _baseline_pairs(cardiac: list[Band], respiratory: list[Band], *, source: str, location_key: str | None = None) -> list[dict[str, Any]]:
    """Build explicitly paired non-physiological equal-width reference bands.

    DREME does not prescribe this exact neighbouring-bin construction in the
    pinned sources available to this project, so provenance calls it a
    paper-derived *necessary adaptation*, not an upstream DREME primitive.
    """
    occupied = respiratory + cardiac
    pairs: list[dict[str, Any]] = []
    for cardiac_band in cardiac:
        lo, hi = cardiac_band
        width = hi - lo
        if width <= 0:
            raise ValueError("cardiac bands must have positive width")
        baseline = (max(width, lo - width), lo)
        while _overlaps(baseline, occupied):
            baseline = (baseline[1], baseline[1] + width)
        pairs.append({
            "cardiac_band_hz": list(cardiac_band),
            "baseline_band_hz": list(baseline),
            "resolution_hz": width,
            "source": source,
            "location_key": location_key,
            "baseline_rule": "paper-derived necessary adaptation: adjacent equal-width non-DC non-physiological band",
        })
    return pairs


def load_training_frequency_prior(path: str | Path, *, allow_template_fallback: bool = False, fallback_respiratory_bands_hz: list[Band] | None = None) -> TrainingFrequencyPrior:
    """Load aggregate v1 Phase-1 evidence without inventing a location map."""
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version", 1) != 1:
        raise ValueError("formal training requires Phase-1 aggregate schema_version=1")
    respiratory = _bands(payload.get("respiratory", {}).get("verified_band_hz"), "respiratory.verified_band_hz")
    if not respiratory:
        if not allow_template_fallback:
            raise ValueError("Phase-1 frequency prior has no verified respiratory band")
        respiratory = list(fallback_respiratory_bands_hz or [])
        if not respiratory:
            raise ValueError("allow_template_fallback requires non-empty configured fallback_respiratory_bands_hz")
    cardiac_payload = payload.get("cardiac", {})
    cardiac = _bands(cardiac_payload.get("union_resolution_bins_hz"), "cardiac.union_resolution_bins_hz")
    if not cardiac:
        raise ValueError("Phase-1 frequency prior has no cardiac resolution bins")
    global_pairs = _baseline_pairs(cardiac, respiratory, source="phase1_global")
    locations: dict[str, LocationFrequencyPrior] = {}
    for candidate in cardiac_payload.get("per_slice_candidates", []):
        key = _location_key(candidate.get("slice_key"))
        if key in locations:
            raise ValueError(f"duplicate Phase-1 cardiac candidate for {key}")
        frequency, df = candidate.get("frequency_hz"), candidate.get("df_hz")
        if not candidate.get("reliable"):
            continue
        if frequency is None or df is None or float(df) <= 0:
            raise ValueError(f"reliable Phase-1 cardiac candidate for {key} lacks finite frequency_hz/df_hz")
        half = float(df) / 2.
        local_cardiac = [(float(frequency) - half, float(frequency) + half)]
        if local_cardiac[0][0] <= 0:
            raise ValueError(f"Phase-1 cardiac candidate for {key} reaches DC")
        locations[key] = LocationFrequencyPrior(respiratory, local_cardiac, _baseline_pairs(local_cardiac, respiratory, source="phase1_per_location", location_key=key), "phase1_per_location")
    return TrainingFrequencyPrior(respiratory, cardiac, global_pairs, "phase1_global", str(source), "phase1_aggregate_v1", {"global_cardiac_source": "union_resolution_bins_hz", "per_location_identity": "Phase-1 slice_key=view/slice_id", "baseline_rule": global_pairs[0]["baseline_rule"]}, locations)
