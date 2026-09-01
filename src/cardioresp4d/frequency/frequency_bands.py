"""Aggregate per-slice frequency candidates without inventing a global heart rate."""

from __future__ import annotations

from typing import Any, Iterable


def aggregate_frequency_bands(slice_results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build transparent per-slice and union-bin frequency evidence for all slices.

    Respiratory verification remains null unless every prerequisite supports it;
    the Phase 1 50-frame acquisition is below the 10 s duration reported as
    necessary to track more than one respiratory cycle.  Cardiac output retains
    per-slice candidates and merged spectral-resolution bins instead of forcing
    a misleading single global frequency across the sequential scan.
    """
    results = list(slice_results)
    respiratory = _aggregate_kind(results, "respiratory_candidate", verify_respiration=True)
    cardiac = _aggregate_kind(results, "cardiac_candidate", verify_respiration=False)
    return {
        "schema_version": 1,
        "method": {
            "pca": "mean-centred time-by-pixels NumPy SVD; temporal PCs = U*S",
            "psd": "one-sided SciPy periodogram; only uniform AcquisitionTime sampling accepted",
            "dominance_threshold": 2.2,
        },
        "slice_count": len(results),
        "respiratory": respiratory,
        "cardiac": cardiac,
        "cardiac_interpretation": (
            "Per-slice cardiac candidates are retained because sequential scans may vary in time; "
            "union bins and the frequency distribution are reported instead of a forced single frequency."
        ),
    }


def _aggregate_kind(
    results: list[dict[str, Any]], candidate_key: str, *, verify_respiration: bool) -> dict[str, Any]:
    """Collect candidates, summary distribution, and merged resolution bins for one signal kind."""
    per_slice = []
    reliable_frequencies: list[float] = []
    bins: list[tuple[float, float]] = []
    for result in results:
        candidate = dict(result[candidate_key])
        candidate["slice_key"] = result.get("slice_key", result.get("slice_id", "unknown"))
        candidate["median_dt_s"] = float(result["median_dt_s"])
        candidate["duration_span_s"] = float(result["duration_span_s"])
        candidate["duration_n_dt_s"] = float(result["duration_n_dt_s"])
        candidate["df_hz"] = float(result["df_hz"])
        candidate["nyquist_hz"] = float(result["nyquist_hz"])
        candidate["explained_variance"] = [float(value) for value in result["explained_variance"]]
        per_slice.append(candidate)
        frequency = candidate["frequency_hz"]
        if candidate["reliable"] and frequency is not None:
            frequency_float = float(frequency)
            reliable_frequencies.append(frequency_float)
            half_bin = float(result["df_hz"]) / 2.0
            bins.append((frequency_float - half_bin, frequency_float + half_bin))
    distribution = None if not reliable_frequencies else {
        "count": len(reliable_frequencies),
        "min_hz": min(reliable_frequencies),
        "max_hz": max(reliable_frequencies),
        "median_hz": _median(reliable_frequencies),
    }
    if verify_respiration:
        any_limited = any(candidate["reason"] == "limited_duration" for candidate in per_slice)
        return {
            "verified_band_hz": None,
            "reason": "limited_duration" if any_limited else "no_reliable_respiratory_candidate",
            "per_slice_candidates": per_slice,
            "reliable_frequency_distribution_hz": distribution,
            "union_resolution_bins_hz": _merge_bins(bins),
        }
    return {
        "per_slice_candidates": per_slice,
        "reliable_frequency_distribution_hz": distribution,
        "union_resolution_bins_hz": _merge_bins(bins),
    }


def _merge_bins(bins: list[tuple[float, float]]) -> list[list[float]]:
    """Merge overlapping spectral-resolution bins in ascending frequency order."""
    if not bins:
        return []
    merged: list[list[float]] = []
    for lower, upper in sorted(bins):
        if not merged or lower > merged[-1][1]:
            merged.append([lower, upper])
        else:
            merged[-1][1] = max(merged[-1][1], upper)
    return merged


def _median(values: list[float]) -> float:
    """Return the middle value without importing numerical dependencies into aggregation."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
