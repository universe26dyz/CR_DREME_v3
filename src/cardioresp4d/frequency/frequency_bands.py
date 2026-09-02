"""Aggregate per-slice frequency candidates without inventing a global heart rate."""

from __future__ import annotations

from typing import Any, Iterable


def aggregate_frequency_bands(slice_results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build transparent per-slice and union-bin frequency evidence for all slices.

    Respiratory and cardiac verification use the same positive-peak and
    dominance evidence.  Both retain per-slice candidates and merged
    spectral-resolution bins instead of forcing a misleading point estimate
    across the sequential scan.  Observation span and ``df`` remain explicit
    limitations on the precision of every reported band.
    """
    results = list(slice_results)
    respiratory = _aggregate_kind(results, "respiratory_candidate", is_respiratory=True)
    cardiac = _aggregate_kind(results, "cardiac_candidate", is_respiratory=False)
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
    results: list[dict[str, Any]], candidate_key: str, *, is_respiratory: bool) -> dict[str, Any]:
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
    union_bins = _merge_bins(bins)
    if is_respiratory:
        return {
            "verified_band_hz": union_bins or None,
            "reason": "reliable_candidates_present" if reliable_frequencies else "no_reliable_respiratory_candidate",
            "per_slice_candidates": per_slice,
            "reliable_frequency_distribution_hz": distribution,
            "union_resolution_bins_hz": union_bins,
            "limitations": {
                "observation_duration_caveat": (
                    "Candidate reliability is based on positive peak power and dominance; "
                    "duration and df limit frequency precision. Interpret verified_band_hz as "
                    "merged periodogram-resolution bins, not exact frequencies."
                ),
                "duration_span_s_range": _range_or_none(
                    [float(result["duration_span_s"]) for result in results]
                ),
                "df_hz_range": _range_or_none([float(result["df_hz"]) for result in results]),
            },
        }
    return {
        "per_slice_candidates": per_slice,
        "reliable_frequency_distribution_hz": distribution,
        "union_resolution_bins_hz": union_bins,
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


def _range_or_none(values: list[float]) -> list[float] | None:
    """Return a two-value range, or null when no slice result was supplied."""
    return None if not values else [min(values), max(values)]
