"""Aggregate auditable cross-slice frequency evidence without inventing a global rate.

功能：汇总 fixed-slice PCA/PSD 候选，同时把逐 slice 候选与跨 slice
resolution-bin 支持度分开报告。
论文来源：image-domain PCA 的频率发现；DREME-MR 的频率先验使用方式。
输入：逐 slice 的 PCA 结果字典以及最低跨 slice 支持比例。
输出：逐 slice 候选、resolution-bin support table，及仅在重复支持时的呼吸
``verified_band_hz``；不产生一个全局心率点估计。
主要步骤：每个可靠候选转换为半个 ``df`` 的 bin，合并重叠 bin，并按全部
eligible slices 计算支持比例。
是否属于原论文直接实现 / 必要适配 / 可选实验：Necessary adaptation.
命令行：内部 library；由 ``pca_motion`` 的 pipeline API 调用。
"""

from __future__ import annotations

from typing import Any, Iterable


def aggregate_frequency_bands(
    slice_results: Iterable[dict[str, Any]], *, consensus_min_slice_fraction: float = 0.5
) -> dict[str, Any]:
    """Build per-slice candidates and recurrent-bin consensus evidence.

    Respiratory and cardiac verification use the same positive-peak and
    dominance evidence.  Both retain per-slice candidates and merged
    spectral-resolution bins instead of forcing a misleading point estimate
    across the sequential scan.  Observation span and ``df`` remain explicit
    limitations on the precision of every reported band.
    """
    if not 0.0 < consensus_min_slice_fraction <= 1.0:
        raise ValueError("consensus_min_slice_fraction must lie in (0, 1]")
    results = list(slice_results)
    respiratory = _aggregate_kind(
        results,
        "respiratory_candidate",
        is_respiratory=True,
        consensus_min_slice_fraction=consensus_min_slice_fraction,
    )
    cardiac = _aggregate_kind(
        results,
        "cardiac_candidate",
        is_respiratory=False,
        consensus_min_slice_fraction=consensus_min_slice_fraction,
    )
    return {
        "schema_version": 1,
        "method": {
            "pca": "mean-centred time-by-pixels NumPy SVD; temporal PCs = U*S",
            "psd": "one-sided SciPy periodogram; only uniform AcquisitionTime sampling accepted",
            "dominance_threshold": 2.2,
            "consensus_min_slice_fraction": consensus_min_slice_fraction,
            "consensus_requires_at_least_two_slices": True,
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
    results: list[dict[str, Any]], candidate_key: str, *, is_respiratory: bool,
    consensus_min_slice_fraction: float,
) -> dict[str, Any]:
    """Collect candidates, summary distribution, and merged resolution bins for one signal kind."""
    per_slice = []
    reliable_frequencies: list[float] = []
    bins: list[tuple[float, float, int]] = []
    for slice_index, result in enumerate(results):
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
            bins.append((frequency_float - half_bin, frequency_float + half_bin, slice_index))
    distribution = None if not reliable_frequencies else {
        "count": len(reliable_frequencies),
        "min_hz": min(reliable_frequencies),
        "max_hz": max(reliable_frequencies),
        "median_hz": _median(reliable_frequencies),
    }
    support = _resolution_bin_support(bins, eligible_slice_count=len(results))
    union_bins = [[item["lower_hz"], item["upper_hz"]] for item in support]
    consensus_bins = [
        [item["lower_hz"], item["upper_hz"]]
        for item in support
        if item["support_count"] >= 2
        and item["support_fraction"] >= consensus_min_slice_fraction
    ]
    consensus = {
        "eligible_slice_count": len(results),
        "min_slice_fraction": consensus_min_slice_fraction,
        "minimum_support_count": 2,
    }
    if is_respiratory:
        return {
            "verified_band_hz": consensus_bins or None,
            "reason": (
                "cross_slice_consensus" if consensus_bins
                else "no_cross_slice_consensus" if reliable_frequencies
                else "no_reliable_respiratory_candidate"
            ),
            "per_slice_candidates": per_slice,
            "reliable_frequency_distribution_hz": distribution,
            "union_resolution_bins_hz": union_bins,
            "resolution_bin_support": support,
            "consensus": consensus,
            "limitations": {
                "observation_duration_caveat": (
                    "Candidate reliability is based on positive peak power and dominance; "
                    "duration and df limit frequency precision. verified_band_hz requires "
                    "at least two slices and the documented all-slice support fraction; it is "
                    "a merged periodogram-resolution bin, not an exact frequency."
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
        "resolution_bin_support": support,
        "consensus": consensus,
    }


def _resolution_bin_support(
    bins: list[tuple[float, float, int]], *, eligible_slice_count: int
) -> list[dict[str, float | int]]:
    """Merge overlapping bins and retain distinct-slice support counts."""
    if not bins:
        return []
    merged: list[dict[str, Any]] = []
    for lower, upper, slice_index in sorted(bins, key=lambda item: (item[0] + item[1]) / 2.0):
        center, resolution = (lower + upper) / 2.0, upper - lower
        if (
            not merged
            or abs(center - merged[-1]["center_hz"])
            > 0.25 * max(resolution, merged[-1]["resolution_hz"])
        ):
            merged.append({
                "lower_hz": lower,
                "upper_hz": upper,
                "center_hz": center,
                "resolution_hz": resolution,
                "slice_indices": {slice_index},
            })
        else:
            merged[-1]["upper_hz"] = max(merged[-1]["upper_hz"], upper)
            merged[-1]["lower_hz"] = min(merged[-1]["lower_hz"], lower)
            merged[-1]["center_hz"] = (merged[-1]["lower_hz"] + merged[-1]["upper_hz"]) / 2.0
            merged[-1]["resolution_hz"] = max(merged[-1]["resolution_hz"], resolution)
            merged[-1]["slice_indices"].add(slice_index)
    return [
        {
            "lower_hz": float(item["lower_hz"]),
            "upper_hz": float(item["upper_hz"]),
            "support_count": len(item["slice_indices"]),
            "support_fraction": len(item["slice_indices"]) / eligible_slice_count,
            "interval_hz": [float(item["lower_hz"]), float(item["upper_hz"])],
            "count": len(item["slice_indices"]),
            "total": eligible_slice_count,
            "fraction": len(item["slice_indices"]) / eligible_slice_count,
        }
        for item in merged
    ]


def _median(values: list[float]) -> float:
    """Return the middle value without importing numerical dependencies into aggregation."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def _range_or_none(values: list[float]) -> list[float] | None:
    """Return a two-value range, or null when no slice result was supplied."""
    return None if not values else [min(values), max(values)]
