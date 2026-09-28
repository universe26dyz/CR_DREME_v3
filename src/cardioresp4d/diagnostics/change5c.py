"""Pure, CPU-testable accounting helpers used by the read-only Change5C tools."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch


def choose_representative_indices(length: int, count: int) -> list[int]:
    """Deterministic full-span indices, with no early-frame bias."""
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    return torch.linspace(0, length - 1, count).round().long().tolist()


def deterministic_pixel_indices(total_pixels: int, cardiac_indices: torch.Tensor, *, count: int, cardiac_fraction: float, seed: int) -> torch.Tensor:
    """Training-equivalent 80/20-with-replacement index selection with local RNG."""
    if total_pixels <= 0 or count <= 0 or not 0 <= cardiac_fraction <= 1:
        raise ValueError("invalid deterministic pixel sampling inputs")
    count = min(count, total_pixels)
    device = cardiac_indices.device
    generator = torch.Generator(device=device).manual_seed(seed)
    all_indices = torch.arange(total_pixels, device=device)
    cardiac_count = int(round(count * cardiac_fraction)) if cardiac_indices.numel() else 0
    chosen = []
    if cardiac_count:
        chosen.append(cardiac_indices[torch.randint(cardiac_indices.numel(), (cardiac_count,), generator=generator, device=device)])
    chosen.append(all_indices[torch.randint(total_pixels, (count - cardiac_count,), generator=generator, device=device)])
    return torch.cat(chosen)


def l2_norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector.detach().reshape(-1)).cpu())


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left, right = left.detach().reshape(-1), right.detach().reshape(-1)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(left, right).div(denominator).cpu())


def summarize_gradient_vectors(vectors: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    return {name: {"l2_norm": l2_norm(vector)} for name, vector in vectors.items()}


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(tensor.mean()), "median": float(tensor.median()), "q25": float(torch.quantile(tensor, .25)), "q75": float(torch.quantile(tensor, .75)), "min": float(tensor.min()), "max": float(tensor.max())}


def ablation_record(view: str, slice_id: str, resp_only_mse: float, resp_plus_card_mse: float, joint_minus_resp: torch.Tensor, *, frame: int | None = None, timestamp_s: float | None = None, n_pixels: int | None = None) -> dict:
    """One primary cardiac-ROI ablation record from matched prediction arrays."""
    improvement = resp_only_mse - resp_plus_card_mse
    relative = 100.0 * improvement / max(resp_only_mse, torch.finfo(torch.float32).eps)
    effect = joint_minus_resp.detach().abs().reshape(-1).float()
    return {"view": view, "slice_id": slice_id, "frame": frame, "timestamp_s": timestamp_s, "n_pixels": int(effect.numel()) if n_pixels is None else n_pixels,
            "resp_only_mse": float(resp_only_mse), "resp_plus_card_mse": float(resp_plus_card_mse), "absolute_mse_improvement": float(improvement), "relative_mse_improvement_percent": float(relative),
            "mean_abs_joint_minus_resp": float(effect.mean()) if effect.numel() else None, "p95_abs_joint_minus_resp": float(torch.quantile(effect, .95)) if effect.numel() else None}


def aggregate_ablation(records: Iterable[dict], *, tolerance: float = 1e-8) -> dict:
    records = list(records)
    gains = [float(record["relative_mse_improvement_percent"]) for record in records if record.get("relative_mse_improvement_percent") is not None]
    effects = [float(record["mean_abs_joint_minus_resp"]) for record in records if record.get("mean_abs_joint_minus_resp") is not None]
    return {"n_records": len(records), "tolerance": tolerance,
            "n_positive_gain": sum(value > tolerance for value in gains), "n_negative_gain": sum(value < -tolerance for value in gains), "n_zero_gain": sum(abs(value) <= tolerance for value in gains),
            "relative_mse_improvement_percent": _quantiles(gains), "mean_abs_joint_minus_resp": _quantiles(effects)}


def _key(record: dict) -> str:
    return f"{record['view']}/{record['slice_id']}"


def paired_ablation_comparison(left: Iterable[dict], right: Iterable[dict], *, tolerance: float = 1e-8) -> dict:
    left_map, right_map = {_key(item): item for item in left}, {_key(item): item for item in right}
    keys = sorted(left_map.keys() & right_map.keys())
    metrics = ("resp_only_mse", "resp_plus_card_mse", "absolute_mse_improvement", "relative_mse_improvement_percent", "mean_abs_joint_minus_resp")
    deltas = {metric: [float(right_map[key][metric]) - float(left_map[key][metric]) for key in keys if left_map[key].get(metric) is not None and right_map[key].get(metric) is not None] for metric in metrics}
    return {"paired_location_keys": keys, "missing_from_right": sorted(left_map.keys() - right_map.keys()), "missing_from_left": sorted(right_map.keys() - left_map.keys()), "tolerance": tolerance, "metric_deltas": {metric: _quantiles(values) | {"n": len(values), "n_positive": sum(value > tolerance for value in values), "n_negative": sum(value < -tolerance for value in values), "n_zero": sum(abs(value) <= tolerance for value in values)} for metric, values in deltas.items()}, "note": "Exact-key descriptive paired comparison only; no statistical significance claim."}


def jacobian_summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().reshape(-1).float()
    if not values.numel():
        raise ValueError("Jacobian summary requires values")
    return {"min": float(values.min()), "p01": float(torch.quantile(values, .01)), "median": float(values.median()), "p99": float(torch.quantile(values, .99)), "max": float(values.max()), "fraction_leq_zero": float((values <= 0).float().mean())}


def cardiac_effect(resp_only: torch.Tensor, resp_plus_card: torch.Tensor) -> torch.Tensor:
    """Signed observation-conditioned cardiac prediction/volume contribution."""
    return resp_plus_card - resp_only


def select_visualization_locations(baseline_semantic: Iterable[dict], candidate_semantic: Iterable[dict], candidate_ablation: Iterable[dict], *, fixed: Iterable[tuple[str, str]] = ()) -> list[dict]:
    """Choose fixed plus extreme paired Change5C locations, retaining every reason."""
    reasons: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for view, slice_id in fixed:
        reasons[(view, slice_id)].append("fixed_representative")
    baseline = {_key(item): item for item in baseline_semantic if item.get("cardiac_pca_waveform_r2") is not None}
    candidate = {_key(item): item for item in candidate_semantic if item.get("cardiac_pca_waveform_r2") is not None}
    paired = sorted(set(baseline) & set(candidate))
    if paired:
        pca_delta = lambda key: float(candidate[key]["cardiac_pca_waveform_r2"]) - float(baseline[key]["cardiac_pca_waveform_r2"])
        for label, key in (("largest_c5b_pca_r2_improvement", max(paired, key=pca_delta)), ("largest_c5b_pca_r2_deterioration", min(paired, key=pca_delta))):
            view, slice_id = key.split("/", 1); reasons[(view, slice_id)].append(label)
    rows = [row for row in candidate_ablation if row.get("relative_mse_improvement_percent") is not None]
    if rows:
        for label, row in (("largest_c5b_reconstruction_gain", max(rows, key=lambda item: float(item["relative_mse_improvement_percent"]))), ("largest_c5b_reconstruction_loss", min(rows, key=lambda item: float(item["relative_mse_improvement_percent"]))), ("largest_cardiac_effect", max(rows, key=lambda item: float(item.get("mean_abs_joint_minus_resp") or 0.))), ("near_zero_cardiac_effect", min(rows, key=lambda item: float(item.get("mean_abs_joint_minus_resp") or 0.)))):
            reasons[(row["view"], row["slice_id"])].append(label)
    return [{"view": view, "slice_id": slice_id, "reasons": value} for (view, slice_id), value in sorted(reasons.items())]
