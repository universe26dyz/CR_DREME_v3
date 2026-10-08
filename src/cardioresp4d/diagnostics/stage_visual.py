"""Pure statistics and selection contracts for read-only stage visual audits."""
from __future__ import annotations

from collections import defaultdict

import torch


def finite_array_stats(values: torch.Tensor) -> dict[str, float]:
    """Summarize unconverted float data and reject non-finite rendering inputs."""
    flat = values.detach().float().reshape(-1).cpu()
    if not flat.numel() or not torch.isfinite(flat).all():
        raise ValueError("stage visual arrays must be non-empty and finite")
    return {"min": float(flat.min()), "max": float(flat.max()), "mean": float(flat.mean()), "median": float(flat.median()), "std": float(flat.std(unbiased=False)), "p95": float(torch.quantile(flat, .95)), "p99": float(torch.quantile(flat, .99)), "finite_fraction": 1., "nonzero_fraction": float((flat != 0).float().mean())}


def temporal_maps_and_metrics(acquired: torch.Tensor, predicted: torch.Tensor, cardiac_mask: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, float | None]]:
    """Return float temporal maps plus whole-FOV/ROI recovery statistics."""
    if acquired.ndim != 3 or acquired.shape != predicted.shape or acquired.shape[1:] != cardiac_mask.shape:
        raise ValueError("matched [T,H,W] acquired/predicted and [H,W] mask required")
    acquired, predicted = acquired.float(), predicted.float()
    if not torch.isfinite(acquired).all() or not torch.isfinite(predicted).all():
        raise ValueError("temporal audit arrays must be finite")
    acquired_std, predicted_std = acquired.std(0, unbiased=False), predicted.std(0, unbiased=False)
    acquired_delta, predicted_delta = acquired[1:] - acquired[:-1], predicted[1:] - predicted[:-1]
    def mean_in(values: torch.Tensor, mask: torch.Tensor | None) -> float | None:
        selected = values if mask is None else values[mask]
        return float(selected.mean()) if selected.numel() else None
    def correlation(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor | None) -> float | None:
        left, right = (left if mask is None else left[:, mask]).reshape(-1), (right if mask is None else right[:, mask]).reshape(-1)
        left, right = left - left.mean(), right - right.mean(); denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        return None if float(denominator) == 0. else float((left @ right / denominator).cpu())
    whole = mean_in(predicted_std, None); acquired_whole = mean_in(acquired_std, None)
    roi = mean_in(predicted_std, cardiac_mask); acquired_roi = mean_in(acquired_std, cardiac_mask)
    return {"acquired_temporal_std": acquired_std, "predicted_temporal_std": predicted_std, "acquired_delta": acquired_delta, "predicted_delta": predicted_delta}, {
        "whole_fov_temporal_std": whole, "cardiac_roi_temporal_std": roi,
        "acquired_whole_fov_temporal_std": acquired_whole, "acquired_cardiac_roi_temporal_std": acquired_roi,
        "temporal_std_recovery_ratio": None if acquired_whole in (None, 0.) else whole / acquired_whole,
        "mean_abs_consecutive_change": float(predicted_delta.abs().mean()), "delta_correlation": correlation(acquired_delta, predicted_delta, None),
        "cardiac_roi_delta_correlation": correlation(acquired_delta, predicted_delta, cardiac_mask),
    }


def dvf_static_dynamic_metrics(dvf_mm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Decompose [T,...,3] physical-mm DVFs into temporal mean and residual."""
    if dvf_mm.ndim < 3 or dvf_mm.shape[-1] != 3 or not torch.isfinite(dvf_mm).all():
        raise ValueError("finite [T,...,3] DVF in physical mm required")
    mean = dvf_mm.float().mean(0); dynamic = dvf_mm.float() - mean
    static_magnitude = torch.linalg.vector_norm(mean, dim=-1); dynamic_magnitude = torch.linalg.vector_norm(dynamic, dim=-1)
    static_rms = float(static_magnitude.square().mean().sqrt()); dynamic_rms = float(dynamic_magnitude.square().mean().sqrt())
    def quantiles(values: torch.Tensor, prefix: str) -> dict[str, float]:
        flat = values.reshape(-1); return {f"{prefix}_{name}_mm": float(getattr(torch, name)(flat)) if name == "max" else float(torch.quantile(flat, level)) for name, level in (("p50", .5), ("p95", .95), ("p99", .99), ("max", 1.))}
    return mean, dynamic, {"mean_DVF_RMS_mm": static_rms, **quantiles(static_magnitude, "mean_DVF"), "dynamic_DVF_temporal_RMS_mm": dynamic_rms, **quantiles(dynamic_magnitude, "dynamic_DVF"), "static_to_dynamic_RMS_ratio": static_rms / max(dynamic_rms, torch.finfo(torch.float32).eps)}


def select_stack_locations(observations, *, existing: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Select lower/middle/upper QC-valid locations by physical slice position."""
    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        if observation.qc_valid:
            grouped[observation.view][observation.slice_id].append(observation)
    result: dict[str, list[str]] = {}
    for view in ("SAX", "2CH", "4CH"):
        available = grouped.get(view, {})
        if existing and view in existing:
            result[view] = [slice_id for slice_id in existing[view] if slice_id in available]
            continue
        positions = []
        for slice_id, rows in available.items():
            centers = torch.stack([row.center_mm.detach().float().cpu() for row in rows]); normals = torch.stack([row.normal.detach().float().cpu() for row in rows])
            normal = normals.mean(0); normal = normal / torch.linalg.vector_norm(normal)
            positions.append((float((centers.mean(0) @ normal)), slice_id))
        positions.sort(); indices = torch.linspace(0, len(positions) - 1, min(3, len(positions))).round().long().tolist() if positions else []
        result[view] = [positions[index][1] for index in indices]
    return result
