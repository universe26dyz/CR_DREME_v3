from __future__ import annotations

import numpy as np
import torch

from cardioresp4d.losses.frequency_loss import non_dc_frequency_grid, nonuniform_dft_at_frequencies, resolve_target_frequency_mask


def location_spectral_metrics(scores: torch.Tensor, timestamps_s: torch.Tensor, target_bands_hz: list[tuple[float, float]]) -> dict[str, float]:
    """NUDFT diagnostics over independent score channels at one fixed location."""
    channels = scores.reshape(scores.shape[0], -1)
    frequencies = non_dc_frequency_grid(timestamps_s).to(device=scores.device, dtype=scores.dtype)
    spectrum = nonuniform_dft_at_frequencies(channels, timestamps_s, frequencies)
    power = spectrum.abs().square().sum(-1)
    mask = resolve_target_frequency_mask(frequencies, target_bands_hz)
    target = power[mask].sum()
    total = power.sum()
    dominant = frequencies[power.argmax()]
    dominant_in_target = any(lower <= float(dominant) <= upper for lower, upper in target_bands_hz)
    return {"channel_count": int(channels.shape[1]), "dominant_frequency_hz": float(dominant), "dominant_in_target_band": float(dominant_in_target), "target_band_power": float(target), "wrong_band_power": float((total - target).clamp_min(0.0)), "correct_band_fraction": float(target / total.clamp_min(torch.finfo(total.dtype).eps)), "wrong_band_fraction": float((total - target).clamp_min(0.0) / total.clamp_min(torch.finfo(total.dtype).eps))}


def temporal_dynamics_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred, truth = predicted.reshape(predicted.shape[0], -1), target.reshape(target.shape[0], -1)
    delta_pred, delta_truth = pred[1:] - pred[:-1], truth[1:] - truth[:-1]
    denominator = delta_pred.norm() * delta_truth.norm()
    return {"temporal_std_pred": float(pred.std(unbiased=False)), "temporal_std_gt": float(truth.std(unbiased=False)), "per_frame_mse": float((pred - truth).square().mean()), "consecutive_delta_cosine": float((delta_pred * delta_truth).sum() / denominator.clamp_min(1e-12)) if len(pred) > 1 else float("nan")}


def dvf_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_dynamic = predicted - predicted.mean(0, keepdim=True)
    gt_dynamic = target - target.mean(0, keepdim=True)
    denominator = pred_dynamic.norm() * gt_dynamic.norm()
    return {"static_rms_mm": float(predicted.mean(0).square().mean().sqrt()), "dynamic_rms_mm": float(pred_dynamic.square().mean().sqrt()), "gt_dynamic_rms_mm": float(gt_dynamic.square().mean().sqrt()), "dynamic_rms_ratio": float(pred_dynamic.square().mean().sqrt() / gt_dynamic.square().mean().sqrt().clamp_min(1e-12)), "endpoint_error_mm": float((predicted - target).square().sum(-1).sqrt().mean()), "dynamic_cosine": float((pred_dynamic * gt_dynamic).sum() / denominator.clamp_min(1e-12))}


def vector_field_agreement_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Pointwise vector agreement where sample locations need not match in time."""
    if predicted.shape != target.shape or predicted.shape[-1] != 3:
        raise ValueError("predicted and target vector fields must share an xyz shape")
    pred_rms = predicted.square().mean().sqrt()
    gt_rms = target.square().mean().sqrt()
    denominator = predicted.norm() * target.norm()
    endpoint_error = (predicted - target).square().sum(-1).sqrt().mean()
    return {
        "pred_rms_mm": float(pred_rms),
        "gt_rms_mm": float(gt_rms),
        "rms_ratio": float(pred_rms / gt_rms.clamp_min(1e-12)),
        "endpoint_error_mm": float(endpoint_error),
        "normalized_epe": float(endpoint_error / gt_rms.clamp_min(1e-12)),
        "spatial_cosine": float((predicted * target).sum() / denominator.clamp_min(1e-12)),
    }


def aggregate_location_metrics(by_location: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate only already-computed fixed-location metrics."""
    keys = sorted({name for values in by_location.values() for name, value in values.items() if isinstance(value, (int, float)) and np.isfinite(value)})
    return {name: {"mean": float(np.mean([values[name] for values in by_location.values() if name in values and np.isfinite(values[name])])), "median": float(np.median([values[name] for values in by_location.values() if name in values and np.isfinite(values[name])])), "p10": float(np.percentile([values[name] for values in by_location.values() if name in values and np.isfinite(values[name])], 10)), "p90": float(np.percentile([values[name] for values in by_location.values() if name in values and np.isfinite(values[name])], 90))} for name in keys}
