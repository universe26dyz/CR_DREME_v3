from __future__ import annotations

import numpy as np
import torch


def dominant_frequency_hz(values: torch.Tensor, timestamps_s: torch.Tensor) -> float:
    signal = values.detach().reshape(values.shape[0], -1).mean(-1).double().cpu().numpy()
    times = timestamps_s.detach().reshape(-1).double().cpu().numpy()
    order = np.argsort(times)
    times, signal = times[order], signal[order]
    unique_times, inverse = np.unique(times, return_inverse=True)
    if len(unique_times) != len(times):
        reduced = np.zeros(len(unique_times), dtype=signal.dtype)
        np.add.at(reduced, inverse, signal)
        signal = reduced / np.bincount(inverse)
        times = unique_times
    if len(signal) < 4 or np.ptp(times) <= 0:
        return float("nan")
    spacing = float(np.median(np.diff(times)))
    centered = signal - signal.mean()
    frequencies = np.fft.rfftfreq(len(centered), d=spacing)
    power = np.square(np.abs(np.fft.rfft(centered)))
    if len(power) <= 1:
        return float("nan")
    return float(frequencies[1 + np.argmax(power[1:])])


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
