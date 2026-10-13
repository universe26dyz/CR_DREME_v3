"""Per-location synthetic image observability metrics for Exp1-B."""
from __future__ import annotations

import torch


def legacy_flattened_temporal_std(images: torch.Tensor) -> float:
    """The intentionally retained Exp1 legacy statistic, used only as a control."""
    return float(images.reshape(images.shape[0], -1).std(unbiased=False))


def _quantiles(values: torch.Tensor, prefix: str) -> dict[str, float]:
    return {
        f"image_true_temporal_std_mean_{prefix}": float(values.mean()),
        f"image_true_temporal_std_median_{prefix}": float(values.median()),
        f"image_true_temporal_std_p10_{prefix}": float(torch.quantile(values, 0.10)),
        f"image_true_temporal_std_p90_{prefix}": float(torch.quantile(values, 0.90)),
    }


def image_sequence_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Metrics for one fixed (view, slice_id) sequence, never mixed locations."""
    if predicted.shape != target.shape or predicted.ndim != 4:
        raise ValueError("images must be matched [time, channel, height, width]")
    pred, truth = predicted[:, 0], target[:, 0]
    result = {
        **_quantiles(pred.std(dim=0, unbiased=False), "pred"),
        **_quantiles(truth.std(dim=0, unbiased=False), "gt"),
        "per_frame_mse": float((pred - truth).square().mean()),
    }
    if pred.shape[0] < 2:
        return {**result, "consecutive_delta_cosine": float("nan"), "consecutive_delta_rms": float("nan")}
    delta_pred, delta_truth = pred[1:] - pred[:-1], truth[1:] - truth[:-1]
    denominator = delta_pred.norm() * delta_truth.norm()
    result["consecutive_delta_cosine"] = float((delta_pred * delta_truth).sum() / denominator.clamp_min(1e-12))
    result["consecutive_delta_rms"] = float(delta_pred.square().mean().sqrt())
    return result


def normalized_cross_correlation(predicted: torch.Tensor, target: torch.Tensor) -> float:
    x, y = predicted.reshape(-1), target.reshape(-1)
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-12))
