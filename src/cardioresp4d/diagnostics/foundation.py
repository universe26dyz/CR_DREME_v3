"""Read-only, paired foundation-reconstruction audit helpers."""
from __future__ import annotations

from typing import Callable

import torch

from cardioresp4d.diagnostics.change5c import diagnostic_rng, stable_diagnostic_seed


def slice_pixels(observation) -> torch.Tensor:
    height, width = observation.image.shape[-2:]
    linear = torch.arange(height * width, device=observation.image.device)
    return torch.stack((linear.remainder(width), torch.div(linear, width, rounding_mode="floor")), -1).to(observation.image.dtype)


def _chunked(pixels: torch.Tensor, chunk_size: int, render: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError("slice_chunk_size must be positive")
    return torch.cat([render(pixels[start:start + chunk_size]) for start in range(0, len(pixels), chunk_size)])


def foundation_predictions(model, observation, *, joint_stage: str, seed: int, psf_seed: int | None = None, slice_chunk_size: int = 1024) -> dict[str, torch.Tensor]:
    """Render matched direct, PSF, respiratory, and joint slice predictions."""
    height, width = observation.image.shape[-2:]
    pixels = slice_pixels(observation)
    points = model._pixel_world(observation, pixels)
    direct = _chunked(pixels, slice_chunk_size, lambda chunk: model.canonical(model._pixel_world(observation, chunk)))
    resolution = torch.tensor([observation.pixel_spacing_mm[1], observation.pixel_spacing_mm[0], observation.slice_thickness_mm], device=points.device, dtype=points.dtype)
    psf_seed = stable_diagnostic_seed(seed, observation.view, observation.slice_id, observation.dynamic_frame_id, purpose="psf") if psf_seed is None else psf_seed

    def canonical_psf(chunk: torch.Tensor) -> torch.Tensor:
        centers = model._pixel_world(observation, chunk)
        count = centers.shape[0]
        return model.psf(model.canonical, centers, resolution.expand(count, -1), row_direction=observation.row_direction.to(centers).expand(count, -1), column_direction=observation.column_direction.to(centers).expand(count, -1), normal=observation.normal.to(centers).expand(count, -1), motion=None)["predicted_intensity"]

    with diagnostic_rng(psf_seed, observation.image.device):
        psf = _chunked(pixels, slice_chunk_size, canonical_psf)
    stage_outputs = {}
    for name, stage in (("resp_only", "stage2c"), ("resp_plus_card", joint_stage)):
        with diagnostic_rng(psf_seed, observation.image.device):
            stage_outputs[name] = _chunked(pixels, slice_chunk_size, lambda chunk, stage=stage: model.predict(observation, chunk, stage)["predicted_intensity"])
    return {"canonical_direct": direct.reshape(height, width), "canonical_psf": psf.reshape(height, width), **{name: value.reshape(height, width) for name, value in stage_outputs.items()}}


def cardiac_intersection(model, observation) -> torch.Tensor:
    pixels = slice_pixels(observation)
    points = model._pixel_world(observation, pixels)
    return ((points >= model.cardiac_lower_world_mm.to(points)) & (points <= model.cardiac_upper_world_mm.to(points))).all(-1).reshape(observation.image.shape[-2:])


def _ncc(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left, right = left.reshape(-1).float(), right.reshape(-1).float()
    centered_left, centered_right = left - left.mean(), right - right.mean()
    denominator = torch.linalg.vector_norm(centered_left) * torch.linalg.vector_norm(centered_right)
    return None if float(denominator) == 0. else float((centered_left @ centered_right / denominator).cpu())


def _gradient_energy(image: torch.Tensor) -> float:
    image = image.float()
    if image.ndim < 2:
        return 0.
    return float((image.diff(dim=-2).square().mean() + image.diff(dim=-1).square().mean()).cpu())


def _region_metrics(acquired: torch.Tensor, predicted: torch.Tensor) -> dict[str, float | None]:
    residual = acquired - predicted
    return {"mse": float(residual.square().mean().cpu()), "mae": float(residual.abs().mean().cpu()), "ncc": _ncc(acquired, predicted), "gradient_energy": _gradient_energy(predicted)}


def image_metrics(acquired: torch.Tensor, predicted: torch.Tensor, cardiac_mask: torch.Tensor) -> dict[str, dict[str, float | None]]:
    if acquired.shape != predicted.shape or acquired.shape != cardiac_mask.shape:
        raise ValueError("matched acquired, predicted, and cardiac-mask image shapes required")
    result = {"whole_fov": _region_metrics(acquired, predicted)}
    result["cardiac_intersection"] = _region_metrics(acquired[cardiac_mask], predicted[cardiac_mask]) if cardiac_mask.any() else {"mse": None, "mae": None, "ncc": None, "gradient_energy": None}
    return result


def temporal_std_map(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 3 or images.shape[0] < 2:
        raise ValueError("temporal images must be [T,H,W] with T >= 2")
    return images.float().std(dim=0, unbiased=False)


def temporal_metrics(acquired: torch.Tensor, predicted: torch.Tensor, cardiac_mask: torch.Tensor) -> dict[str, dict[str, float | None]]:
    if acquired.shape != predicted.shape or acquired.ndim != 3 or acquired.shape[1:] != cardiac_mask.shape:
        raise ValueError("matched [T,H,W] acquired/predicted and [H,W] cardiac mask required")
    acquired_delta, predicted_delta = acquired[1:] - acquired[:-1], predicted[1:] - predicted[:-1]

    def metrics(mask: torch.Tensor | None) -> dict[str, float | None]:
        acquired_values = acquired if mask is None else acquired[:, mask]
        predicted_values = predicted if mask is None else predicted[:, mask]
        acquired_changes = acquired_delta if mask is None else acquired_delta[:, mask]
        predicted_changes = predicted_delta if mask is None else predicted_delta[:, mask]
        acquired_std = float(acquired_values.float().std(dim=0, unbiased=False).mean().cpu())
        predicted_std = float(predicted_values.float().std(dim=0, unbiased=False).mean().cpu())
        return {"mean_temporal_std": predicted_std, "acquired_mean_temporal_std": acquired_std, "mean_abs_consecutive_change": float(predicted_changes.abs().mean().cpu()), "delta_correlation": _ncc(acquired_changes, predicted_changes)}

    return {"whole_fov": metrics(None), "cardiac_intersection": metrics(cardiac_mask) if cardiac_mask.any() else {"mean_temporal_std": None, "acquired_mean_temporal_std": None, "mean_abs_consecutive_change": None, "delta_correlation": None}}
