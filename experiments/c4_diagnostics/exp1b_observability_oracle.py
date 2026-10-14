"""C4 Exp1-B: synthetic observability, direct MBC, and timestamp-oracle tests.

This module intentionally owns all orchestration; production C4 modules remain
read-only dependencies.  Temporal results are formed per fixed view/slice
before their robust aggregate is written.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import torch
import yaml

from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField
from cardioresp4d.training.model import SourceFirstDynamicModel

from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import (
    _build_observations, _load_domain, fallback_geometry, geometry_from_manifest,
)
from experiments.c4_diagnostics.synthetic.metrics import aggregate_location_metrics, dvf_metrics, location_spectral_metrics, vector_field_agreement_metrics
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.observability import image_sequence_metrics, normalized_cross_correlation
from experiments.c4_diagnostics.synthetic.oracle import DirectMBCFit, OracleScoreGenerator, OracleScoreScales, TargetPSFOffsetCache, build_oracle_motion_model, build_target_psf_offset_cache, derive_oracle_score_scales
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from experiments.c4_diagnostics.synthetic.render import GeometryFrame, pixel_centers, rescale_frame_for_rendering, render_synthetic_observation


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is non-empty: {path}; pass --overwrite to reuse it")
        for item in path.iterdir():
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    path.mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(exist_ok=True)


def _group(observations: list, frames: list[GeometryFrame]) -> dict[str, list[tuple[int, object, GeometryFrame]]]:
    grouped: dict[str, list[tuple[int, object, GeometryFrame]]] = {}
    for index, (observation, frame) in enumerate(zip(observations, frames)):
        grouped.setdefault(f"{observation.view}/{observation.slice_id}", []).append((index, observation, frame))
    return {key: sorted(value, key=lambda item: item[1].timestamp_s) for key, value in grouped.items()}


def _motion_for_branch(motion: GroundTruthMotion, branch: str) -> GroundTruthMotion:
    if branch == "resp":
        return replace(motion, cardiac_amplitude_mm=0.0)
    if branch == "card":
        return replace(motion, respiratory_amplitude_mm=0.0)
    if branch == "zero":
        return replace(motion, respiratory_amplitude_mm=0.0, cardiac_amplitude_mm=0.0)
    return motion


def _render_sequence(sequence: list[tuple[int, object, GeometryFrame]], *, image_shape: tuple[int, int], phantom: AnalyticCanonicalPhantom, motion: GroundTruthMotion, psf_samples: int, seed: int) -> torch.Tensor:
    images = []
    for index, _, frame in sequence:
        rendered = render_synthetic_observation(frame, height=image_shape[0], width=image_shape[1], phantom=phantom, motion=motion, psf_samples=psf_samples, seed=seed + index, dynamic_frame_id=index)
        images.append(rendered.image)
    return torch.stack(images)


@contextmanager
def _psf_rng(device: torch.device, seed: int):
    devices = [device.index if device.index is not None else 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
                yield
        else:
            yield


def _oracle_render(model, observation, scores: dict[str, torch.Tensor], branch: str, pixel_uv: torch.Tensor, *, psf_training_mode: str, target_seed: int, psf_offset_cache: TargetPSFOffsetCache | None = None) -> torch.Tensor:
    if psf_training_mode == "matched_crn":
        if psf_offset_cache is None:
            raise ValueError("matched_crn requires cached full-image target PSF offsets")
        return model.render(observation, scores, branch, pixel_uv, standard_normal_offsets=psf_offset_cache.for_pixel_uv(pixel_uv))
    if psf_training_mode == "stochastic":
        return model.render(observation, scores, branch, pixel_uv)
    raise ValueError("psf_training_mode must be matched_crn or stochastic")


def _full_image_uv(observation) -> torch.Tensor:
    """Return row-major image pixels as ``(u, v)`` coordinates."""
    height, width = observation.image.shape[-2:]
    return torch.stack(
        torch.meshgrid(
            torch.arange(width, device=observation.image.device),
            torch.arange(height, device=observation.image.device),
            indexing="xy",
        ),
        dim=-1,
    ).reshape(-1, 2).to(dtype=observation.image.dtype)


def _render_full_image_chunked(model, observation, scores: dict[str, torch.Tensor], branch: str, *, psf_training_mode: str, target_seed: int, pixel_chunk: int, psf_offset_cache: TargetPSFOffsetCache | None = None) -> torch.Tensor:
    """Render one image without presenting all pixels as one MBC batch."""
    if pixel_chunk < 1:
        raise ValueError("evaluation pixel chunk must be positive")
    chunks = []
    for pixel_uv in _full_image_uv(observation).split(pixel_chunk):
        chunks.append(
            _oracle_render(
                model,
                observation,
                scores,
                branch,
                pixel_uv,
                psf_training_mode=psf_training_mode,
                target_seed=target_seed,
                psf_offset_cache=psf_offset_cache,
            )
        )
    return torch.cat(chunks)


def _center_support_metrics(predicted: torch.Tensor, target: torch.Tensor, pixel_centers_world_mm: torch.Tensor, *, tolerance_mm: float = 1e-4) -> dict[str, float]:
    """Temporal DVF metrics only after proving pixel centres correspond in time."""
    if predicted.shape != target.shape or predicted.shape != pixel_centers_world_mm.shape or predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError("center-support fields and pixel-center world coordinates must be [time, pixels, xyz]")
    reference = pixel_centers_world_mm[0]
    if not torch.allclose(pixel_centers_world_mm, reference.expand_as(pixel_centers_world_mm), atol=tolerance_mm, rtol=0.0):
        raise ValueError("pixel-center world coordinates differ across frames of one fixed location")
    values = dvf_metrics(predicted, target)
    values["normalized_epe"] = values["endpoint_error_mm"] / max(values["gt_dynamic_rms_mm"], 1e-12)
    values["point_count"] = float(predicted.shape[1])
    return values


def _respiratory_support_audit(model, sequence: list[tuple[int, object, GeometryFrame]], psf_offset_caches: dict[int, TargetPSFOffsetCache], motion: GroundTruthMotion, generator: OracleScoreGenerator, *, evaluation_pixel_chunk: int) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Audit respiratory fields on pixel centres and exact target PSF samples."""
    center_pred: list[torch.Tensor] = []; center_gt: list[torch.Tensor] = []; center_points: list[torch.Tensor] = []
    psf_pred: list[torch.Tensor] = []; psf_gt: list[torch.Tensor] = []
    for _, observation, _ in sequence:
        timestamp = observation.image.new_tensor([observation.timestamp_s])
        learned_motion = model.motion(generator(timestamp), "resp")
        frame_center_pred: list[torch.Tensor] = []; frame_center_gt: list[torch.Tensor] = []; frame_points: list[torch.Tensor] = []
        frame_psf_pred: list[torch.Tensor] = []; frame_psf_gt: list[torch.Tensor] = []
        for pixel_uv in _full_image_uv(observation).split(evaluation_pixel_chunk):
            points = SourceFirstDynamicModel._pixel_world(observation, pixel_uv.to(observation.image))
            predicted = learned_motion.respiratory_field(points[None])[0]
            target = motion.respiratory_dvf_mm(points[None], timestamp)[0]
            samples = model.psf_sample_world_points(observation, pixel_uv, psf_offset_caches[observation.dynamic_frame_id].for_pixel_uv(pixel_uv))
            psf_predicted = learned_motion.respiratory_field(samples[None])[0]
            psf_target = motion.respiratory_dvf_mm(samples[None], timestamp)[0]
            frame_center_pred.append(predicted.detach().cpu()); frame_center_gt.append(target.detach().cpu()); frame_points.append(points.detach().cpu())
            frame_psf_pred.append(psf_predicted.detach().cpu()); frame_psf_gt.append(psf_target.detach().cpu())
        center_pred.append(torch.cat(frame_center_pred)); center_gt.append(torch.cat(frame_center_gt)); center_points.append(torch.cat(frame_points))
        psf_pred.append(torch.cat(frame_psf_pred)); psf_gt.append(torch.cat(frame_psf_gt))
    center_prediction, center_target, center_world = torch.stack(center_pred), torch.stack(center_gt), torch.stack(center_points)
    psf_prediction, psf_target = torch.stack(psf_pred), torch.stack(psf_gt)
    center = _center_support_metrics(center_prediction, center_target, center_world)
    psf = vector_field_agreement_metrics(psf_prediction, psf_target)
    values = {f"resp_center_support_{key}": value for key, value in center.items()}
    values.update({f"resp_psf_support_{key}": value for key, value in psf.items()})
    values["resp_psf_support_point_count"] = float(psf_prediction.shape[0] * psf_prediction.shape[1] * psf_prediction.shape[2])
    stem = f"oracle_{sequence[0][1].view}_{sequence[0][1].slice_id}".replace("/", "_")
    arrays = {
        f"{stem}_center_support_pred_dvf_mm": center_prediction.numpy(),
        f"{stem}_center_support_gt_dvf_mm": center_target.numpy(),
        f"{stem}_psf_support_pred_dvf_mm": psf_prediction.numpy(),
        f"{stem}_psf_support_gt_dvf_mm": psf_target.numpy(),
    }
    return values, arrays


def _save_oracle_checkpoint(model, path: Path, metadata: dict[str, object]) -> None:
    """Persist a CPU-only diagnostic model snapshot for read-only audits."""
    payload = {
        "checkpoint_schema": "c4_exp1b_oracle_last_v1",
        **metadata,
        "model_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
    }
    torch.save(payload, path)


def _roi_mse(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    selected = (predicted - target).square()[mask]
    return float(selected.mean()) if selected.numel() else float("nan")


def _cardiac_support_mask(sequence: list[tuple[int, object, GeometryFrame]], motion: GroundTruthMotion, image_shape: tuple[int, int]) -> torch.Tensor:
    masks = []
    for _, _, frame in sequence:
        rendered = rescale_frame_for_rendering(frame, height=image_shape[0], width=image_shape[1])
        support = motion.cardiac_support(pixel_centers(rendered, *image_shape)).reshape(1, *image_shape)
        masks.append(support > 1e-4)
    return torch.stack(masks)


def _aggregate_by_amplitude_scale(rows: dict[str, dict[str, float]]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, dict[str, dict[str, float]]] = {}
    for location, values in rows.items():
        scale = f"{float(values['amplitude_scale']):g}"
        grouped.setdefault(scale, {})[location] = values
    return {scale: aggregate_location_metrics(values) for scale, values in grouped.items()}


def _dynamic_context(sequence: list[tuple[int, object, GeometryFrame]], motion: GroundTruthMotion, image_shape: tuple[int, int]) -> dict[str, float]:
    frame = rescale_frame_for_rendering(sequence[0][2], height=image_shape[0], width=image_shape[1])
    points = pixel_centers(frame, *image_shape)[None]
    timestamps = points.new_tensor([item[1].timestamp_s for item in sequence])
    tiled = points.expand(len(sequence), -1, -1)
    resp = motion.respiratory_dvf_mm(tiled, timestamps)
    card = motion.cardiac_dvf_mm(tiled, timestamps)
    resp_dynamic = resp - resp.mean(0, keepdim=True)
    card_dynamic = card - card.mean(0, keepdim=True)
    inplane = float(frame.pixel_spacing_mm.mean())
    resp_rms, card_rms = float(resp_dynamic.square().mean().sqrt()), float(card_dynamic.square().mean().sqrt())
    return {"rendered_row_spacing_mm": float(frame.pixel_spacing_mm[0]), "rendered_column_spacing_mm": float(frame.pixel_spacing_mm[1]), "inplane_spacing_mm": inplane, "gt_resp_dynamic_dvf_rms_mm": resp_rms, "gt_card_dynamic_dvf_rms_mm": card_rms, "resp_dvf_to_pixel_ratio": resp_rms / max(inplane, 1e-12), "card_dvf_to_pixel_ratio": card_rms / max(inplane, 1e-12)}


def observability_audit(observations: list, frames: list[GeometryFrame], *, image_shape: tuple[int, int], phantom: AnalyticCanonicalPhantom, motion: GroundTruthMotion, psf_samples: int, seed: int, amplitude_scales: list[float]) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    rows: dict[str, dict[str, float]] = {}
    arrays: dict[str, np.ndarray] = {}
    for scale in amplitude_scales:
        scaled = replace(motion, respiratory_amplitude_mm=motion.respiratory_amplitude_mm * scale, cardiac_amplitude_mm=motion.cardiac_amplitude_mm * scale)
        for location, sequence in _group(observations, frames).items():
            target = _render_sequence(sequence, image_shape=image_shape, phantom=phantom, motion=scaled, psf_samples=psf_samples, seed=seed)
            variants = {"gt_motion": scaled, "zero_motion": _motion_for_branch(scaled, "zero"), "resp_only_gt": _motion_for_branch(scaled, "resp"), "card_only_gt": _motion_for_branch(scaled, "card")}
            values: dict[str, float] = {"amplitude_scale": float(scale), **_dynamic_context(sequence, scaled, image_shape)}
            predictions: dict[str, torch.Tensor] = {}
            for name, variant in variants.items():
                prediction = _render_sequence(sequence, image_shape=image_shape, phantom=phantom, motion=variant, psf_samples=psf_samples, seed=seed)
                independent = _render_sequence(sequence, image_shape=image_shape, phantom=phantom, motion=variant, psf_samples=psf_samples, seed=seed + 100_000)
                metrics = image_sequence_metrics(prediction, target)
                values.update({f"{name}_{metric}": value for metric, value in metrics.items()})
                values[f"{name}_ncc"] = normalized_cross_correlation(prediction, target)
                values[f"{name}_psf_self_noise_mse"] = float((prediction - independent).square().mean())
                if prediction.shape[0] > 1:
                    values[f"{name}_psf_self_noise_delta_rms"] = float(((prediction[1:] - prediction[:-1]) - (independent[1:] - independent[:-1])).square().mean().sqrt())
                else:
                    values[f"{name}_psf_self_noise_delta_rms"] = float("nan")
                predictions[name] = prediction
                arrays[f"observability_{scale:g}_{location.replace('/', '_')}_{name}"] = prediction.detach().cpu().numpy()
            gt_mse = values["gt_motion_per_frame_mse"]
            zero_mse = values["zero_motion_per_frame_mse"]
            values["zero_vs_gt_mse_gap"] = zero_mse - gt_mse
            values["resp_only_vs_gt_mse_gap"] = values["resp_only_gt_per_frame_mse"] - gt_mse
            values["card_only_vs_gt_mse_gap"] = values["card_only_gt_per_frame_mse"] - gt_mse
            values["zero_motion_observability_fraction"] = (zero_mse - gt_mse) / max(zero_mse, 1e-12)
            target_variance = float(target[:, 0].var(dim=0, unbiased=False).mean())
            target_delta = target[1:] - target[:-1]
            zero_prediction = predictions["zero_motion"]
            delta_error = float(((zero_prediction[1:] - zero_prediction[:-1] - target_delta).square().mean().sqrt()))
            target_delta_rms = float(target_delta.square().mean().sqrt())
            values["target_temporal_variance_mean"] = target_variance
            values["zero_motion_mse_over_target_temporal_variance"] = zero_mse / max(target_variance, 1e-12)
            values["zero_motion_delta_error_over_gt_delta_rms"] = delta_error / max(target_delta_rms, 1e-12)
            respiratory_effect = values["card_only_gt_per_frame_mse"]
            cardiac_effect = values["resp_only_gt_per_frame_mse"]
            psf_noise = values["gt_motion_psf_self_noise_mse"]
            values["whole_fov_respiratory_effect_mse"] = respiratory_effect
            values["whole_fov_cardiac_effect_mse"] = cardiac_effect
            values["respiratory_effect_to_psf_noise_ratio"] = respiratory_effect / max(psf_noise, 1e-12)
            values["cardiac_effect_to_psf_noise_ratio"] = cardiac_effect / max(psf_noise, 1e-12)
            mask = _cardiac_support_mask(sequence, scaled, image_shape).to(target.device)
            values["cardiac_support_roi_fraction"] = float(mask.float().mean())
            values["cardiac_support_roi_respiratory_effect_mse"] = _roi_mse(predictions["card_only_gt"], target, mask)
            values["cardiac_support_roi_cardiac_effect_mse"] = _roi_mse(predictions["resp_only_gt"], target, mask)
            roi_noise = _roi_mse(predictions["gt_motion"], _render_sequence(sequence, image_shape=image_shape, phantom=phantom, motion=scaled, psf_samples=psf_samples, seed=seed + 100_000), mask)
            values["cardiac_support_roi_psf_self_noise_mse"] = roi_noise
            values["cardiac_support_roi_respiratory_effect_to_psf_noise_ratio"] = values["cardiac_support_roi_respiratory_effect_mse"] / max(roi_noise, 1e-12) if math.isfinite(values["cardiac_support_roi_respiratory_effect_mse"]) else float("nan")
            values["cardiac_support_roi_cardiac_effect_to_psf_noise_ratio"] = values["cardiac_support_roi_cardiac_effect_mse"] / max(roi_noise, 1e-12) if math.isfinite(values["cardiac_support_roi_cardiac_effect_mse"]) else float("nan")
            rows[f"{location}|scale={scale:g}"] = values
    return rows, arrays


def _source_mbc_shapes(config: dict[str, object], smoke: bool) -> tuple[int, tuple[tuple[int, int, int], ...], tuple[int, int, int], int]:
    if smoke:
        return 4, ((2, 2, 2),) * 3, (2, 2, 2), 1
    model = config["model"]
    return int(model["respiratory_mbc"]["hidden_dim"]), tuple(tuple(int(x) for x in shape) for shape in model["respiratory_mbc"]["logical_control_shapes"]), tuple(int(x) for x in model["cardiac_mbc"]["logical_control_shape"]), int(model["psf"]["n_samples"])


def _normalization_grid_shapes(config: dict[str, object], smoke: bool) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if smoke:
        return (3, 3, 3), (3, 3, 3)
    regularization = config["training"]["motion_regularization"]
    return tuple(int(value) for value in regularization["respiratory_evaluation_grid"]["shape"]), tuple(int(value) for value in regularization["cardiac_evaluation_grid"]["shape"])


def _direct_branch_learning_rate(config: dict[str, object], branch: str) -> float:
    optimizer = config["training"]["optimizer"]
    if branch == "resp":
        return float(optimizer["stage2"]["respiratory_mbc_lr"])
    if branch == "card":
        return float(optimizer["stage3a"]["cardiac_mbc_lr"])
    raise ValueError("direct branch must be resp or card")


def _direct_fit(domain: dict[str, object], phantom: AnalyticCanonicalPhantom, motion: GroundTruthMotion, config: dict[str, object], score_scales: OracleScoreScales, *, device: torch.device, steps: int, smoke: bool) -> dict[str, dict[str, float]]:
    hidden, resp_shapes, card_shape, psf_samples = _source_mbc_shapes(config, smoke)
    normalization_grid_shapes = _normalization_grid_shapes(config, smoke)
    normalization_weight = float(config["training"]["loss_weights"]["mbc_normalization"])
    output: dict[str, dict[str, float]] = {}
    for branch in ("resp", "card"):
        model = build_oracle_motion_model(domain, phantom, device=device, active_respiratory_levels=3, hidden_dim=hidden, respiratory_shapes=resp_shapes, cardiac_shape=card_shape, psf_samples=psf_samples, normalization_grid_shapes=normalization_grid_shapes)
        lr = _direct_branch_learning_rate(config, branch)
        fitter = DirectMBCFit(model, motion, branch=branch, grid_shape=(3, 3, 3) if smoke else normalization_grid_shapes[0 if branch == "resp" else 1], lr=lr, normalization_weight=normalization_weight, score_scales=score_scales)
        for _ in range(steps):
            fitter.step()
        output[branch] = {**fitter.metrics(), "learning_rate": lr}
    return output


def _oracle_optimizer(model, branch: str, config: dict[str, object]) -> torch.optim.Optimizer:
    optimizer = config["training"]["optimizer"]
    if branch == "resp":
        return torch.optim.Adam(model.respiratory_mbc.parameters(), lr=float(optimizer["stage2"]["respiratory_mbc_lr"]))
    if branch == "card":
        return torch.optim.Adam(model.cardiac_mbc.parameters(), lr=float(optimizer["stage3a"]["cardiac_mbc_lr"]))
    if branch == "combined":
        return torch.optim.Adam([
            {"params": model.respiratory_mbc.parameters(), "lr": float(optimizer["stage3b"]["respiratory_mbc_lr"])},
            {"params": model.cardiac_mbc.parameters(), "lr": float(optimizer["stage3b"]["cardiac_mbc_lr"])},
        ])
    raise ValueError("branch must be resp, card, or combined")


def _uv(observation, count: int, generator: torch.Generator) -> torch.Tensor:
    height, width = observation.image.shape[-2:]
    return torch.stack((torch.randint(width, (count,), generator=generator, device=observation.image.device), torch.randint(height, (count,), generator=generator, device=observation.image.device)), dim=-1).float()


def _oracle_target_observations(frames: list[GeometryFrame], *, image_shape: tuple[int, int], phantom: AnalyticCanonicalPhantom, motion: GroundTruthMotion, branch: str, psf_samples: int, seed: int) -> tuple[list, dict[int, TargetPSFOffsetCache]]:
    observations = _build_observations(frames, image_shape=image_shape, phantom=phantom, motion=_motion_for_branch(motion, branch), psf_samples=psf_samples, seed=seed)
    return observations, {item.dynamic_frame_id: build_target_psf_offset_cache(item, psf_samples=psf_samples, seed=seed + item.dynamic_frame_id) for item in observations}


def _unavailable_spectral_metrics(channel_count: int) -> dict[str, float]:
    return {
        "spectral_available": 0.0,
        "channel_count": float(channel_count),
        "dominant_frequency_hz": float("nan"),
        "dominant_in_target_band": float("nan"),
        "target_band_power": float("nan"),
        "wrong_band_power": float("nan"),
        "correct_band_fraction": float("nan"),
        "wrong_band_fraction": float("nan"),
    }


def _oracle_spectral_metrics(scores: torch.Tensor, timestamps_s: torch.Tensor, frequency_hz: float, *, formal_sequence: bool) -> dict[str, float]:
    """Run one branch's NUDFT diagnostic only when its target is supported."""
    band = (frequency_hz - .05, frequency_hz + .05)
    try:
        ordered = timestamps_s.sort().values
        if ordered.numel() < 3:
            raise ValueError("need at least three timestamped frames")
        median_delta = ordered.diff().median()
        if not torch.isfinite(median_delta) or not float(median_delta) > 0:
            raise ValueError("timestamps need positive median spacing")
        if band[1] > .5 / float(median_delta):
            raise ValueError("target band exceeds timestamp Nyquist support")
        return {"spectral_available": 1.0, **location_spectral_metrics(scores, timestamps_s, [band])}
    except ValueError:
        if formal_sequence:
            raise
        return _unavailable_spectral_metrics(scores.reshape(scores.shape[0], -1).shape[1])


@torch.inference_mode()
def _oracle_evaluation(model, observations: list, psf_offset_caches: dict[int, TargetPSFOffsetCache], motion: GroundTruthMotion, generator: OracleScoreGenerator, branch: str, psf_training_mode: str, domain: dict[str, object], pixel_samples: int, evaluation_pixel_chunk: int) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    del domain, pixel_samples
    grouped = _group(observations, [GeometryFrame(item.view, item.slice_id, item.center_mm, item.row_direction, item.column_direction, item.normal, item.pixel_spacing_mm, item.slice_thickness_mm, item.timestamp_s, item.image.shape[-2], item.image.shape[-1]) for item in observations])
    result: dict[str, dict[str, float]] = {}; arrays: dict[str, np.ndarray] = {}
    resp_grid, _ = SourceFirstDynamicModel._physical_grid(model.lower_world_mm, model.upper_world_mm, (4, 4, 4))
    card_grid, _ = SourceFirstDynamicModel._physical_grid(model.cardiac_lower_world_mm, model.cardiac_upper_world_mm, (4, 4, 4))
    for location, sequence in grouped.items():
        predicted_images: list[torch.Tensor] = []; independent_images: list[torch.Tensor] = []; target_images: list[torch.Tensor] = []; resp_pred: list[torch.Tensor] = []; resp_gt: list[torch.Tensor] = []; card_pred: list[torch.Tensor] = []; card_gt: list[torch.Tensor] = []; total_pred: list[torch.Tensor] = []; total_gt: list[torch.Tensor] = []; ts: list[float] = []
        for _, observation, _ in sequence:
            timestamp = resp_grid.new_tensor([observation.timestamp_s]); scores = generator(timestamp); learned_motion = model.motion(scores, branch)
            if psf_training_mode == "matched_crn":
                prediction = _render_full_image_chunked(model, observation, scores, branch, psf_training_mode="matched_crn", target_seed=observation.dynamic_frame_id, psf_offset_cache=psf_offset_caches[observation.dynamic_frame_id], pixel_chunk=evaluation_pixel_chunk)
                with _psf_rng(resp_grid.device, 1000 + observation.dynamic_frame_id):
                    independent = _render_full_image_chunked(model, observation, scores, branch, psf_training_mode="stochastic", target_seed=1000 + observation.dynamic_frame_id, pixel_chunk=evaluation_pixel_chunk)
                independent_images.append(independent.reshape(1, *observation.image.shape[-2:]))
            else:
                with _psf_rng(resp_grid.device, 1000 + observation.dynamic_frame_id):
                    prediction = _render_full_image_chunked(model, observation, scores, branch, psf_training_mode="stochastic", target_seed=1000 + observation.dynamic_frame_id, pixel_chunk=evaluation_pixel_chunk)
            predicted_images.append(prediction.reshape(1, *observation.image.shape[-2:]))
            target_images.append(observation.image)
            if branch in {"resp", "combined"}:
                resp_pred.append(learned_motion.respiratory_field(resp_grid)); resp_gt.append(motion.respiratory_dvf_mm(resp_grid, timestamp))
            if branch in {"card", "combined"}:
                card_pred.append(learned_motion.cardiac_field(card_grid)); card_gt.append(motion.cardiac_dvf_mm(card_grid, timestamp))
            if branch == "combined":
                total_pred.append(learned_motion(resp_grid)["reference_points_mm"] - resp_grid)
                total_gt.append(motion.pullback_reference_mm(resp_grid, timestamp) - resp_grid)
            ts.append(observation.timestamp_s)
        image_prefix = "paired_psf" if psf_training_mode == "matched_crn" else "stochastic_psf"
        values = {f"{image_prefix}_{key}": value for key, value in image_sequence_metrics(torch.stack(predicted_images), torch.stack(target_images)).items()}
        if independent_images:
            values.update({f"independent_psf_{key}": value for key, value in image_sequence_metrics(torch.stack(independent_images), torch.stack(target_images)).items()})
        for label, predicted, target in (("resp_dvf", resp_pred, resp_gt), ("card_dvf", card_pred, card_gt), ("total_dvf", total_pred, total_gt)):
            if predicted:
                dvf_values = dvf_metrics(torch.cat(predicted), torch.cat(target))
                dvf_values["normalized_epe"] = dvf_values["endpoint_error_mm"] / max(dvf_values["gt_dynamic_rms_mm"], 1e-12)
                values.update({f"{label}_{key}": value for key, value in dvf_values.items()})
        if branch in {"resp", "combined"}:
            support_values, support_arrays = _respiratory_support_audit(model, sequence, psf_offset_caches, motion, generator, evaluation_pixel_chunk=evaluation_pixel_chunk)
            values.update(support_values); arrays.update(support_arrays)
        all_scores = generator(torch.as_tensor(ts, device=resp_grid.device))
        score_branches = (("resp", all_scores["resp_scores"], motion.respiratory_frequency_hz),) if branch == "resp" else (("card", all_scores["card_scores"], motion.cardiac_frequency_hz),) if branch == "card" else (("resp", all_scores["resp_scores"], motion.respiratory_frequency_hz), ("card", all_scores["card_scores"], motion.cardiac_frequency_hz))
        for score_name, score_values, frequency in score_branches:
            score_metrics = _oracle_spectral_metrics(score_values, torch.as_tensor(ts, device=resp_grid.device), frequency, formal_sequence=len(ts) >= 50)
            values.update({f"oracle_{score_name}_score_{key}": value for key, value in score_metrics.items()})
        result[location] = values
        stem = location.replace("/", "_"); arrays[f"oracle_{stem}_timestamps_s"] = np.asarray(ts); arrays[f"oracle_{stem}_predicted_images"] = torch.stack(predicted_images).detach().cpu().numpy()
    return result, arrays


def _oracle_run(domain: dict[str, object], frames: list[GeometryFrame], *, image_shape: tuple[int, int], phantom: AnalyticCanonicalPhantom, motion: GroundTruthMotion, config: dict[str, object], score_scales: OracleScoreScales, branch: str, device: torch.device, steps: int, pixel_samples: int, observations_per_update: int, psf_training_mode: str, evaluation_pixel_chunk: int, checkpoint_path: Path, checkpoint_metadata: dict[str, object], seed: int, smoke: bool) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], list[dict[str, float]]]:
    hidden, resp_shapes, card_shape, psf_samples = _source_mbc_shapes(config, smoke)
    normalization_grid_shapes = _normalization_grid_shapes(config, smoke)
    observations, psf_offset_caches = _oracle_target_observations(frames, image_shape=image_shape, phantom=phantom, motion=motion, branch=branch, psf_samples=psf_samples, seed=seed)
    model = build_oracle_motion_model(domain, phantom, device=device, active_respiratory_levels=3, hidden_dim=hidden, respiratory_shapes=resp_shapes, cardiac_shape=card_shape, psf_samples=psf_samples, normalization_grid_shapes=normalization_grid_shapes)
    train_branch = {"resp": "resp", "card": "card", "combined": "combined"}[branch]
    model.configure_trainable(train_branch)
    optimizer = _oracle_optimizer(model, train_branch, config)
    score_generator = OracleScoreGenerator(motion, score_scales, active_respiratory_levels=3)
    random = torch.Generator(device=device).manual_seed(seed)
    norm_weight = float(config["training"]["loss_weights"]["mbc_normalization"])
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True); losses = []
        offset = (step * observations_per_update) % len(observations)
        for position in range(observations_per_update):
            observation = observations[(offset + position) % len(observations)]
            scores = score_generator(torch.as_tensor([observation.timestamp_s], device=device))
            uv = _uv(observation, min(pixel_samples, observation.image.shape[-1] * observation.image.shape[-2]), random)
            prediction = _oracle_render(model, observation, scores, train_branch, uv, psf_training_mode=psf_training_mode, target_seed=seed + observation.dynamic_frame_id, psf_offset_cache=psf_offset_caches[observation.dynamic_frame_id])
            target = observation.image[0, uv[:, 1].long(), uv[:, 0].long()]
            losses.append((prediction - target).square().mean())
        loss = torch.stack(losses).mean() + norm_weight * model.normalization(train_branch)
        loss.backward(); optimizer.step()
    parameter_groups = [{"lr": float(group["lr"]), "parameter_count": sum(parameter.numel() for parameter in group["params"])} for group in optimizer.param_groups]
    _save_oracle_checkpoint(model, checkpoint_path, checkpoint_metadata)
    optimizer.zero_grad(set_to_none=True)
    loss = losses = prediction = target = scores = uv = None
    del optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    rows, arrays = _oracle_evaluation(model, observations, psf_offset_caches, motion, score_generator, train_branch, psf_training_mode, domain, pixel_samples, evaluation_pixel_chunk)
    return rows, arrays, parameter_groups


def _write_rows(path: Path, rows: dict[str, dict[str, float]], *, include_scale: bool = False) -> None:
    names = sorted({name for values in rows.values() for name in values})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["location", *names]); writer.writeheader()
        writer.writerows({"location": location, **values} for location, values in rows.items())


def _write_aggregate(path: Path, aggregate: dict[str, dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "mean", "median", "p10", "p90"]); writer.writeheader()
        writer.writerows({"metric": name, **values} for name, values in aggregate.items())


def _write_observability_aggregate(path: Path, aggregates: dict[str, dict[str, dict[str, float]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["amplitude_scale", "metric", "mean", "median", "p10", "p90"]); writer.writeheader()
        writer.writerows({"amplitude_scale": scale, "metric": metric, **values} for scale, aggregate in aggregates.items() for metric, values in aggregate.items())


def _direct_recovery_label(metrics: dict[str, float], thresholds: dict[str, float]) -> str:
    ok = metrics["normalized_epe"] <= thresholds["max_normalized_epe"] and metrics["spatial_cosine"] >= thresholds["min_spatial_cosine"] and thresholds["min_rms_ratio"] <= metrics["rms_ratio"] <= thresholds["max_rms_ratio"]
    return "pass" if ok else "fail"


def _recovery_label(aggregate: dict[str, dict[str, float]], branch: str, thresholds: dict[str, float]) -> str:
    if not aggregate:
        return "indeterminate"
    def passed(prefix: str) -> bool:
        value = lambda key: aggregate.get(f"{prefix}_{key}", {}).get("median", float("nan"))
        return thresholds["min_rms_ratio"] <= value("dynamic_rms_ratio") <= thresholds["max_rms_ratio"] and value("dynamic_cosine") >= thresholds["min_spatial_cosine"] and value("normalized_epe") <= thresholds["max_normalized_epe"]
    return "pass" if (passed("resp_dvf") if branch == "resp" else passed("card_dvf") if branch == "card" else passed("resp_dvf") and passed("card_dvf")) else "fail"


def _support_recovery_label(aggregate: dict[str, dict[str, float]], prefix: str, thresholds: dict[str, float], *, dynamic: bool) -> str:
    if not aggregate:
        return "indeterminate"
    def metric(name: str) -> float:
        return aggregate.get(f"{prefix}_{name}", {}).get("median", float("nan"))
    ratio_name, cosine_name = ("dynamic_rms_ratio", "dynamic_cosine") if dynamic else ("rms_ratio", "spatial_cosine")
    passed = thresholds["min_rms_ratio"] <= metric(ratio_name) <= thresholds["max_rms_ratio"] and metric(cosine_name) >= thresholds["min_spatial_cosine"] and metric("normalized_epe") <= thresholds["max_normalized_epe"]
    return "pass" if passed else "fail"


def _support_mechanistic_interpretation(full_grid: str, center_support: str, psf_support: str) -> str:
    if full_grid == "fail" and center_support == "pass":
        return "full-volume recovery fails while acquired center-plane support passes; off-support 3D null-space/extrapolation is implicated"
    if center_support == "fail" and psf_support == "fail":
        return "field recovery fails even on directly queried acquisition support; image-equivalent motion ambiguity/identifiability remains"
    if center_support == "fail" and psf_support == "pass":
        return "center-plane support fails while exact PSF-query support passes"
    if center_support == "pass" and psf_support == "pass":
        return "motion is recovered on acquisition support; remaining full-volume error is predominantly off-support"
    return "support audit is indeterminate"


def _observability_label(aggregate: dict[str, dict[str, float]]) -> str:
    """Avoid turning the deterministic GT rerender's zero MSE into a verdict."""
    return "quantitative_only" if aggregate else "indeterminate"


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    _prepare_output(args.output_dir, args.overwrite)
    with args.source_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with args.experiment_config.open(encoding="utf-8") as handle:
        synthetic = yaml.safe_load(handle)["synthetic"]
    domain = _load_domain(args.canonical_domain)
    motion = GroundTruthMotion.from_domain(domain, respiratory_frequency_hz=float(synthetic["respiratory_frequency_hz"]), cardiac_frequency_hz=float(synthetic["cardiac_frequency_hz"]), respiratory_amplitude_mm=float(synthetic["respiratory_amplitude_mm"]), cardiac_amplitude_mm=float(synthetic["cardiac_amplitude_mm"]))
    normalization_grid_shapes = _normalization_grid_shapes(config, args.smoke)
    cardiac = domain["cardiac_box"]
    score_scales = derive_oracle_score_scales(motion, lower_world_mm=torch.as_tensor(domain["world_min_mm"], dtype=torch.float32, device=device), upper_world_mm=torch.as_tensor(domain["world_max_mm"], dtype=torch.float32, device=device), cardiac_lower_world_mm=torch.as_tensor(cardiac["min_mm"], dtype=torch.float32, device=device), cardiac_upper_world_mm=torch.as_tensor(cardiac["max_mm"], dtype=torch.float32, device=device), respiratory_grid_shape=normalization_grid_shapes[0], cardiac_grid_shape=normalization_grid_shapes[1])
    thresholds = {"max_normalized_epe": float(getattr(args, "max_normalized_epe", .25)), "min_spatial_cosine": float(getattr(args, "min_spatial_cosine", .90)), "min_rms_ratio": float(getattr(args, "min_rms_ratio", .75)), "max_rms_ratio": float(getattr(args, "max_rms_ratio", 1.25))}
    image_shape = tuple(int(value) for value in synthetic["image_shape"])
    if args.smoke:
        image_shape = (8, 8)
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=device) if args.manifest else fallback_geometry(locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, image_shape=image_shape, motion=motion, device=device)
    phantom = AnalyticCanonicalPhantom(domain).to(device)
    psf_samples = 1 if args.smoke else int(synthetic["psf_samples"])
    observations = _build_observations(frames, image_shape=image_shape, phantom=phantom, motion=motion, psf_samples=psf_samples, seed=args.seed)
    obs_rows: dict[str, dict[str, float]] = {}; arrays: dict[str, np.ndarray] = {}; direct: dict[str, dict[str, float]] = {}; oracle_rows: dict[str, dict[str, float]] = {}; optimizer_parameter_groups: list[dict[str, float]] = []
    if args.mode in {"observability", "all"}:
        obs_rows, values = observability_audit(observations, frames, image_shape=image_shape, phantom=phantom, motion=motion, psf_samples=psf_samples, seed=args.seed, amplitude_scales=args.amplitude_scales)
        arrays.update(values)
    if args.mode in {"direct_mbc", "all"}:
        direct = _direct_fit(domain, phantom, motion, config, score_scales, device=device, steps=args.steps, smoke=args.smoke)
    branch_map = {"oracle_resp_only": "resp", "oracle_card_only": "card", "oracle_combined": "combined"}
    if args.mode in branch_map:
        oracle_checkpoint_metadata = {
            "experiment": "exp1b_observability_oracle",
            "mode": args.mode,
            "branch": branch_map[args.mode],
            "psf_training_mode": args.psf_training_mode,
            "source_config": str(args.source_config),
            "experiment_config": str(args.experiment_config),
            "seed": args.seed,
            "steps": args.steps,
            "pixel_samples": args.pixel_samples,
            "observations_per_update": args.observations_per_update,
            "evaluation_pixel_chunk": getattr(args, "evaluation_pixel_chunk", 64),
            "active_respiratory_levels": 3,
            "oracle_score_scales": score_scales.summary(),
            "normalization_grid_shapes": {"respiratory": list(normalization_grid_shapes[0]), "cardiac": list(normalization_grid_shapes[1])},
        }
        oracle_rows, values, optimizer_parameter_groups = _oracle_run(domain, frames, image_shape=image_shape, phantom=phantom, motion=motion, config=config, score_scales=score_scales, branch=branch_map[args.mode], device=device, steps=args.steps, pixel_samples=args.pixel_samples, observations_per_update=args.observations_per_update, psf_training_mode=args.psf_training_mode, evaluation_pixel_chunk=getattr(args, "evaluation_pixel_chunk", 64), checkpoint_path=args.output_dir / "oracle_model_last.pt", checkpoint_metadata=oracle_checkpoint_metadata, seed=args.seed, smoke=args.smoke)
        arrays.update(values)
    obs_aggregate = _aggregate_by_amplitude_scale(obs_rows) if obs_rows else {}
    oracle_aggregate = aggregate_location_metrics(oracle_rows) if oracle_rows else {}
    _write_rows(args.output_dir / "observability_per_location.csv", obs_rows)
    _write_observability_aggregate(args.output_dir / "observability_aggregate.csv", obs_aggregate)
    with (args.output_dir / "direct_mbc_fit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["branch", "learning_rate", "epe_mm", "gt_rms_mm", "normalized_epe", "spatial_cosine", "rms_ratio", "max_error_mm", "mbc_normalization", "mbc_normalization_weight"]); writer.writeheader(); writer.writerows({"branch": key, **value} for key, value in direct.items())
    _write_rows(args.output_dir / "oracle_per_location.csv", oracle_rows)
    _write_aggregate(args.output_dir / "oracle_aggregate.csv", oracle_aggregate)
    np.savez_compressed(args.output_dir / "recovery_arrays.npz", **arrays)
    if obs_rows:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(5, 3)); axis.bar(range(len(obs_rows)), [value["zero_vs_gt_mse_gap"] for value in obs_rows.values()]); axis.set_ylabel("zero-vs-GT MSE gap"); axis.set_xlabel("fixed location / amplitude"); figure.tight_layout(); figure.savefig(args.output_dir / "figures" / "observability_mse_gaps.png", dpi=140); plt.close(figure)
    observability = _observability_label(obs_aggregate)
    direct_label = lambda branch: "indeterminate" if branch not in direct else _direct_recovery_label(direct[branch], thresholds)
    recovery = _recovery_label(oracle_aggregate, branch_map[args.mode], thresholds) if args.mode in branch_map else "indeterminate"
    full_resp_recovery = _recovery_label(oracle_aggregate, "resp", thresholds) if args.mode in {"oracle_resp_only", "oracle_combined"} else "indeterminate"
    center_support_recovery = _support_recovery_label(oracle_aggregate, "resp_center_support", thresholds, dynamic=True) if args.mode in {"oracle_resp_only", "oracle_combined"} else "indeterminate"
    psf_support_recovery = _support_recovery_label(oracle_aggregate, "resp_psf_support", thresholds, dynamic=False) if args.mode in {"oracle_resp_only", "oracle_combined"} else "indeterminate"
    support_interpretation = _support_mechanistic_interpretation(full_resp_recovery, center_support_recovery, psf_support_recovery)
    interpretation = "observability not evaluated"
    if observability == "quantitative_only": interpretation = "observability is reported quantitatively; deterministic GT rerender MSE is not used as a strong/weak verdict"
    elif any(direct_label(branch) == "fail" for branch in direct): interpretation = "production MBC direct representability is the limiting factor"
    elif oracle_rows and recovery == "fail": interpretation = "MBC is tested directly; oracle image-domain recovery points to reconstruction-gradient/optimization"
    elif oracle_rows and recovery == "pass": interpretation = "oracle recovery supports an encoder/state-inference explanation for original Exp1 failure"
    summary = {"experiment": "exp1b_observability_oracle", "mode": args.mode, "observability": observability, "resp_direct_mbc_fit": direct_label("resp"), "card_direct_mbc_fit": direct_label("card"), "oracle_resp_recovery": recovery if args.mode == "oracle_resp_only" else "indeterminate", "oracle_resp_center_support_recovery": center_support_recovery if args.mode == "oracle_resp_only" else "indeterminate", "oracle_resp_psf_support_recovery": psf_support_recovery if args.mode == "oracle_resp_only" else "indeterminate", "oracle_card_recovery": recovery if args.mode == "oracle_card_only" else "indeterminate", "oracle_combined_recovery": recovery if args.mode == "oracle_combined" else "indeterminate", "support_mechanistic_interpretation": support_interpretation, "mechanistic_interpretation": interpretation, "observability_aggregate_by_amplitude_scale": obs_aggregate, "direct_mbc_metrics": direct, "direct_mbc_normalization_weight": float(config["training"]["loss_weights"]["mbc_normalization"]), "oracle_aggregate": oracle_aggregate, "oracle_score_scales": score_scales.summary(), "recovery_thresholds": thresholds, "training": {"steps": args.steps, "pixel_samples": args.pixel_samples, "evaluation_pixel_chunk": getattr(args, "evaluation_pixel_chunk", 64), "observations_per_update": args.observations_per_update, "psf_training_mode": getattr(args, "psf_training_mode", "stochastic"), "optimizer_parameter_groups": optimizer_parameter_groups, "normalization_grid_shapes": {"respiratory": list(normalization_grid_shapes[0]), "cardiac": list(normalization_grid_shapes[1])}, "canonical_trainable": False, "encoder_trainable": False}}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp1-B synthetic observability and timestamp-oracle diagnostic")
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml")
    parser.add_argument("--manifest", type=Path); parser.add_argument("--qc-table", type=Path); parser.add_argument("--canonical-domain", type=Path)
    parser.add_argument("--mode", choices=("observability", "direct_mbc", "oracle_resp_only", "oracle_card_only", "oracle_combined", "all"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--psf-training-mode", choices=("matched_crn", "stochastic"), default="stochastic")
    parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50); parser.add_argument("--steps", type=int, default=400); parser.add_argument("--pixel-samples", type=int, default=256); parser.add_argument("--evaluation-pixel-chunk", type=int, default=64); parser.add_argument("--observations-per-update", type=int, default=32); parser.add_argument("--amplitude-scales", type=lambda value: [float(item) for item in value.split(",")], default=[1.0]); parser.add_argument("--max-normalized-epe", type=float, default=.25); parser.add_argument("--min-spatial-cosine", type=float, default=.90); parser.add_argument("--min-rms-ratio", type=float, default=.75); parser.add_argument("--max-rms-ratio", type=float, default=1.25); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (args.manifest is None) != (args.qc_table is None): parser.error("--manifest and --qc-table must be provided together")
    if args.smoke:
        args.steps = min(args.steps, 2); args.pixel_samples = min(args.pixel_samples, 8); args.observations_per_update = min(args.observations_per_update, 2); args.locations_per_view = min(args.locations_per_view, 1); args.frames_per_location = min(args.frames_per_location, 4)
    summary = run(args)
    print(f"Exp1-B {summary['mode']}: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
