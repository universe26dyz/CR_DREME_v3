#!/usr/bin/env python3
"""Exp3-A: learned-score × learned-MBC exact-full-batch recovery probe."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import torch
import yaml

from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField
from cardioresp4d.training.model import SourceFirstDynamicModel
from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, _load_domain, fallback_geometry, geometry_from_manifest
from experiments.c4_diagnostics.exp2_checkpoint_common import parameter_snapshot
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import (
    EPSILON,
    _ENCODER_PREFIXES,
    _build_intervention_model,
    _full_image_uv,
    _git_sha,
    _group_by_location,
    _gt_respiratory_scores,
    _respiratory_only_motion,
    _score_values,
    _set_seed,
    _validate_observations,
)
from experiments.c4_diagnostics.exp2c4h_oracle_field_fullbatch_reconstruction import _fixed_pixel_grid, _observation_audit, _restore_rng_state
from experiments.c4_diagnostics.synthetic.metrics import aggregate_location_metrics
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom


_LEVEL0_PREFIX = "respiratory_mbc.levels.0."
_ALLOWED_PREFIXES = (*_ENCODER_PREFIXES, _LEVEL0_PREFIX)


def _trainable_parameter_names(model: SourceFirstDynamicModel) -> set[str]:
    return {name for name, _ in model.named_parameters() if name.startswith(_ALLOWED_PREFIXES)}


def _configure_joint_training(model: SourceFirstDynamicModel, *, film_lr: float, respiratory_mbc_lr: float) -> torch.optim.Adam:
    """Freeze production model except respiratory encoder and active MBC level 0."""
    if film_lr <= 0 or respiratory_mbc_lr <= 0:
        raise ValueError("Exp3-A learning rates must be positive")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    encoder = [(name, parameter) for name, parameter in model.named_parameters() if name.startswith(_ENCODER_PREFIXES)]
    level0 = [(name, parameter) for name, parameter in model.named_parameters() if name.startswith(_LEVEL0_PREFIX)]
    if not encoder or not level0:
        raise RuntimeError("Exp3-A did not find respiratory encoder or MBC level0 parameters")
    for _, parameter in encoder + level0:
        parameter.requires_grad_(True)
    model.respiratory_mbc.set_active_levels(1)
    optimizer = torch.optim.Adam((
        {"params": [parameter for _, parameter in encoder], "lr": film_lr},
        {"params": [parameter for _, parameter in level0], "lr": respiratory_mbc_lr},
    ))
    owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected = {id(parameter) for _, parameter in encoder + level0}
    if owned != expected or any(parameter.requires_grad != (id(parameter) in expected) for parameter in model.parameters()):
        raise RuntimeError("Exp3-A optimizer ownership is not exact respiratory encoder + MBC level0")
    return optimizer


def _prediction_loss_terms(model: SourceFirstDynamicModel, observation, pixel_uv: torch.Tensor) -> dict[str, torch.Tensor]:
    """Use the unmodified production stage2a renderer and pullback path."""
    render = model.predict(observation, pixel_uv, stage="stage2a")
    target = observation.image[0, pixel_uv[:, 1].long(), pixel_uv[:, 0].long()]
    return {"data_mse": (render["predicted_intensity"] - target).square().mean(), "resp_scores": render["scores"]["resp_scores"]}


def _accumulate_full_batch(model: SourceFirstDynamicModel, observations: list, *, pixel_mode: str, pixel_samples: int, accumulation_chunk_size: int) -> dict[str, Any]:
    if pixel_mode != "fixed_grid" or not observations or accumulation_chunk_size <= 0:
        raise ValueError("Exp3-A requires non-empty deterministic fixed-grid full batches")
    total = observations[0].image.new_zeros(())
    observation_ids: list[int] = []
    scale = 1.0 / len(observations)
    for start in range(0, len(observations), accumulation_chunk_size):
        chunk: list[torch.Tensor] = []
        for observation in observations[start : start + accumulation_chunk_size]:
            loss = _prediction_loss_terms(model, observation, _fixed_pixel_grid(observation, count=pixel_samples))["data_mse"]
            chunk.append(loss); total = total + loss.detach(); observation_ids.append(int(observation.dynamic_frame_id))
        (torch.stack(chunk).sum() * scale).backward()
    return {"full_batch_reconstruction_mse": total / len(observations), "observation_ids": observation_ids}


@torch.inference_mode()
def _full_batch_reconstruction_mse(model: SourceFirstDynamicModel, observations: list, *, pixel_samples: int) -> float:
    return float(torch.stack([_prediction_loss_terms(model, observation, _fixed_pixel_grid(observation, count=pixel_samples))["data_mse"] for observation in observations]).mean())


def _dynamic_dvf_values(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_dynamic, target_dynamic = prediction - prediction.mean(0, keepdim=True), target - target.mean(0, keepdim=True)
    pred_rms, target_rms = pred_dynamic.square().mean().sqrt(), target_dynamic.square().mean().sqrt()
    dynamic_epe = (pred_dynamic - target_dynamic).square().sum(-1).sqrt().mean()
    return {
        "static_rms_mm": float(prediction.mean(0).square().mean().sqrt()),
        "dynamic_rms_mm": float(pred_rms), "gt_dynamic_rms_mm": float(target_rms),
        "dynamic_rms_ratio": float(pred_rms / target_rms.clamp_min(EPSILON)),
        "dynamic_cosine": float((pred_dynamic * target_dynamic).sum() / (pred_dynamic.norm() * target_dynamic.norm()).clamp_min(EPSILON)),
        "dynamic_normalized_epe": float(dynamic_epe / target_rms.clamp_min(EPSILON)),
    }


@torch.inference_mode()
def _metrics(model: SourceFirstDynamicModel, groups: dict[str, list], motion: GroundTruthMotion, *, grid_shape: tuple[int, int, int]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    grid, _ = SourceFirstDynamicModel._physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, grid_shape)
    rows: dict[str, dict[str, float]] = {}; arrays: dict[str, np.ndarray] = {}
    all_scores: list[tuple[torch.Tensor, torch.Tensor]] = []
    by_view_scores: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for location, sequence in groups.items():
        full_pred: list[torch.Tensor] = []; full_gt: list[torch.Tensor] = []; center_pred: list[torch.Tensor] = []; center_gt: list[torch.Tensor] = []
        scores: list[torch.Tensor] = []; gt_scores: list[torch.Tensor] = []; image_losses: list[torch.Tensor] = []; timestamps: list[float] = []
        for observation in sequence:
            score = model.encode_motion_scores(observation)["resp_scores"][:, :1, :]
            timestamp = grid.new_tensor([observation.timestamp_s]); gt_score = _gt_respiratory_scores(motion, timestamp)
            field = ScoreWeightedMBCField(model.respiratory_mbc, score)
            full_pred.append(field(grid)[0]); full_gt.append(motion.respiratory_dvf_mm(grid, timestamp)[0])
            uv = _full_image_uv(observation); centers = SourceFirstDynamicModel._pixel_world(observation, uv)
            center_pred.append(field(centers[None])[0]); center_gt.append(motion.respiratory_dvf_mm(centers[None], timestamp)[0])
            image_losses.append(_prediction_loss_terms(model, observation, uv)["data_mse"])
            scores.append(score[0]); gt_scores.append(gt_score[0]); timestamps.append(observation.timestamp_s)
        predicted, target = torch.stack(scores), torch.stack(gt_scores)
        row = {"image_data_mse": float(torch.stack(image_losses).mean()), **_score_values(predicted, target), **{f"full_fov_{name}": value for name, value in _dynamic_dvf_values(torch.stack(full_pred), torch.stack(full_gt)).items()}, **{f"pixel_center_{name}": value for name, value in _dynamic_dvf_values(torch.stack(center_pred), torch.stack(center_gt)).items()}}
        rows[location] = row; stem = location.replace("/", "_")
        arrays[f"{stem}_timestamps_s"] = np.asarray(timestamps); arrays[f"{stem}_predicted_scores"] = predicted.cpu().numpy(); arrays[f"{stem}_gt_scores"] = target.cpu().numpy()
        all_scores.append((predicted, target)); by_view_scores.setdefault(sequence[0].view, []).append((predicted, target))
    pooled = lambda pairs: _score_values(torch.cat([item[0] for item in pairs]), torch.cat([item[1] for item in pairs]))
    by_view = {view: aggregate_location_metrics({name: rows[name] for name, sequence in groups.items() if sequence[0].view == view}) for view in sorted(by_view_scores)}
    return {"per_location": rows, "aggregate": aggregate_location_metrics(rows), "by_view": by_view, "overall_score_metrics": pooled(all_scores), "by_view_score_metrics": {view: pooled(items) for view, items in by_view_scores.items()}}, arrays


def _aggregate_dvf(metrics: dict[str, Any]) -> dict[str, float]:
    aggregate = metrics["aggregate"]
    return {name: float(aggregate[name]["mean"]) for name in ("full_fov_dynamic_cosine", "full_fov_dynamic_rms_ratio", "full_fov_dynamic_normalized_epe")}


def _classify_dvf(metrics: dict[str, float]) -> dict[str, bool | str]:
    """Classify only recovered DVF: score scale/sign are bilinear gauge-dependent."""
    escape = metrics["full_fov_dynamic_cosine"] >= .90 and .5 <= metrics["full_fov_dynamic_rms_ratio"] <= 1.5 and metrics["full_fov_dynamic_normalized_epe"] <= .75
    strict = metrics["full_fov_dynamic_cosine"] >= .95 and .8 <= metrics["full_fov_dynamic_rms_ratio"] <= 1.2 and metrics["full_fov_dynamic_normalized_epe"] <= .50
    return {"classification": "joint_fullbatch_coldstart_escape" if escape else "joint_fullbatch_recovery_failure", "strict_recovery": strict}


def _empty_thresholds() -> dict[str, int | None]:
    return {name: None for name in ("first_full_fov_cosine_ge_0p80_step", "first_full_fov_cosine_ge_0p90_step", "first_full_fov_cosine_ge_0p95_step", "first_full_fov_rms_ratio_ge_0p1_step", "first_full_fov_rms_ratio_in_0p5_1p5_step", "first_full_fov_rms_ratio_in_0p8_1p2_step", "first_full_fov_nepe_le_1p0_step", "first_full_fov_nepe_le_0p75_step", "first_full_fov_nepe_le_0p5_step", "first_coldstart_escape_step", "first_strict_recovery_step")}


def _update_thresholds(thresholds: dict[str, int | None] | None, metrics: dict[str, float], *, step: int) -> dict[str, int | None]:
    result, classification = _empty_thresholds() if thresholds is None else dict(thresholds), _classify_dvf(metrics)
    c, r, n = metrics["full_fov_dynamic_cosine"], metrics["full_fov_dynamic_rms_ratio"], metrics["full_fov_dynamic_normalized_epe"]
    conditions = {"first_full_fov_cosine_ge_0p80_step": c >= .80, "first_full_fov_cosine_ge_0p90_step": c >= .90, "first_full_fov_cosine_ge_0p95_step": c >= .95, "first_full_fov_rms_ratio_ge_0p1_step": r >= .1, "first_full_fov_rms_ratio_in_0p5_1p5_step": .5 <= r <= 1.5, "first_full_fov_rms_ratio_in_0p8_1p2_step": .8 <= r <= 1.2, "first_full_fov_nepe_le_1p0_step": n <= 1., "first_full_fov_nepe_le_0p75_step": n <= .75, "first_full_fov_nepe_le_0p5_step": n <= .5, "first_coldstart_escape_step": classification["classification"] == "joint_fullbatch_coldstart_escape", "first_strict_recovery_step": bool(classification["strict_recovery"])}
    for name, reached in conditions.items():
        if reached and result[name] is None:
            result[name] = step
    return result


def _parameter_norm(model: SourceFirstDynamicModel, prefix: str) -> float:
    values = [parameter.detach().reshape(-1) for name, parameter in model.named_parameters() if name.startswith(prefix)]
    return float(torch.cat(values).norm())


def _gradient_norm(model: SourceFirstDynamicModel, prefix: str) -> float:
    values = [parameter.grad.detach().reshape(-1) for name, parameter in model.named_parameters() if name.startswith(prefix) and parameter.grad is not None]
    return float(torch.cat(values).norm()) if values else 0.0


def _curve_row(step: int, metrics: dict[str, Any], *, reconstruction_mse: float, model: SourceFirstDynamicModel) -> dict[str, float | int]:
    dvf, score = _aggregate_dvf(metrics), metrics["overall_score_metrics"]
    return {"step": step, "full_batch_reconstruction_mse": reconstruction_mse, **dvf, "predicted_score_temporal_rms": score["predicted_score_temporal_rms"], "encoder_parameter_norm": _parameter_norm(model, "film_encoder."), "respiratory_mbc_level0_parameter_norm": _parameter_norm(model, _LEVEL0_PREFIX), "encoder_grad_norm": _gradient_norm(model, "film_encoder."), "respiratory_mbc_level0_grad_norm": _gradient_norm(model, _LEVEL0_PREFIX), "predicted_respiratory_dvf_rms_mm": float(metrics["aggregate"]["full_fov_dynamic_rms_mm"]["mean"])}


def _write_curve(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError("Exp3-A output directory is non-empty; pass --overwrite to reuse it")
        for item in path.iterdir():
            if item.is_dir():
                import shutil; shutil.rmtree(item)
            else:
                item.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _checkpoint_state(*, model, optimizer, step: int, protocol: dict[str, Any], pixel_protocol: dict[str, Any], observation_audit: dict[str, Any], thresholds, curve, initial, initial_arrays):
    rng_state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "git_sha": _git_sha(), "protocol": protocol, "pixel_protocol": pixel_protocol, "observation_audit": observation_audit, "thresholds": thresholds, "curve": curve, "initial_metrics": initial, "initial_arrays": {name: torch.from_numpy(value) for name, value in initial_arrays.items()}, "rng_state": rng_state}


def _save_checkpoint(path: Path, **state: Any) -> None:
    torch.save(_checkpoint_state(**state), path)


def _changed_parameters(model: SourceFirstDynamicModel, before: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    after = dict(model.named_parameters())
    if set(after) != set(before):
        raise RuntimeError("Exp3-A model parameter set changed")
    changed = [name for name, value in after.items() if not torch.equal(before[name], value.detach().cpu())]
    if not changed or set(changed) - _trainable_parameter_names(model):
        raise RuntimeError("Exp3-A modified frozen parameters or did not update trainable factors")
    return changed, sorted(set(before) - set(changed))


def _exposure_accounting(*, steps: int, n_observations: int, pixels_per_observation: int) -> dict[str, int]:
    return {"optimizer_steps": steps, "observations_per_optimizer_step": n_observations, "pixels_per_observation": pixels_per_observation, "observation_exposures": steps * n_observations, "pixel_observation_exposures": steps * n_observations * pixels_per_observation, "per_frame_exposures": steps}


def run(args: argparse.Namespace) -> dict[str, Any]:
    device, resume_from = torch.device(args.device), getattr(args, "resume_from", None)
    checkpoint_every = int(getattr(args, "checkpoint_every", 1000))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if min(args.steps, args.eval_every, args.pixel_samples, args.accumulation_chunk_size, checkpoint_every) <= 0:
        raise ValueError("Exp3-A step, evaluation, pixel, accumulation, and checkpoint arguments must be positive")
    if not args.smoke and (args.manifest is None or args.qc_table is None or args.canonical_domain is None):
        raise ValueError("formal Exp3-A requires --manifest, --qc-table, and --canonical-domain")
    if not args.smoke and (args.locations_per_view, args.frames_per_location, args.pixel_mode, args.pixel_samples) != (3, 50, "fixed_grid", 256):
        raise ValueError("formal Exp3-A requires 9 locations / 450 frames and fixed_grid / 256 pixels")
    if resume_from is None:
        _prepare_output(args.output_dir, args.overwrite)
    elif args.overwrite or not args.output_dir.is_dir():
        raise ValueError("Exp3-A resume requires an existing output directory without --overwrite")
    _set_seed(args.seed)
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); synthetic = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))["synthetic"]; domain = _load_domain(args.canonical_domain)
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(domain, respiratory_frequency_hz=float(synthetic["respiratory_frequency_hz"]), cardiac_frequency_hz=float(synthetic["cardiac_frequency_hz"]), respiratory_amplitude_mm=float(synthetic["respiratory_amplitude_mm"]), cardiac_amplitude_mm=float(synthetic["cardiac_amplitude_mm"])))
    if motion.cardiac_amplitude_mm != 0.0:
        raise RuntimeError("Exp3-A target must have exactly zero cardiac amplitude")
    image_shape, psf_samples = ((8, 8), 1) if args.smoke else (tuple(int(value) for value in synthetic["image_shape"]), int(synthetic["psf_samples"]))
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=device) if args.manifest is not None else fallback_geometry(locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, image_shape=image_shape, motion=motion, device=device)
    observations = _build_observations(frames, image_shape=image_shape, phantom=AnalyticCanonicalPhantom(domain).to(device), motion=motion, psf_samples=psf_samples, seed=args.seed)
    groups = _validate_observations(observations, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, smoke=args.smoke); observation_audit = _observation_audit(observations)
    if not observation_audit["timestamps_strictly_increasing"]:
        raise RuntimeError("Exp3-A observations must have increasing timestamps per location")
    pixel_uv = _fixed_pixel_grid(observations[0], count=args.pixel_samples)
    pixel_protocol = {"pixel_mode": args.pixel_mode, "requested_pixel_samples": args.pixel_samples, "actual_pixel_count": int(pixel_uv.shape[0]), "fixed_uv": pixel_uv.detach().cpu().long().tolist()}
    model = _build_intervention_model(config, domain, n_dynamic_frames=len(observations), seed=args.seed, device=device)
    if model.film_implementation != "full_geometry_single_film":
        raise RuntimeError("Exp3-A requires production GeometryFiLMMotionEncoder")
    film_lr, mbc_lr = float(config["training"]["optimizer"]["stage2"]["film_lr"]), float(config["training"]["optimizer"]["stage2"]["respiratory_mbc_lr"])
    optimizer = _configure_joint_training(model, film_lr=film_lr, respiratory_mbc_lr=mbc_lr)
    protocol = {"exact_full_batch": True, "optimizer_step_after_all_observations": True, "uses_sampler": False, "objective": "reconstruction_image_data_mse_only", "pixel_mode": "fixed_grid", "active_respiratory_levels": 1, "trainable_modules": [*_ENCODER_PREFIXES, _LEVEL0_PREFIX]}
    checkpoint = None
    if resume_from is not None:
        checkpoint = torch.load(Path(resume_from), map_location=device, weights_only=True)
        if checkpoint.get("observation_audit") != observation_audit or checkpoint.get("pixel_protocol") != pixel_protocol or checkpoint.get("protocol") != protocol:
            raise ValueError("Exp3-A resume checkpoint has an incompatible scientific or pixel protocol")
        if not 0 <= int(checkpoint.get("step", -1)) < args.steps:
            raise ValueError("Exp3-A resume checkpoint step must be smaller than --steps")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"]); _restore_rng_state(checkpoint["rng_state"])
    before = parameter_snapshot(model); renderer_calls = [0]; training_renderer_calls = [0]; mbc_calls = [0]; training_mbc_calls = [0]; cardiac_calls = [0]; uncertainty_calls = [0]
    hooks = [model.psf.register_forward_hook(lambda *_: renderer_calls.__setitem__(0, renderer_calls[0] + 1)), model.respiratory_mbc.register_forward_hook(lambda *_: mbc_calls.__setitem__(0, mbc_calls[0] + 1)), model.cardiac_mbc.register_forward_hook(lambda *_: cardiac_calls.__setitem__(0, cardiac_calls[0] + 1)), model.uncertainty.register_forward_hook(lambda *_: uncertainty_calls.__setitem__(0, uncertainty_calls[0] + 1))]
    try:
        if checkpoint is None:
            initial, initial_arrays = _metrics(model, groups, motion, grid_shape=tuple(args.evaluation_grid_shape)); thresholds = _update_thresholds(None, _aggregate_dvf(initial), step=0)
            curve = [_curve_row(0, initial, reconstruction_mse=_full_batch_reconstruction_mse(model, observations, pixel_samples=args.pixel_samples), model=model)]
            _write_curve(args.output_dir / "training_curve.csv", curve); (args.output_dir / "metrics_initial.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8"); start_step = 0
        else:
            initial = checkpoint["initial_metrics"]; initial_arrays = {name: value.cpu().numpy() for name, value in checkpoint["initial_arrays"].items()}; thresholds = checkpoint["thresholds"]; curve = checkpoint["curve"]; start_step = int(checkpoint["step"])
        expected_ids = [observation.dynamic_frame_id for observation in observations]
        for step in range(start_step + 1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True); renderer_before, mbc_before = renderer_calls[0], mbc_calls[0]
            batch = _accumulate_full_batch(model, observations, pixel_mode=args.pixel_mode, pixel_samples=args.pixel_samples, accumulation_chunk_size=args.accumulation_chunk_size)
            training_renderer_calls[0] += renderer_calls[0] - renderer_before; training_mbc_calls[0] += mbc_calls[0] - mbc_before
            if batch["observation_ids"] != expected_ids:
                raise RuntimeError("Exp3-A full batch did not use every observation exactly once in stable order")
            if step == start_step + 1:
                for prefix in (*_ENCODER_PREFIXES, _LEVEL0_PREFIX):
                    if not any(name.startswith(prefix) and parameter.grad is not None and torch.isfinite(parameter.grad).all() and float(parameter.grad.norm()) > 0 for name, parameter in model.named_parameters()):
                        raise RuntimeError(f"Exp3-A reconstruction loss did not reach {prefix}")
                if any(parameter.grad is not None for name, parameter in model.named_parameters() if not name.startswith(_ALLOWED_PREFIXES)):
                    raise RuntimeError("Exp3-A backward reached frozen parameters")
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                evaluated, _ = _metrics(model, groups, motion, grid_shape=tuple(args.evaluation_grid_shape)); thresholds = _update_thresholds(thresholds, _aggregate_dvf(evaluated), step=step)
                curve.append(_curve_row(step, evaluated, reconstruction_mse=float(batch["full_batch_reconstruction_mse"]), model=model)); _write_curve(args.output_dir / "training_curve.csv", curve)
            if step % checkpoint_every == 0:
                state = {"model": model, "optimizer": optimizer, "step": step, "protocol": protocol, "pixel_protocol": pixel_protocol, "observation_audit": observation_audit, "thresholds": thresholds, "curve": curve, "initial": initial, "initial_arrays": initial_arrays}
                _save_checkpoint(args.output_dir / f"checkpoint_{step}.pt", **state); _save_checkpoint(args.output_dir / "checkpoint_last.pt", **state)
        final, final_arrays = _metrics(model, groups, motion, grid_shape=tuple(args.evaluation_grid_shape))
    finally:
        for hook in hooks:
            hook.remove()
    if renderer_calls[0] <= 0 or training_renderer_calls[0] <= 0 or mbc_calls[0] <= 0 or training_mbc_calls[0] <= 0 or cardiac_calls[0] != 0 or uncertainty_calls[0] != 0:
        raise RuntimeError("Exp3-A production-path runtime audit failed")
    changed, unchanged = _changed_parameters(model, before)
    frozen_audit = {"film_encoder_card_unchanged": not any(name.startswith("film_encoder.card.") for name in changed), "canonical_unchanged": not any(name.startswith("canonical.") for name in changed), "respiratory_mbc_higher_levels_unchanged": not any(name.startswith("respiratory_mbc.levels.1.") or name.startswith("respiratory_mbc.levels.2.") for name in changed), "cardiac_mbc_unchanged": not any(name.startswith("cardiac_mbc.") for name in changed), "uncertainty_unchanged": not any(name.startswith("uncertainty.") for name in changed)}
    if not all(frozen_audit.values()):
        raise RuntimeError("Exp3-A frozen module audit failed")
    final_dvf, classification = _aggregate_dvf(final), _classify_dvf(_aggregate_dvf(final))
    best = min(curve, key=lambda row: row["full_batch_reconstruction_mse"])
    _write_curve(args.output_dir / "training_curve.csv", curve); (args.output_dir / "metrics_initial.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8"); (args.output_dir / "metrics_final.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output_dir / "recovery_arrays.npz", **initial_arrays, **{f"final_{name}": value for name, value in final_arrays.items()})
    state = {"model": model, "optimizer": optimizer, "step": args.steps, "protocol": protocol, "pixel_protocol": pixel_protocol, "observation_audit": observation_audit, "thresholds": thresholds, "curve": curve, "initial": initial, "initial_arrays": initial_arrays}; _save_checkpoint(args.output_dir / "checkpoint_last.pt", **state)
    summary = {"experiment": "Exp3-A / Recovery-1 joint full-batch reconstruction", "git_sha": _git_sha(), "seed": args.seed, "source_config": str(args.source_config), "experiment_config": str(args.experiment_config), "manifest": None if args.manifest is None else str(args.manifest), "qc_table": None if args.qc_table is None else str(args.qc_table), "canonical_domain": None if args.canonical_domain is None else str(args.canonical_domain), "scientific_question": "Can learned respiratory score × learned MBC recover respiratory DVF under long deterministic reconstruction-only optimization?", "interpretation_contract": "Raw score amplitude/MSE is gauge-dependent in learned-score × learned-MBC joint training and is diagnostic only; final respiratory DVF metrics are the primary criterion.", "observation_audit": observation_audit, "pixel_protocol": pixel_protocol, "full_batch_protocol": protocol, "model": "GeometryFiLMMotionEncoder + RespiratorySINRMBCAdapter + ScoreWeightedMBCField", "film_implementation": model.film_implementation, "active_respiratory_levels": 1, "trainable_modules": [*_ENCODER_PREFIXES, _LEVEL0_PREFIX], "trainable_parameter_names": sorted(_trainable_parameter_names(model)), "frozen_module_audit": frozen_audit, "parameters_changed": changed, "parameters_unchanged": unchanged, "film_lr": film_lr, "respiratory_mbc_lr": mbc_lr, "steps": args.steps, "eval_every": args.eval_every, "checkpoint_every": checkpoint_every, "accumulation_chunk_size": args.accumulation_chunk_size, "resumed_from": None if resume_from is None else str(resume_from), "renderer_calls": renderer_calls[0], "training_renderer_calls": training_renderer_calls[0], "learned_respiratory_mbc_calls": mbc_calls[0], "training_learned_respiratory_mbc_calls": training_mbc_calls[0], "cardiac_mbc_calls": cardiac_calls[0], "uncertainty_calls": uncertainty_calls[0], "exposure_accounting": _exposure_accounting(steps=args.steps, n_observations=len(observations), pixels_per_observation=pixel_protocol["actual_pixel_count"]), "threshold_steps": thresholds, "classification": classification["classification"], "strict_recovery": classification["strict_recovery"], "initial_metrics": initial, "final_metrics": final, "best_metrics": {"initial_reconstruction_mse": curve[0]["full_batch_reconstruction_mse"], "best_reconstruction_mse": best["full_batch_reconstruction_mse"], "best_reconstruction_step": best["step"], "final_reconstruction_mse": curve[-1]["full_batch_reconstruction_mse"], "final_dvf": final_dvf}}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3-A learned-score learned-MBC full-batch recovery")
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs/source_first_change4_paperaligned.yaml"); parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml"); parser.add_argument("--manifest", type=Path); parser.add_argument("--qc-table", type=Path); parser.add_argument("--canonical-domain", type=Path); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50); parser.add_argument("--steps", type=int, default=10000); parser.add_argument("--eval-every", type=int, default=250); parser.add_argument("--pixel-mode", choices=("fixed_grid",), default="fixed_grid"); parser.add_argument("--pixel-samples", type=int, default=256); parser.add_argument("--accumulation-chunk-size", type=int, default=5); parser.add_argument("--checkpoint-every", type=int, default=1000); parser.add_argument("--resume-from", type=Path); parser.add_argument("--evaluation-grid-shape", type=lambda value: tuple(int(item) for item in value.split(",")), default=(8, 8, 8)); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); summary = run(args); print(f"Exp3-A: {args.output_dir / 'summary.json'} ({summary['observation_audit']['n_locations']} locations)")


if __name__ == "__main__":
    main()
