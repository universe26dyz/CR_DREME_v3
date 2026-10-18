#!/usr/bin/env python3
"""Exp2-C4a: fixed GT respiratory score, train respiratory MBC level 0 only."""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import torch
import yaml
from torch import nn

from cardioresp4d.losses.motion_loss import dreme_mbc_normalization
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField, SequentialPullbackMotion
from cardioresp4d.training.build_model import build_source_first_model
from cardioresp4d.training.model import SourceFirstDynamicModel
from cardioresp4d.training.sampler import ViewLocationBalancedSampler
from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, _load_domain, fallback_geometry, geometry_from_manifest
from experiments.c4_diagnostics.exp2_checkpoint_common import parameter_snapshot, reconstruct_motion_only_model
from experiments.c4_diagnostics.exp2a_encoder_temporal_identifiability import validate_source_timestamps
from experiments.c4_diagnostics.synthetic.metrics import aggregate_location_metrics, dvf_metrics, vector_field_agreement_metrics
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from experiments.c4_diagnostics.synthetic.render import PhantomCanonicalAdapter


EPSILON = 1e-12


class _ZeroDisplacement(nn.Module):
    def forward(self, points_mm: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(points_mm)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _respiratory_only_motion(motion: GroundTruthMotion) -> GroundTruthMotion:
    return replace(motion, cardiac_amplitude_mm=0.0)


def _gt_respiratory_scores(motion: GroundTruthMotion, timestamps_s: torch.Tensor) -> torch.Tensor:
    """One-level [B, 1, xyz] score carrying the physical respiratory amplitude."""
    timestamps = torch.as_tensor(timestamps_s).reshape(-1)
    coefficient = motion.respiratory_amplitude_mm * motion.respiratory_score(timestamps)
    scores = coefficient[:, None, None].expand(-1, 1, 3)
    if scores.shape != (timestamps.numel(), 1, 3) or not torch.isfinite(scores).all() or scores.requires_grad:
        raise ValueError("C4a GT respiratory scores must be finite fixed [B,1,3]")
    return scores


def _checkpoint_state(checkpoint: Path | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        return checkpoint
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("C4a checkpoint initialization requires an Exp1 synthetic_recovery_last.pt payload")
    return payload["model"]


def _build_intervention_model(config: dict[str, Any], domain: dict[str, object], *, n_dynamic_frames: int, init_mode: str, checkpoint: Path | dict[str, torch.Tensor] | None, seed: int, device: torch.device | None = None) -> SourceFirstDynamicModel:
    """Build the baseline model, replacing canonical before any strict checkpoint load."""
    device = torch.device("cpu") if device is None else device
    _set_seed(seed)
    if init_mode == "fresh":
        if checkpoint is not None:
            raise ValueError("fresh C4a initialization must not receive a checkpoint")
        model = build_source_first_model(config, domain, n_dynamic_frames=n_dynamic_frames, device=device)
        model.canonical = PhantomCanonicalAdapter(AnalyticCanonicalPhantom(domain)).to(device)
    elif init_mode == "checkpoint":
        if checkpoint is None:
            raise ValueError("checkpoint C4a initialization requires --checkpoint")
        model = reconstruct_motion_only_model(config, domain, _checkpoint_state(checkpoint), n_dynamic_frames=n_dynamic_frames, device=device)
    else:
        raise ValueError("init_mode must be fresh or checkpoint")
    return model.to(device).eval()


def _level0_parameter_names(model: SourceFirstDynamicModel) -> set[str]:
    return {f"respiratory_mbc.levels.0.{name}" for name, _ in model.respiratory_mbc.levels[0].named_parameters()}


def _configure_level0_only(model: SourceFirstDynamicModel, *, learning_rate: float) -> torch.optim.Adam:
    if learning_rate <= 0:
        raise ValueError("respiratory MBC learning rate must be positive")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    level0 = list(model.respiratory_mbc.levels[0].parameters())
    for parameter in level0:
        parameter.requires_grad_(True)
    model.respiratory_mbc.set_active_levels(1)
    optimizer = torch.optim.Adam(level0, lr=learning_rate)
    owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if owned != {id(parameter) for parameter in level0}:
        raise RuntimeError("C4a optimizer contains parameters outside respiratory_mbc.level0")
    if any(parameter.requires_grad != (id(parameter) in owned) for parameter in model.parameters()):
        raise RuntimeError("C4a trainable parameter ownership is not level0-only")
    return optimizer


def _full_image_uv(observation) -> torch.Tensor:
    height, width = observation.image.shape[-2:]
    return torch.stack(torch.meshgrid(torch.arange(width, device=observation.image.device), torch.arange(height, device=observation.image.device), indexing="xy"), dim=-1).reshape(-1, 2).to(observation.image.dtype)


def _sample_pixels(observation, count: int) -> torch.Tensor:
    height, width = observation.image.shape[-2:]
    if count <= 0:
        raise ValueError("pixel sample count must be positive")
    linear = torch.randint(height * width, (min(count, height * width),), device=observation.image.device)
    return torch.stack((linear.remainder(width), torch.div(linear, width, rounding_mode="floor")), dim=-1).to(observation.image.dtype)


def _render_with_gt_score(model: SourceFirstDynamicModel, observation, pixel_uv: torch.Tensor, motion: GroundTruthMotion) -> torch.Tensor:
    """Production geometry/PSF/pullback render with encoder and cardiac branch absent."""
    points = SourceFirstDynamicModel._pixel_world(observation, pixel_uv.to(observation.image))
    spacing = observation.pixel_spacing_mm.to(points)
    resolution = torch.stack((spacing[1], spacing[0], points.new_tensor(observation.slice_thickness_mm))).expand(points.shape[0], 3)
    score = _gt_respiratory_scores(motion, points.new_tensor([observation.timestamp_s]))
    respiratory = ScoreWeightedMBCField(model.respiratory_mbc, score)
    pullback = SequentialPullbackMotion(respiratory, _ZeroDisplacement())
    axes = {
        "row_direction": observation.row_direction.to(points).expand(points.shape[0], -1),
        "column_direction": observation.column_direction.to(points).expand(points.shape[0], -1),
        "normal": observation.normal.to(points).expand(points.shape[0], -1),
    }
    return model.psf(model.canonical, points, resolution, **axes, motion=pullback)["predicted_intensity"]


def _normalization_loss(model: SourceFirstDynamicModel) -> torch.Tensor:
    grid, _ = SourceFirstDynamicModel._physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, model.respiratory_smoothness_shape)
    return dreme_mbc_normalization(model.respiratory_mbc.raw_active(grid, active_levels=1))


def _intervention_loss_terms(model: SourceFirstDynamicModel, observation, motion: GroundTruthMotion, *, pixel_uv: torch.Tensor, objective: str, normalization_weight: float) -> dict[str, torch.Tensor]:
    if objective not in {"data_only", "formal_mbc_norm"}:
        raise ValueError("objective must be data_only or formal_mbc_norm")
    if int(model.respiratory_mbc.active_level_count) != 1:
        raise RuntimeError("C4a requires exactly one active respiratory MBC level")
    target = observation.image[0, pixel_uv[:, 1].long(), pixel_uv[:, 0].long()]
    prediction = _render_with_gt_score(model, observation, pixel_uv, motion)
    data_mse = (prediction - target).square().mean()
    normalization = _normalization_loss(model)
    total = data_mse if objective == "data_only" else data_mse + normalization_weight * normalization
    return {"data_mse": data_mse, "mbc_normalization": normalization, "total_loss": total}


def _group_by_location(observations: list) -> dict[str, list]:
    groups: dict[str, list] = {}
    for observation in observations:
        groups.setdefault(f"{observation.view}/{observation.slice_id}", []).append(observation)
    result = {key: sorted(values, key=lambda item: item.timestamp_s) for key, values in groups.items()}
    if any(any(second.timestamp_s <= first.timestamp_s for first, second in zip(sequence, sequence[1:])) for sequence in result.values()):
        raise ValueError("C4a requires strictly increasing timestamps within every location")
    return result


def _validate_observations(observations: list, *, locations_per_view: int, frames_per_location: int, smoke: bool) -> dict[str, list]:
    groups = _group_by_location(observations)
    expected_locations = 3 * locations_per_view
    expected_frames = expected_locations * frames_per_location
    if len(groups) != expected_locations or len(observations) != expected_frames or any(len(sequence) != frames_per_location for sequence in groups.values()):
        raise ValueError("C4a observations do not match the requested balanced location/frame count")
    if not smoke and (expected_locations != 9 or expected_frames != 450):
        raise ValueError("formal C4a requires 9 locations and 450 frames")
    return groups


def _dvf_values(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    values = dvf_metrics(prediction, target)
    values["dynamic_normalized_epe"] = values["endpoint_error_mm"] / max(values["gt_dynamic_rms_mm"], EPSILON)
    return values


@torch.inference_mode()
def _evaluate(model: SourceFirstDynamicModel, groups: dict[str, list], motion: GroundTruthMotion, *, grid_shape: tuple[int, int, int]) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, float]]:
    grid, _ = SourceFirstDynamicModel._physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, grid_shape)
    rows: dict[str, dict[str, float]] = {}
    arrays: dict[str, np.ndarray] = {}
    for location, sequence in groups.items():
        full_pred: list[torch.Tensor] = []; full_gt: list[torch.Tensor] = []
        center_pred: list[torch.Tensor] = []; center_gt: list[torch.Tensor] = []
        image_losses: list[torch.Tensor] = []; timestamps: list[float] = []; scores: list[torch.Tensor] = []
        for observation in sequence:
            timestamp = grid.new_tensor([observation.timestamp_s])
            score = _gt_respiratory_scores(motion, timestamp)
            field = ScoreWeightedMBCField(model.respiratory_mbc, score)
            full_pred.append(field(grid)[0]); full_gt.append(motion.respiratory_dvf_mm(grid, timestamp)[0])
            uv = _full_image_uv(observation)
            centers = SourceFirstDynamicModel._pixel_world(observation, uv)
            center_pred.append(field(centers[None])[0]); center_gt.append(motion.respiratory_dvf_mm(centers[None], timestamp)[0])
            prediction = _render_with_gt_score(model, observation, uv, motion)
            image_losses.append((prediction - observation.image[0, uv[:, 1].long(), uv[:, 0].long()]).square().mean())
            timestamps.append(observation.timestamp_s); scores.append(score[0])
        full_values = _dvf_values(torch.stack(full_pred), torch.stack(full_gt))
        center_values = _dvf_values(torch.stack(center_pred), torch.stack(center_gt))
        rows[location] = {"image_data_mse": float(torch.stack(image_losses).mean()), **{f"full_fov_{name}": value for name, value in full_values.items()}, **{f"pixel_center_{name}": value for name, value in center_values.items()}}
        stem = location.replace("/", "_")
        arrays[f"{stem}_timestamps_s"] = np.asarray(timestamps)
        arrays[f"{stem}_gt_model_scores"] = torch.stack(scores).cpu().numpy()
    reference_time = grid.new_tensor([1.0 / (4.0 * motion.respiratory_frequency_hz)])
    reference_score = _gt_respiratory_scores(motion, reference_time)
    spatial_prediction = ScoreWeightedMBCField(model.respiratory_mbc, reference_score)(grid)[0]
    spatial_target = motion.respiratory_dvf_mm(grid, reference_time)[0]
    return rows, arrays, vector_field_agreement_metrics(spatial_prediction, spatial_target)


def _parameter_change_audit(model: SourceFirstDynamicModel, before: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    after = dict(model.named_parameters())
    if set(after) != set(before):
        raise RuntimeError("C4a model parameter set changed during intervention")
    changed = [name for name, parameter in after.items() if not torch.equal(before[name], parameter.detach().cpu())]
    allowed = _level0_parameter_names(model)
    unexpected = set(changed) - allowed
    if unexpected:
        raise RuntimeError("C4a modified frozen parameters: " + ", ".join(sorted(unexpected)))
    return changed, sorted(set(before) - set(changed))


def _assert_only_level0_changed(model: SourceFirstDynamicModel, before: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    changed, unchanged = _parameter_change_audit(model, before)
    if not changed:
        raise RuntimeError("C4a training did not update respiratory_mbc.level0")
    return changed, unchanged


def _write_curve(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "data_mse", "mbc_normalization", "total_loss", "resp_mbc_level0_parameter_norm"])
        writer.writeheader(); writer.writerows(rows)


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is non-empty: {path}; pass --overwrite to reuse it")
        for item in path.iterdir():
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    path.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if args.init == "checkpoint" and args.checkpoint is None:
        raise ValueError("--init checkpoint requires --checkpoint")
    if args.init == "fresh" and args.checkpoint is not None:
        raise ValueError("--checkpoint is only valid with --init checkpoint")
    if not args.smoke and (args.manifest is None or args.qc_table is None or args.canonical_domain is None):
        raise ValueError("formal C4a requires --manifest, --qc-table, and --canonical-domain")
    _prepare_output(args.output_dir, args.overwrite); _set_seed(args.seed)
    with args.source_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with args.experiment_config.open(encoding="utf-8") as handle:
        synthetic = yaml.safe_load(handle)["synthetic"]
    domain = _load_domain(args.canonical_domain)
    full_motion = GroundTruthMotion.from_domain(domain, respiratory_frequency_hz=float(synthetic["respiratory_frequency_hz"]), cardiac_frequency_hz=float(synthetic["cardiac_frequency_hz"]), respiratory_amplitude_mm=float(synthetic["respiratory_amplitude_mm"]), cardiac_amplitude_mm=float(synthetic["cardiac_amplitude_mm"]))
    motion = _respiratory_only_motion(full_motion)
    if motion.cardiac_amplitude_mm != 0.0:
        raise RuntimeError("C4a target must have exactly zero cardiac amplitude")
    image_shape = (8, 8) if args.smoke else tuple(int(value) for value in synthetic["image_shape"])
    psf_samples = 1 if args.smoke else int(synthetic["psf_samples"])
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=device) if args.manifest is not None else fallback_geometry(locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, image_shape=image_shape, motion=motion, device=device)
    observations = _build_observations(frames, image_shape=image_shape, phantom=AnalyticCanonicalPhantom(domain).to(device), motion=motion, psf_samples=psf_samples, seed=args.seed)
    groups = _validate_observations(observations, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, smoke=args.smoke)
    if args.init == "checkpoint":
        recovery_arrays = args.checkpoint.parent / "recovery_arrays.npz"
        if not recovery_arrays.is_file() or not validate_source_timestamps(recovery_arrays, groups):
            raise ValueError("C4a checkpoint initialization requires matching checkpoint-adjacent recovery_arrays.npz timestamps")
    model = _build_intervention_model(config, domain, n_dynamic_frames=len(observations), init_mode=args.init, checkpoint=args.checkpoint, seed=args.seed, device=device)
    optimizer = _configure_level0_only(model, learning_rate=float(config["training"]["optimizer"]["stage2"]["respiratory_mbc_lr"]))
    before = parameter_snapshot(model)
    encoder_calls = [0]
    hook = model.film_encoder.register_forward_hook(lambda *_: encoder_calls.__setitem__(0, encoder_calls[0] + 1))
    try:
        initial_rows, initial_arrays, initial_spatial = _evaluate(model, groups, motion, grid_shape=tuple(args.evaluation_grid_shape))
        sampler = ViewLocationBalancedSampler(observations, seed=args.seed)
        curve: list[dict[str, float | int]] = []
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            losses = [_intervention_loss_terms(model, observation, motion, pixel_uv=_sample_pixels(observation, args.pixel_samples), objective=args.objective, normalization_weight=float(config["training"]["loss_weights"]["mbc_normalization"])) for observation in sampler.sample_step(observations_per_update=args.observations_per_update)]
            data_mse = torch.stack([item["data_mse"] for item in losses]).mean()
            normalization = _normalization_loss(model)
            total_loss = data_mse if args.objective == "data_only" else data_mse + float(config["training"]["loss_weights"]["mbc_normalization"]) * normalization
            total_loss.backward()
            if step == 1:
                if not any(parameter.grad is not None and torch.isfinite(parameter.grad).all() and float(parameter.grad.norm()) > 0.0 for parameter in model.respiratory_mbc.levels[0].parameters()):
                    raise RuntimeError("C4a data loss did not reach respiratory_mbc.level0")
                if any(parameter.grad is not None for parameter in model.film_encoder.parameters()) or any(parameter.grad is not None for level in model.respiratory_mbc.levels[1:] for parameter in level.parameters()) or any(parameter.grad is not None for parameter in model.cardiac_mbc.parameters()) or any(parameter.grad is not None for parameter in model.canonical.parameters()):
                    raise RuntimeError("C4a backward reached a frozen non-level0 parameter")
            optimizer.step()
            level0_norm = torch.cat([parameter.detach().reshape(-1) for parameter in model.respiratory_mbc.levels[0].parameters()]).norm()
            curve.append({"step": step, "data_mse": float(data_mse.detach()), "mbc_normalization": float(normalization.detach()), "total_loss": float(total_loss.detach()), "resp_mbc_level0_parameter_norm": float(level0_norm)})
        final_rows, final_arrays, final_spatial = _evaluate(model, groups, motion, grid_shape=tuple(args.evaluation_grid_shape))
        if encoder_calls[0] != 0:
            raise RuntimeError("C4a intervention called the encoder")
    finally:
        hook.remove()
    changed, unchanged = _assert_only_level0_changed(model, before)
    frozen_audit = {
        "encoder_unchanged": all(name.startswith("film_encoder.") for name in unchanged if name.startswith("film_encoder.")),
        "canonical_unchanged": all(name.startswith("canonical.") for name in unchanged if name.startswith("canonical.")),
        "cardiac_mbc_unchanged": all(name.startswith("cardiac_mbc.") for name in unchanged if name.startswith("cardiac_mbc.")),
        "respiratory_mbc_level1_unchanged": all(name.startswith("respiratory_mbc.levels.1.") for name in unchanged if name.startswith("respiratory_mbc.levels.1.")),
        "respiratory_mbc_level2_unchanged": all(name.startswith("respiratory_mbc.levels.2.") for name in unchanged if name.startswith("respiratory_mbc.levels.2.")),
        "uncertainty_unchanged": all(name.startswith("uncertainty.") for name in unchanged if name.startswith("uncertainty.")),
    }
    if not all(frozen_audit.values()):
        raise RuntimeError("C4a frozen-module parameter audit failed")
    initial = {"per_location": initial_rows, "aggregate": aggregate_location_metrics(initial_rows), "spatial_field_reference": initial_spatial}
    final = {"per_location": final_rows, "aggregate": aggregate_location_metrics(final_rows), "spatial_field_reference": final_spatial}
    _write_curve(args.output_dir / "training_curve.csv", curve)
    (args.output_dir / "metrics_initial.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "metrics_final.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output_dir / "recovery_arrays.npz", **initial_arrays, **{f"final_{name}": value for name, value in final_arrays.items()})
    summary = {
        "experiment": "Exp2-C4a GT-score MBC-only", "git_sha": _git_sha(), "source_config": str(args.source_config), "experiment_config": str(args.experiment_config), "manifest": None if args.manifest is None else str(args.manifest), "qc_table": None if args.qc_table is None else str(args.qc_table), "canonical_domain": None if args.canonical_domain is None else str(args.canonical_domain), "seed": args.seed,
        "n_locations": len(groups), "n_frames": len(observations), "intervention_target": "respiratory_only", "cardiac_amplitude_mm": motion.cardiac_amplitude_mm, "gt_score_definition": "score[:,0,:] = respiratory_amplitude_mm * sin(2*pi*respiratory_frequency_hz*timestamp_s)", "gt_score_carries_amplitude": True, "active_respiratory_levels": 1, "trainable_module": "respiratory_mbc.level0", "init_mode": args.init, "checkpoint": None if args.checkpoint is None else str(args.checkpoint), "objective": args.objective, "steps": args.steps, "learning_rate": float(config["training"]["optimizer"]["stage2"]["respiratory_mbc_lr"]), "observations_per_update": args.observations_per_update, "pixel_samples": args.pixel_samples, "formal_training_budget_provenance": "paper-aligned stage2a 50+200, stage2b 50+200, stage2c 50+200 = 750 steps", "parameters_changed": changed, "parameters_unchanged": unchanged, "frozen_parameter_audit": frozen_audit, "trainable_parameter_names": sorted(_level0_parameter_names(model)), "encoder_calls": encoder_calls[0], "initial_metrics": initial, "final_metrics": final,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2-C4a fixed GT respiratory score, train MBC level0 only")
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs/source_first_change4_paperaligned.yaml"); parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml")
    parser.add_argument("--manifest", type=Path); parser.add_argument("--qc-table", type=Path); parser.add_argument("--canonical-domain", type=Path); parser.add_argument("--checkpoint", type=Path); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--init", choices=("fresh", "checkpoint"), default="fresh"); parser.add_argument("--objective", choices=("data_only", "formal_mbc_norm"), default="data_only"); parser.add_argument("--steps", type=int, default=750); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50); parser.add_argument("--observations-per-update", type=int, default=32); parser.add_argument("--pixel-samples", type=int, default=256); parser.add_argument("--evaluation-grid-shape", type=lambda value: tuple(int(item) for item in value.split(",")), default=(8, 8, 8)); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.observations_per_update < 3 or args.pixel_samples <= 0 or len(args.evaluation_grid_shape) != 3 or min(args.evaluation_grid_shape) < 2:
        parser.error("invalid C4a training or evaluation arguments")
    summary = run(args)
    print(f"Exp2-C4a: {args.output_dir / 'summary.json'} ({summary['n_locations']} locations)")


if __name__ == "__main__":
    main()
