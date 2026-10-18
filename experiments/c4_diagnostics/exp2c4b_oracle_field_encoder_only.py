#!/usr/bin/env python3
"""Exp2-C4b: fixed analytic respiratory field, train encoder scores only."""
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

from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField, SequentialPullbackMotion
from cardioresp4d.training.build_model import build_source_first_model
from cardioresp4d.training.model import SourceFirstDynamicModel
from cardioresp4d.training.sampler import ViewLocationBalancedSampler
from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, _load_domain, fallback_geometry, geometry_from_manifest
from experiments.c4_diagnostics.exp2_checkpoint_common import parameter_snapshot
from experiments.c4_diagnostics.synthetic.metrics import aggregate_location_metrics, dvf_metrics
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from experiments.c4_diagnostics.synthetic.render import PhantomCanonicalAdapter


EPSILON = 1e-12
_ENCODER_PREFIXES = ("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp.")


class _ZeroDisplacement(nn.Module):
    def forward(self, points_mm: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(points_mm)


class _OracleRespiratoryMBC(nn.Module):
    """One-level GT unit-amplitude respiratory basis for ScoreWeightedMBCField."""
    def __init__(self, motion: GroundTruthMotion) -> None:
        super().__init__()
        if motion.respiratory_frequency_hz <= 0 or motion.respiratory_amplitude_mm <= 0:
            raise ValueError("C4b oracle requires positive respiratory frequency and amplitude")
        self.motion = motion
        self.active_level_count = 1

    def forward(self, points_mm: torch.Tensor) -> torch.Tensor:
        if points_mm.ndim != 3 or points_mm.shape[-1] != 3:
            raise ValueError("oracle points must have shape [B,N,3]")
        t_ref = points_mm.new_tensor(1.0 / (4.0 * self.motion.respiratory_frequency_hz))
        basis = self.motion.respiratory_dvf_mm(points_mm, t_ref) / self.motion.respiratory_amplitude_mm
        return basis[:, None]


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
    timestamps = torch.as_tensor(timestamps_s).reshape(-1)
    return (motion.respiratory_amplitude_mm * motion.respiratory_score(timestamps))[:, None, None].expand(-1, 1, 3)


def _build_intervention_model(config: dict[str, Any], domain: dict[str, object], *, n_dynamic_frames: int, seed: int, device: torch.device | None = None) -> SourceFirstDynamicModel:
    device = torch.device("cpu") if device is None else device
    _set_seed(seed)
    model = build_source_first_model(config, domain, n_dynamic_frames=n_dynamic_frames, device=device)
    model.canonical = PhantomCanonicalAdapter(AnalyticCanonicalPhantom(domain)).to(device)
    if model.film_implementation != "full_geometry_single_film":
        raise ValueError("C4b primary intervention requires GeometryFiLMMotionEncoder")
    return model.to(device).eval()


def _encoder_parameter_names(model: SourceFirstDynamicModel) -> set[str]:
    return {name for name, _ in model.named_parameters() if name.startswith(_ENCODER_PREFIXES)}


def _configure_encoder_only(model: SourceFirstDynamicModel, *, learning_rate: float) -> torch.optim.Adam:
    if learning_rate <= 0:
        raise ValueError("film learning rate must be positive")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = [(name, parameter) for name, parameter in model.named_parameters() if name.startswith(_ENCODER_PREFIXES)]
    if not selected:
        raise RuntimeError("C4b found no respiratory encoder parameters")
    for _, parameter in selected:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam([parameter for _, parameter in selected], lr=learning_rate)
    owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if owned != {id(parameter) for _, parameter in selected} or any(parameter.requires_grad != (id(parameter) in owned) for parameter in model.parameters()):
        raise RuntimeError("C4b optimizer ownership is not respiratory-encoder-only")
    return optimizer


def _full_image_uv(observation) -> torch.Tensor:
    height, width = observation.image.shape[-2:]
    return torch.stack(torch.meshgrid(torch.arange(width, device=observation.image.device), torch.arange(height, device=observation.image.device), indexing="xy"), dim=-1).reshape(-1, 2).to(observation.image.dtype)


def _sample_pixels(observation, count: int) -> torch.Tensor:
    height, width = observation.image.shape[-2:]
    linear = torch.randint(height * width, (min(count, height * width),), device=observation.image.device)
    return torch.stack((linear.remainder(width), torch.div(linear, width, rounding_mode="floor")), dim=-1).to(observation.image.dtype)


def _encoder_resp_score(model: SourceFirstDynamicModel, observation) -> torch.Tensor:
    score = model.encode_motion_scores(observation)["resp_scores"][:, :1, :]
    if score.shape != (1, 1, 3):
        raise RuntimeError("C4b encoder must provide level-0 [1,1,3] respiratory scores")
    return score


def _render_with_oracle_field(model: SourceFirstDynamicModel, observation, pixel_uv: torch.Tensor, oracle: _OracleRespiratoryMBC) -> tuple[torch.Tensor, torch.Tensor]:
    points = SourceFirstDynamicModel._pixel_world(observation, pixel_uv.to(observation.image))
    spacing = observation.pixel_spacing_mm.to(points)
    resolution = torch.stack((spacing[1], spacing[0], points.new_tensor(observation.slice_thickness_mm))).expand(points.shape[0], 3)
    score = _encoder_resp_score(model, observation)
    pullback = SequentialPullbackMotion(ScoreWeightedMBCField(oracle, score), _ZeroDisplacement())
    axes = {"row_direction": observation.row_direction.to(points).expand(points.shape[0], -1), "column_direction": observation.column_direction.to(points).expand(points.shape[0], -1), "normal": observation.normal.to(points).expand(points.shape[0], -1)}
    return model.psf(model.canonical, points, resolution, **axes, motion=pullback)["predicted_intensity"], score


def _intervention_loss_terms(model: SourceFirstDynamicModel, observation, oracle: _OracleRespiratoryMBC, *, pixel_uv: torch.Tensor) -> dict[str, torch.Tensor]:
    target = observation.image[0, pixel_uv[:, 1].long(), pixel_uv[:, 0].long()]
    prediction, score = _render_with_oracle_field(model, observation, pixel_uv, oracle)
    return {"data_mse": (prediction - target).square().mean(), "resp_score": score}


def _group_by_location(observations: list) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for observation in observations:
        grouped.setdefault(f"{observation.view}/{observation.slice_id}", []).append(observation)
    result = {key: sorted(values, key=lambda item: item.timestamp_s) for key, values in grouped.items()}
    if any(any(second.timestamp_s <= first.timestamp_s for first, second in zip(sequence, sequence[1:])) for sequence in result.values()):
        raise ValueError("C4b requires strictly increasing timestamps within each location")
    return result


def _validate_observations(observations: list, *, locations_per_view: int, frames_per_location: int, smoke: bool) -> dict[str, list]:
    groups = _group_by_location(observations)
    if len(groups) != 3 * locations_per_view or len(observations) != 3 * locations_per_view * frames_per_location or any(len(sequence) != frames_per_location for sequence in groups.values()):
        raise ValueError("C4b observations do not match requested balanced geometry")
    if not smoke and (len(groups) != 9 or len(observations) != 450):
        raise ValueError("formal C4b requires 9 locations and 450 frames")
    return groups


def _dvf_values(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    values = dvf_metrics(prediction, target)
    pred_dynamic = prediction - prediction.mean(dim=0, keepdim=True)
    gt_dynamic = target - target.mean(dim=0, keepdim=True)
    dynamic_epe = (pred_dynamic - gt_dynamic).square().sum(dim=-1).sqrt().mean()
    values["dynamic_normalized_epe"] = float(dynamic_epe / gt_dynamic.square().mean().sqrt().clamp_min(EPSILON))
    return values


def _assert_oracle_matches_gt(oracle: _OracleRespiratoryMBC, motion: GroundTruthMotion, points: torch.Tensor) -> None:
    timestamps = points.new_tensor([0.0, .37, .81])
    expanded_points = points.expand(timestamps.numel(), -1, -1)
    predicted = ScoreWeightedMBCField(oracle, _gt_respiratory_scores(motion, timestamps))(expanded_points)
    target = motion.respiratory_dvf_mm(expanded_points, timestamps)
    if not torch.allclose(predicted, target, atol=1e-6, rtol=1e-6):
        raise RuntimeError("C4b oracle spatial field does not reproduce analytic GT respiratory DVF")


def _score_values(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_dynamic, gt_dynamic = prediction - prediction.mean(0, keepdim=True), target - target.mean(0, keepdim=True)
    pred_rms, gt_rms = pred_dynamic.square().mean().sqrt(), gt_dynamic.square().mean().sqrt()
    return {"predicted_score_temporal_mean": float(prediction.mean()), "predicted_score_static_rms": float(prediction.mean(0).square().mean().sqrt()), "predicted_score_temporal_rms": float(pred_rms), "gt_score_temporal_rms": float(gt_rms), "temporal_rms_ratio": float(pred_rms / gt_rms.clamp_min(EPSILON)), "gt_correlation": float((pred_dynamic * gt_dynamic).sum() / (pred_dynamic.norm() * gt_dynamic.norm()).clamp_min(EPSILON)), "score_mse": float((prediction - target).square().mean()), "score_mae": float((prediction - target).abs().mean()), "static_to_temporal_rms_ratio": float(prediction.mean(0).square().mean().sqrt() / pred_rms.clamp_min(EPSILON))}


@torch.inference_mode()
def _evaluate(model: SourceFirstDynamicModel, groups: dict[str, list], motion: GroundTruthMotion, oracle: _OracleRespiratoryMBC, *, grid_shape: tuple[int, int, int]) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    grid, _ = SourceFirstDynamicModel._physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, grid_shape)
    rows: dict[str, dict[str, float]] = {}; arrays: dict[str, np.ndarray] = {}
    for location, sequence in groups.items():
        full_pred: list[torch.Tensor] = []; full_gt: list[torch.Tensor] = []; center_pred: list[torch.Tensor] = []; center_gt: list[torch.Tensor] = []; image_losses: list[torch.Tensor] = []; predicted_scores: list[torch.Tensor] = []; gt_scores: list[torch.Tensor] = []; timestamps: list[float] = []
        for observation in sequence:
            score = _encoder_resp_score(model, observation)
            timestamp = grid.new_tensor([observation.timestamp_s]); gt_score = _gt_respiratory_scores(motion, timestamp)
            field = ScoreWeightedMBCField(oracle, score)
            full_pred.append(field(grid)[0]); full_gt.append(motion.respiratory_dvf_mm(grid, timestamp)[0])
            uv = _full_image_uv(observation); centers = SourceFirstDynamicModel._pixel_world(observation, uv)
            center_pred.append(field(centers[None])[0]); center_gt.append(motion.respiratory_dvf_mm(centers[None], timestamp)[0])
            image, _ = _render_with_oracle_field(model, observation, uv, oracle)
            image_losses.append((image - observation.image[0, uv[:, 1].long(), uv[:, 0].long()]).square().mean())
            predicted_scores.append(score[0]); gt_scores.append(gt_score[0]); timestamps.append(observation.timestamp_s)
        row = {"image_data_mse": float(torch.stack(image_losses).mean()), **_score_values(torch.stack(predicted_scores), torch.stack(gt_scores)), **{f"full_fov_{name}": value for name, value in _dvf_values(torch.stack(full_pred), torch.stack(full_gt)).items()}, **{f"pixel_center_{name}": value for name, value in _dvf_values(torch.stack(center_pred), torch.stack(center_gt)).items()}}
        rows[location] = row; stem = location.replace("/", "_")
        arrays[f"{stem}_timestamps_s"] = np.asarray(timestamps); arrays[f"{stem}_predicted_scores"] = torch.stack(predicted_scores).cpu().numpy(); arrays[f"{stem}_gt_scores"] = torch.stack(gt_scores).cpu().numpy()
    return rows, arrays


def _assert_only_encoder_changed(model: SourceFirstDynamicModel, before: dict[str, torch.Tensor], *, require_change: bool = True) -> tuple[list[str], list[str]]:
    after = dict(model.named_parameters())
    if set(after) != set(before):
        raise RuntimeError("C4b model parameter set changed")
    changed = [name for name, parameter in after.items() if not torch.equal(before[name], parameter.detach().cpu())]
    unexpected = set(changed) - _encoder_parameter_names(model)
    if unexpected or (require_change and not changed):
        raise RuntimeError("C4b frozen-module audit failed")
    return changed, sorted(set(before) - set(changed))


def _write_curve(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "data_mse", "encoder_parameter_norm", "sampled_predicted_score_rms"])
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
    if args.init != "fresh":
        raise ValueError("C4b primary intervention supports --init fresh only")
    if not args.smoke and (args.manifest is None or args.qc_table is None or args.canonical_domain is None):
        raise ValueError("formal C4b requires --manifest, --qc-table, and --canonical-domain")
    _prepare_output(args.output_dir, args.overwrite); _set_seed(args.seed)
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); synthetic = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))["synthetic"]; domain = _load_domain(args.canonical_domain)
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(domain, respiratory_frequency_hz=float(synthetic["respiratory_frequency_hz"]), cardiac_frequency_hz=float(synthetic["cardiac_frequency_hz"]), respiratory_amplitude_mm=float(synthetic["respiratory_amplitude_mm"]), cardiac_amplitude_mm=float(synthetic["cardiac_amplitude_mm"])))
    if motion.cardiac_amplitude_mm != 0.0:
        raise RuntimeError("C4b target must have exactly zero cardiac amplitude")
    image_shape = (8, 8) if args.smoke else tuple(int(value) for value in synthetic["image_shape"]); psf_samples = 1 if args.smoke else int(synthetic["psf_samples"])
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=device) if args.manifest is not None else fallback_geometry(locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, image_shape=image_shape, motion=motion, device=device)
    observations = _build_observations(frames, image_shape=image_shape, phantom=AnalyticCanonicalPhantom(domain).to(device), motion=motion, psf_samples=psf_samples, seed=args.seed)
    groups = _validate_observations(observations, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, smoke=args.smoke)
    model = _build_intervention_model(config, domain, n_dynamic_frames=len(observations), seed=args.seed, device=device); oracle = _OracleRespiratoryMBC(motion).to(device); oracle_grid, _ = SourceFirstDynamicModel._physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, (3, 3, 3)); _assert_oracle_matches_gt(oracle, motion, oracle_grid); optimizer = _configure_encoder_only(model, learning_rate=float(config["training"]["optimizer"]["stage2"]["film_lr"])); before = parameter_snapshot(model)
    mbc_calls = [0]; hook = model.respiratory_mbc.register_forward_hook(lambda *_: mbc_calls.__setitem__(0, mbc_calls[0] + 1))
    try:
        initial_rows, initial_arrays = _evaluate(model, groups, motion, oracle, grid_shape=tuple(args.evaluation_grid_shape)); sampler = ViewLocationBalancedSampler(observations, seed=args.seed); curve: list[dict[str, float | int]] = []
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True); losses = [_intervention_loss_terms(model, observation, oracle, pixel_uv=_sample_pixels(observation, args.pixel_samples)) for observation in sampler.sample_step(observations_per_update=args.observations_per_update)]; data_mse = torch.stack([item["data_mse"] for item in losses]).mean(); data_mse.backward()
            if step == 1:
                for prefix in _ENCODER_PREFIXES:
                    if not any(name.startswith(prefix) and parameter.grad is not None and torch.isfinite(parameter.grad).all() and float(parameter.grad.norm()) > 0.0 for name, parameter in model.named_parameters()):
                        raise RuntimeError(f"C4b data loss did not reach {prefix}")
                if any(parameter.grad is not None for name, parameter in model.named_parameters() if not name.startswith(_ENCODER_PREFIXES)):
                    raise RuntimeError("C4b backward reached frozen parameters")
            optimizer.step(); parameters = [parameter.detach().reshape(-1) for name, parameter in model.named_parameters() if name.startswith(_ENCODER_PREFIXES)]; score = torch.cat([item["resp_score"].detach().reshape(-1) for item in losses])
            curve.append({"step": step, "data_mse": float(data_mse.detach()), "encoder_parameter_norm": float(torch.cat(parameters).norm()), "sampled_predicted_score_rms": float(score.square().mean().sqrt())})
        final_rows, final_arrays = _evaluate(model, groups, motion, oracle, grid_shape=tuple(args.evaluation_grid_shape))
    finally:
        hook.remove()
    if mbc_calls[0] != 0:
        raise RuntimeError("C4b called learned respiratory MBC")
    changed, unchanged = _assert_only_encoder_changed(model, before)
    frozen_audit = {"film_encoder_card_unchanged": not any(name.startswith("film_encoder.card.") for name in changed), "canonical_unchanged": not any(name.startswith("canonical.") for name in changed), "respiratory_mbc_unchanged": not any(name.startswith("respiratory_mbc.") for name in changed), "cardiac_mbc_unchanged": not any(name.startswith("cardiac_mbc.") for name in changed), "uncertainty_unchanged": not any(name.startswith("uncertainty.") for name in changed)}
    if not all(frozen_audit.values()):
        raise RuntimeError("C4b frozen-module audit failed")
    initial = {"per_location": initial_rows, "aggregate": aggregate_location_metrics(initial_rows)}; final = {"per_location": final_rows, "aggregate": aggregate_location_metrics(final_rows)}
    _write_curve(args.output_dir / "training_curve.csv", curve); (args.output_dir / "metrics_initial.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8"); (args.output_dir / "metrics_final.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8"); np.savez_compressed(args.output_dir / "recovery_arrays.npz", **initial_arrays, **{f"final_{name}": value for name, value in final_arrays.items()})
    summary = {"experiment": "Exp2-C4b oracle-field encoder-only", "git_sha": _git_sha(), "seed": args.seed, "n_locations": len(groups), "n_frames": len(observations), "intervention_target": "respiratory_only", "cardiac_amplitude_mm": motion.cardiac_amplitude_mm, "oracle_spatial_field": "analytic_ground_truth: respiratory_dvf_mm(x, quarter_cycle) / respiratory_amplitude_mm", "oracle_validation": "ScoreWeightedMBCField(oracle, gt_score) == analytic respiratory_dvf_mm", "active_score_levels": 1, "trainable_module": "film_encoder.image, geometry_mlp, head, resp", "init_mode": args.init, "objective": "data_only", "steps": args.steps, "film_lr": float(config["training"]["optimizer"]["stage2"]["film_lr"]), "observations_per_update": args.observations_per_update, "pixel_samples": args.pixel_samples, "parameters_changed": changed, "parameters_unchanged": unchanged, "encoder_parameter_change_audit": bool(changed), "frozen_module_audit": frozen_audit, "trainable_parameter_names": sorted(_encoder_parameter_names(model)), "learned_respiratory_mbc_calls": mbc_calls[0], "initial_metrics": initial, "final_metrics": final}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2-C4b fixed oracle respiratory field, train encoder scores only")
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs/source_first_change4_paperaligned.yaml"); parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml"); parser.add_argument("--manifest", type=Path); parser.add_argument("--qc-table", type=Path); parser.add_argument("--canonical-domain", type=Path); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--init", choices=("fresh",), default="fresh"); parser.add_argument("--steps", type=int, default=750); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50); parser.add_argument("--observations-per-update", type=int, default=32); parser.add_argument("--pixel-samples", type=int, default=256); parser.add_argument("--evaluation-grid-shape", type=lambda value: tuple(int(item) for item in value.split(",")), default=(8, 8, 8)); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.observations_per_update < 3 or args.pixel_samples <= 0 or len(args.evaluation_grid_shape) != 3 or min(args.evaluation_grid_shape) < 2:
        parser.error("invalid C4b training or evaluation arguments")
    summary = run(args); print(f"Exp2-C4b: {args.output_dir / 'summary.json'} ({summary['n_locations']} locations)")


if __name__ == "__main__":
    main()
