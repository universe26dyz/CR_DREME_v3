#!/usr/bin/env python3
"""Exp2-C4c: deterministic oracle-field respiratory coefficient loss landscape."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import torch
import yaml

from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField, SequentialPullbackMotion
from cardioresp4d.training.model import SourceFirstDynamicModel
from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, _load_domain, fallback_geometry, geometry_from_manifest
from experiments.c4_diagnostics.exp2_checkpoint_common import parameter_snapshot
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import _OracleRespiratoryMBC, _ZeroDisplacement, _assert_oracle_matches_gt, _build_intervention_model, _full_image_uv, _git_sha, _respiratory_only_motion, _set_seed, _validate_observations
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom


EPSILON = 1e-12


def _build_model(config: dict[str, Any], domain: dict[str, object], *, n_dynamic_frames: int, seed: int, device: torch.device | None = None):
    return _build_intervention_model(config, domain, n_dynamic_frames=n_dynamic_frames, seed=seed, device=device)


def _coefficient_grid(minimum: float, maximum: float, step: float) -> np.ndarray:
    if not np.isfinite([minimum, maximum, step]).all() or step <= 0 or maximum <= minimum:
        raise ValueError("coefficient range must be finite, ordered, and have a positive step")
    grid = np.arange(minimum, maximum + .5 * step, step, dtype=np.float64)
    if not np.any(np.isclose(grid, 0.0, atol=1e-8)):
        raise ValueError("coefficient grid must include zero")
    return grid


def _render_coefficient(model, observation, oracle: _OracleRespiratoryMBC, coefficient_mm: float, *, seed: int) -> torch.Tensor:
    uv = _full_image_uv(observation); points = SourceFirstDynamicModel._pixel_world(observation, uv)
    resolution = torch.stack((observation.pixel_spacing_mm[1].to(points), observation.pixel_spacing_mm[0].to(points), points.new_tensor(observation.slice_thickness_mm))).expand(points.shape[0], 3)
    score = points.new_full((1, 1, 3), coefficient_mm)
    motion = SequentialPullbackMotion(ScoreWeightedMBCField(oracle, score), _ZeroDisplacement())
    axes = {"row_direction": observation.row_direction.to(points).expand(points.shape[0], -1), "column_direction": observation.column_direction.to(points).expand(points.shape[0], -1), "normal": observation.normal.to(points).expand(points.shape[0], -1)}
    devices = [] if points.device.type == "cpu" else [torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if points.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        return model.psf(model.canonical, points, resolution, **axes, motion=motion)["predicted_intensity"]


def _mse_at(model, observation, oracle: _OracleRespiratoryMBC, coefficient_mm: float, *, seed: int) -> float:
    prediction = _render_coefficient(model, observation, oracle, coefficient_mm, seed=seed)
    target = observation.image[0].reshape(-1)
    value = (prediction - target).square().mean()
    if not torch.isfinite(value):
        raise RuntimeError("C4c encountered a non-finite image loss")
    return float(value)


def _evaluate_frame(model, observation, oracle: _OracleRespiratoryMBC, motion: GroundTruthMotion, coefficient_grid_mm: np.ndarray, *, finite_difference_delta_mm: float, seed: int) -> tuple[dict[str, Any], np.ndarray]:
    gt = float(motion.respiratory_amplitude_mm * motion.respiratory_score(torch.tensor(observation.timestamp_s)))
    frame_seed = seed + int(observation.dynamic_frame_id)
    curve = np.asarray([_mse_at(model, observation, oracle, float(coefficient), seed=frame_seed) for coefficient in coefficient_grid_mm], dtype=np.float64)
    if not np.isfinite(curve).all():
        raise RuntimeError("C4c encountered non-finite landscape values")
    index = int(curve.argmin()); argmin = float(coefficient_grid_mm[index]); argmin_mse = float(curve[index])
    mse_gt = _mse_at(model, observation, oracle, gt, seed=frame_seed); mse_zero = _mse_at(model, observation, oracle, 0.0, seed=frame_seed)
    plus = _mse_at(model, observation, oracle, gt + finite_difference_delta_mm, seed=frame_seed); minus = _mse_at(model, observation, oracle, gt - finite_difference_delta_mm, seed=frame_seed)
    dynamic_range = (curve.max() - curve.min()) / max(float(curve.min()), EPSILON)
    return {"view": observation.view, "slice_id": observation.slice_id, "timestamp_s": float(observation.timestamp_s), "gt_coefficient_mm": gt, "argmin_coefficient_mm": argmin, "argmin_mse": argmin_mse, "mse_at_gt": mse_gt, "mse_at_zero": mse_zero, "absolute_coefficient_error_mm": abs(argmin - gt), "signed_coefficient_error_mm": argmin - gt, "loss_improvement_argmin_vs_zero": mse_zero - argmin_mse, "loss_improvement_gt_vs_zero": mse_zero - mse_gt, "dL_da_at_gt": (plus - minus) / (2.0 * finite_difference_delta_mm), "d2L_da2_at_gt": (plus - 2.0 * mse_gt + minus) / finite_difference_delta_mm ** 2, "loss_dynamic_range": dynamic_range}, curve


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    gt = np.asarray([row["gt_coefficient_mm"] for row in rows], dtype=float); argmin = np.asarray([row["argmin_coefficient_mm"] for row in rows], dtype=float); error = argmin - gt
    slope, intercept = np.polyfit(gt, argmin, 1) if len(rows) >= 2 and np.std(gt) > EPSILON else (float("nan"), float("nan"))
    pearson = float(np.corrcoef(gt, argmin)[0, 1]) if len(rows) >= 2 and np.std(gt) > EPSILON and np.std(argmin) > EPSILON else float("nan")
    ratios_gt = np.asarray([row["mse_at_gt"] / max(row["mse_at_zero"], EPSILON) for row in rows]); ratios_min = np.asarray([row["argmin_mse"] / max(row["mse_at_zero"], EPSILON) for row in rows])
    return {"n_frames": float(len(rows)), "pearson_argmin_vs_gt": pearson, "mae_argmin_vs_gt_mm": float(np.abs(error).mean()), "rmse_argmin_vs_gt_mm": float(np.sqrt(np.square(error).mean())), "slope": float(slope), "intercept_mm": float(intercept), "fraction_abs_error_le_0p25_mm": float((np.abs(error) <= .25).mean()), "fraction_abs_error_le_0p5_mm": float((np.abs(error) <= .5).mean()), "fraction_abs_argmin_le_0p5_mm": float((np.abs(argmin) <= .5).mean()), "median_mse_gt_over_zero": float(np.median(ratios_gt)), "median_argmin_mse_over_zero": float(np.median(ratios_min)), "median_abs_dL_da_gt": float(np.median(np.abs([row["dL_da_at_gt"] for row in rows]))), "fraction_curvature_gt_positive": float(np.mean([row["d2L_da2_at_gt"] > 0 for row in rows])), "median_loss_dynamic_range": float(np.median([row["loss_dynamic_range"] for row in rows]))}


def _grouped_aggregates(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(str(row[key]) for key in keys), []).append(row)
    return [{**dict(zip(keys, group)), **_aggregate(items)} for group, items in sorted(grouped.items())]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is non-empty: {path}; pass --overwrite to reuse it")
        for item in path.iterdir():
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _assert_parameters_unchanged(model, before: dict[str, torch.Tensor]) -> None:
    after = dict(model.named_parameters())
    if set(after) != set(before) or any(not torch.equal(before[name], parameter.detach().cpu()) for name, parameter in after.items()):
        raise RuntimeError("C4c changed model parameters")


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if not args.smoke and (args.manifest is None or args.qc_table is None or args.canonical_domain is None):
        raise ValueError("formal C4c requires --manifest, --qc-table, and --canonical-domain")
    grid = _coefficient_grid(args.coefficient_min_mm, args.coefficient_max_mm, args.coefficient_step_mm)
    if not args.smoke and (grid.min() > -6.0 or grid.max() < 6.0):
        raise ValueError("formal C4c coefficient grid must cover the [-6,+6] mm GT range")
    if args.finite_difference_delta_mm <= 0:
        raise ValueError("finite-difference delta must be positive")
    _prepare_output(args.output_dir, args.overwrite); _set_seed(args.seed)
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); synthetic = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))["synthetic"]; domain = _load_domain(args.canonical_domain)
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(domain, respiratory_frequency_hz=float(synthetic["respiratory_frequency_hz"]), cardiac_frequency_hz=float(synthetic["cardiac_frequency_hz"]), respiratory_amplitude_mm=float(synthetic["respiratory_amplitude_mm"]), cardiac_amplitude_mm=float(synthetic["cardiac_amplitude_mm"])))
    if motion.cardiac_amplitude_mm != 0.0:
        raise RuntimeError("C4c target must have exactly zero cardiac amplitude")
    image_shape = (8, 8) if args.smoke else tuple(int(value) for value in synthetic["image_shape"]); psf_samples = 1 if args.smoke else int(synthetic["psf_samples"])
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=device) if args.manifest is not None else fallback_geometry(locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, image_shape=image_shape, motion=motion, device=device)
    observations = _build_observations(frames, image_shape=image_shape, phantom=AnalyticCanonicalPhantom(domain).to(device), motion=motion, psf_samples=psf_samples, seed=args.seed)
    groups = _validate_observations(observations, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, smoke=args.smoke)
    model = _build_model(config, domain, n_dynamic_frames=len(observations), seed=args.seed, device=device); oracle = _OracleRespiratoryMBC(motion).to(device); check_points, _ = SourceFirstDynamicModel._physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, (3, 3, 3)); _assert_oracle_matches_gt(oracle, motion, check_points); before = parameter_snapshot(model)
    encoder_calls = [0]; mbc_calls = [0]; encoder_hook = model.film_encoder.register_forward_hook(lambda *_: encoder_calls.__setitem__(0, encoder_calls[0] + 1)); mbc_hook = model.respiratory_mbc.register_forward_hook(lambda *_: mbc_calls.__setitem__(0, mbc_calls[0] + 1))
    try:
        rows: list[dict[str, Any]] = []; curves: list[np.ndarray] = []
        for observation in observations:
            row, curve = _evaluate_frame(model, observation, oracle, motion, grid, finite_difference_delta_mm=args.finite_difference_delta_mm, seed=args.seed)
            rows.append(row); curves.append(curve)
    finally:
        encoder_hook.remove(); mbc_hook.remove()
    if encoder_calls[0] != 0 or mbc_calls[0] != 0:
        raise RuntimeError("C4c called encoder or learned respiratory MBC")
    _assert_parameters_unchanged(model, before)
    by_location = _grouped_aggregates(rows, ("view", "slice_id")); by_view = _grouped_aggregates(rows, ("view",)); overall = _aggregate(rows)
    _write_csv(args.output_dir / "per_frame_landscape.csv", rows); _write_csv(args.output_dir / "aggregate_by_location.csv", by_location); _write_csv(args.output_dir / "aggregate_by_view.csv", by_view)
    np.savez_compressed(args.output_dir / "landscape_arrays.npz", coefficient_grid_mm=grid, loss_curves=np.asarray(curves), gt_coefficients_mm=np.asarray([row["gt_coefficient_mm"] for row in rows]), argmin_coefficients_mm=np.asarray([row["argmin_coefficient_mm"] for row in rows]), timestamps_s=np.asarray([row["timestamp_s"] for row in rows]), frame_keys=np.asarray([f"{row['view']}/{row['slice_id']}" for row in rows]))
    summary = {"experiment": "Exp2-C4c oracle-field respiratory score loss landscape", "git_sha": _git_sha(), "seed": args.seed, "n_locations": len(groups), "n_frames": len(observations), "cardiac_amplitude_mm": motion.cardiac_amplitude_mm, "oracle_spatial_field": "C4b _OracleRespiratoryMBC", "coefficient_grid_mm": {"min": args.coefficient_min_mm, "max": args.coefficient_max_mm, "step": args.coefficient_step_mm}, "finite_difference_delta_mm": args.finite_difference_delta_mm, "landscape_protocol": "full-image MSE; frame-specific PSF RNG replayed for every coefficient; exact GT coefficient evaluated outside grid", "encoder_calls": encoder_calls[0], "learned_respiratory_mbc_calls": mbc_calls[0], "parameters_unchanged": True, "overall": overall, "by_location": by_location, "by_view": by_view}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2-C4c oracle-field respiratory score loss landscape")
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs/source_first_change4_paperaligned.yaml"); parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml"); parser.add_argument("--manifest", type=Path); parser.add_argument("--qc-table", type=Path); parser.add_argument("--canonical-domain", type=Path); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50); parser.add_argument("--coefficient-min-mm", type=float, default=-9.0); parser.add_argument("--coefficient-max-mm", type=float, default=9.0); parser.add_argument("--coefficient-step-mm", type=float, default=.1); parser.add_argument("--finite-difference-delta-mm", type=float, default=.05); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); summary = run(args); print(f"Exp2-C4c: {args.output_dir / 'summary.json'} ({summary['n_frames']} frames)")


if __name__ == "__main__":
    main()
