#!/usr/bin/env python3
"""Exp2-C2: frozen centered-respiratory-score dynamic gain sweep."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import torch

from cardioresp4d.training.model import SourceFirstDynamicModel
from experiments.c4_diagnostics.exp2_checkpoint_common import assert_parameters_unchanged, full_image_uv, matched_psf_sample_world_points, parameter_snapshot, prepare_frozen_motion_audit
from experiments.c4_diagnostics.exp2c_posthoc_score_centering import _dvf_values, _geometry_kwargs, _image_values, _respiratory_field, _write_rows, rendered_frame_image, render_resp_scores_from_cached_points
from experiments.c4_diagnostics.synthetic.oracle import build_target_psf_offset_cache


EPSILON = 1e-12
DEFAULT_GAINS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def gain_scores(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    """Per-location score DC removal followed by a scalar dynamic gain."""
    if not math.isfinite(alpha):
        raise ValueError("gain must be finite")
    return (scores - scores.mean(dim=0, keepdim=True)) * float(alpha)


def dynamic_cosine(predicted: torch.Tensor, target: torch.Tensor) -> float:
    pred_dynamic = predicted - predicted.mean(dim=0, keepdim=True)
    target_dynamic = target - target.mean(dim=0, keepdim=True)
    denominator = pred_dynamic.norm() * target_dynamic.norm()
    return float("nan") if float(denominator) <= EPSILON else float((pred_dynamic * target_dynamic).sum() / denominator)


def alpha_dvf_least_squares(predicted_alpha1: torch.Tensor, target: torch.Tensor) -> float:
    """Float64 scalar-amplitude oracle; leaves waveform and MBC field fixed."""
    source = predicted_alpha1.to(torch.float64)
    target64 = target.to(torch.float64)
    denominator = (source * source).sum()
    if not torch.isfinite(denominator) or float(denominator) <= EPSILON:
        return float("nan")
    numerator = (source * target64).sum()
    return float(numerator / denominator) if torch.isfinite(numerator) else float("nan")


def rendered_gain_frame(canonical, model: SourceFirstDynamicModel, query_points: torch.Tensor, scores: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
    return rendered_frame_image(render_resp_scores_from_cached_points(canonical, query_points, model, scores), image_shape)


def render_gain_images_from_cached_points(canonical, model: SourceFirstDynamicModel, query_points: torch.Tensor, scores_by_gain: dict[float, torch.Tensor], image_shape: tuple[int, int]) -> dict[float, torch.Tensor]:
    """Render all gains from one caller-owned matched-CRN query tensor."""
    return {gain: rendered_gain_frame(canonical, model, query_points, scores, image_shape) for gain, scores in scores_by_gain.items()}


def gain_grid_best(mse_by_gain: dict[float, float], gains: list[float]) -> tuple[float, float, bool]:
    finite = [(gain, mse_by_gain[gain]) for gain in gains if gain in mse_by_gain and math.isfinite(mse_by_gain[gain])]
    if not finite:
        return float("nan"), float("nan"), False
    gain, mse = min(finite, key=lambda item: item[1])
    return gain, mse, bool(gain == gains[0] or gain == gains[-1])


def _grid(model: SourceFirstDynamicModel, shape: tuple[int, int, int]) -> torch.Tensor:
    return SourceFirstDynamicModel._physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, shape)[0][0]


def _finite_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if not torch.isfinite(predicted).all():
        return {name: float("nan") for name in ("static_rms_mm", "dynamic_rms_mm", "gt_dynamic_rms_mm", "dynamic_rms_ratio", "dynamic_cosine", "endpoint_error_mm", "normalized_epe")}
    return _dvf_values(predicted, target)


def _oracle_metrics(predicted_alpha1: torch.Tensor, target: torch.Tensor) -> tuple[float, dict[str, float]]:
    alpha = alpha_dvf_least_squares(predicted_alpha1, target)
    if not math.isfinite(alpha):
        return alpha, {name: float("nan") for name in ("normalized_epe", "dynamic_rms_ratio", "dynamic_cosine")}
    values = _finite_metrics(predicted_alpha1 * alpha, target)
    return alpha, {name: values[name] for name in ("normalized_epe", "dynamic_rms_ratio", "dynamic_cosine")}


def _aggregate_oracle_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    names = ("alpha_dvf_full_volume", "alpha_dvf_pixel_center", "alpha_image_grid_best", "alpha_image_over_alpha_dvf_full", "alpha_image_over_alpha_dvf_center")
    result: list[dict[str, object]] = []
    for name in names:
        values = np.asarray([float(row[name]) for row in rows if math.isfinite(float(row[name]))], dtype=np.float64)
        if values.size:
            result.append({"scope": "overall", "metric": name, "n_locations": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)), "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90))})
    return result


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is non-empty: {path}; pass --overwrite to reuse it")
        for item in path.iterdir():
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    path.mkdir(parents=True, exist_ok=True)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    _prepare_output(args.output_dir, args.overwrite)
    audit = prepare_frozen_motion_audit(args)
    model, before = audit.model, parameter_snapshot(audit.model)
    gains = [float(value) for value in args.gains]
    grid = _grid(model, tuple(args.full_grid_shape))
    source_seed = int(audit.payload["summary"]["provenance"]["seed"])
    dvf_rows: list[dict[str, object]] = []; image_rows: list[dict[str, object]] = []; oracle_rows: list[dict[str, object]] = []
    for location, sequence in sorted(audit.groups.items()):
        view = sequence[0].view
        scores = torch.cat([model.film_encoder(item.image.unsqueeze(0), **_geometry_kwargs(item))["resp_scores"] for item in sequence])
        centered = gain_scores(scores, 1.0)
        supports = {"full_volume": [grid for _ in sequence], "pixel_center": [SourceFirstDynamicModel._pixel_world(item, full_image_uv(item).to(item.image)) for item in sequence]}
        centers = torch.stack(supports["pixel_center"])
        if not torch.allclose(centers, centers[0].expand_as(centers), atol=1e-4, rtol=0.0):
            raise ValueError(f"pixel centers differ across time for {location}")
        oracle: dict[str, tuple[float, dict[str, float]]] = {}
        for support, points_per_frame in supports.items():
            target = torch.stack([audit.motion.respiratory_dvf_mm(points[None], points.new_tensor([item.timestamp_s]))[0] for item, points in zip(sequence, points_per_frame)])
            predicted_by_gain: dict[float, torch.Tensor] = {}
            for alpha in gains:
                prediction = torch.stack([_respiratory_field(model, score, points) for score, points in zip(gain_scores(scores, alpha), points_per_frame)])
                predicted_by_gain[alpha] = prediction
                values = _finite_metrics(prediction, target)
                dvf_rows.append({"location": location, "view": view, "support": support, "alpha": alpha, "finite": bool(torch.isfinite(prediction).all()), **values})
            if 1.0 not in predicted_by_gain:
                predicted_alpha1 = torch.stack([_respiratory_field(model, score, points) for score, points in zip(centered, points_per_frame)])
            else:
                predicted_alpha1 = predicted_by_gain[1.0]
            oracle[support] = _oracle_metrics(predicted_alpha1, target)
        images_by_gain: dict[float, list[torch.Tensor]] = {gain: [] for gain in gains}
        target_images: list[torch.Tensor] = []
        for frame_index, item in enumerate(sequence):
            uv = full_image_uv(item)
            cache = build_target_psf_offset_cache(item, psf_samples=model.psf.n_samples, seed=source_seed + item.dynamic_frame_id)
            query = matched_psf_sample_world_points(model, item, uv, cache.for_pixel_uv(uv))
            target_images.append(rendered_frame_image(audit.model.canonical(query + audit.motion.respiratory_dvf_mm(query, query.new_tensor([item.timestamp_s]))).mean(-1), item.image.shape[-2:]))
            rendered = render_gain_images_from_cached_points(audit.model.canonical, model, query, {gain: gain_scores(scores, gain)[frame_index] for gain in gains}, item.image.shape[-2:])
            for gain, image in rendered.items():
                images_by_gain[gain].append(image)
        targets = torch.stack(target_images)
        stacked = {gain: torch.stack(images) for gain, images in images_by_gain.items()}
        mse_by_gain: dict[float, float] = {}
        alpha0 = stacked.get(0.0)
        alpha1 = stacked.get(1.0)
        for gain, images in stacked.items():
            finite = bool(torch.isfinite(images).all())
            if finite:
                values = _image_values(images, targets, alpha1 if alpha1 is not None else images)
                mse_by_gain[gain] = values["paired_psf_mse"]
                effect0 = float((images - alpha0).square().mean()) if alpha0 is not None else float("nan")
                effect1 = float((images - alpha1).square().mean()) if alpha1 is not None else float("nan")
                image_rows.append({"location": location, "view": view, "alpha": gain, "finite": True, "paired_psf_rmse": math.sqrt(values["paired_psf_mse"]), "image_effect_vs_alpha1_mse": effect1, "image_effect_vs_alpha0_mse": effect0, **values})
            else:
                image_rows.append({"location": location, "view": view, "alpha": gain, "finite": False, **{name: float("nan") for name in ("paired_psf_mse", "paired_psf_rmse", "temporal_dynamic_rms", "gt_temporal_dynamic_rms", "temporal_dynamic_rms_ratio", "consecutive_delta_cosine", "image_effect_vs_alpha1_mse", "image_effect_vs_alpha0_mse")}})
        alpha_image, image_mse, boundary = gain_grid_best(mse_by_gain, gains)
        alpha_full, full_values = oracle["full_volume"]
        alpha_center, center_values = oracle["pixel_center"]
        oracle_rows.append({"location": location, "view": view, "alpha_dvf_full_volume": alpha_full, "alpha_dvf_pixel_center": alpha_center, "nepe_at_alpha_dvf_full_volume": full_values["normalized_epe"], "nepe_at_alpha_dvf_pixel_center": center_values["normalized_epe"], "dynamic_rms_ratio_at_alpha_dvf_full_volume": full_values["dynamic_rms_ratio"], "dynamic_rms_ratio_at_alpha_dvf_pixel_center": center_values["dynamic_rms_ratio"], "dynamic_cosine_full_volume": full_values["dynamic_cosine"], "dynamic_cosine_pixel_center": center_values["dynamic_cosine"], "alpha_image_grid_best": alpha_image, "image_grid_best_mse": image_mse, "image_optimum_at_boundary": boundary, "alpha_image_over_alpha_dvf_full": float("nan") if not math.isfinite(alpha_image) or not math.isfinite(alpha_full) or abs(alpha_full) <= EPSILON else alpha_image / alpha_full, "alpha_image_over_alpha_dvf_center": float("nan") if not math.isfinite(alpha_image) or not math.isfinite(alpha_center) or abs(alpha_center) <= EPSILON else alpha_image / alpha_center})
    assert_parameters_unchanged(model, before)
    aggregate = _aggregate_oracle_rows(oracle_rows)
    _write_rows(args.output_dir / "gain_sweep_dvf_per_location.csv", dvf_rows)
    _write_rows(args.output_dir / "gain_sweep_image_per_location.csv", image_rows)
    _write_rows(args.output_dir / "gain_oracle_per_location.csv", oracle_rows)
    _write_rows(args.output_dir / "gain_sweep_aggregate.csv", aggregate)
    summary = {"experiment": "exp2c2_dynamic_score_gain_sweep", "checkpoint": str(args.checkpoint), "source_checkpoint_git_sha": audit.payload["summary"].get("provenance", {}).get("git_sha"), "diagnostic_git_sha": _git_sha(), "source_config": str(args.source_config), "experiment_config": str(args.experiment_config), "manifest": str(args.manifest), "qc_table": str(args.qc_table), "canonical_domain": str(args.canonical_domain), "source_recovery_arrays": str(audit.recovery_arrays), "timestamps_validated": audit.timestamps_validated, "n_locations": len(audit.groups), "n_frames": len(audit.observations), "gains": gains, "full_volume_grid_shape": list(args.full_grid_shape), "matched_crn": True, "optimizer_created": False, "parameters_unchanged": True, "respiratory_score_definition": "alpha * (score - per_location_temporal_mean)", "image_rendering": "respiratory-only analytic GT and frozen respiratory MBC render on identical per-frame cached matched-CRN PSF query points", "aggregate": aggregate}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2-C2 frozen dynamic respiratory score-gain sweep")
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs/source_first_change4_paperaligned.yaml"); parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml")
    parser.add_argument("--device", default="cpu"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50); parser.add_argument("--full-grid-shape", type=lambda value: tuple(int(item) for item in value.split(",")), default=(16, 16, 16)); parser.add_argument("--gains", type=lambda value: [float(item) for item in value.split(",")], default=list(DEFAULT_GAINS))
    args = parser.parse_args()
    if len(args.full_grid_shape) != 3 or min(args.full_grid_shape) < 2:
        parser.error("--full-grid-shape must be three comma-separated values >=2")
    if not args.gains or any(not math.isfinite(value) for value in args.gains):
        parser.error("--gains must be a non-empty comma-separated list of finite values")
    summary = run(args)
    print(f"Exp2-C2: {args.output_dir / 'summary.json'} ({summary['n_locations']} locations)")


if __name__ == "__main__":
    main()
