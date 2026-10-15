#!/usr/bin/env python3
"""Exp2-C: frozen-checkpoint respiratory-score centering counterfactual."""
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

import torch

from cardioresp4d.losses.frequency_loss import non_dc_frequency_grid, nonuniform_dft_at_frequencies
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField
from cardioresp4d.training.model import SourceFirstDynamicModel
from experiments.c4_diagnostics.exp2_checkpoint_common import (
    assert_parameters_unchanged,
    full_image_uv,
    matched_psf_sample_world_points,
    parameter_snapshot,
    prepare_frozen_motion_audit,
)
from experiments.c4_diagnostics.synthetic.metrics import dvf_metrics, location_spectral_metrics, temporal_dynamics_metrics
from experiments.c4_diagnostics.synthetic.oracle import build_target_psf_offset_cache
from experiments.c4_diagnostics.synthetic.prior import build_synthetic_frequency_prior


EPSILON = 1e-12


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _geometry_kwargs(observation) -> dict[str, torch.Tensor]:
    image = observation.image.unsqueeze(0)
    return {"center_mm": observation.center_mm.to(image)[None], "row_direction": observation.row_direction.to(image)[None], "column_direction": observation.column_direction.to(image)[None], "normal": observation.normal.to(image)[None], "pixel_spacing_mm": observation.pixel_spacing_mm.to(image)[None], "slice_thickness_mm": image.new_tensor([[observation.slice_thickness_mm]])}


def centered_scores(scores: torch.Tensor, mean: torch.Tensor | None = None) -> torch.Tensor:
    if scores.ndim != 3 or scores.shape[-1] != 3:
        raise ValueError("respiratory scores must be [time, levels, 3]")
    baseline = scores.mean(dim=0, keepdim=True) if mean is None else mean.reshape(1, *scores.shape[1:]).to(scores)
    return scores - baseline


def temporal_rms(scores: torch.Tensor) -> float:
    return float((scores - scores.mean(dim=0, keepdim=True)).square().mean().sqrt())


def centered_spectrum(scores: torch.Tensor, timestamps_s: torch.Tensor) -> torch.Tensor:
    frequencies = non_dc_frequency_grid(timestamps_s).to(scores)
    return nonuniform_dft_at_frequencies(scores.reshape(scores.shape[0], -1), timestamps_s, frequencies)


def _safe_correlation(values: torch.Tensor, reference: torch.Tensor) -> float:
    flat = values.reshape(values.shape[0], -1)
    dynamic = flat - flat.mean(dim=0, keepdim=True)
    target = reference - reference.mean()
    denominator = dynamic.square().sum(dim=0).sqrt() * target.square().sum().sqrt()
    correlation = (dynamic * target[:, None]).sum(0) / denominator.clamp_min(EPSILON)
    finite = correlation[denominator > EPSILON]
    return float(finite.abs().max()) if finite.numel() else float("nan")


def _respiratory_field(model: SourceFirstDynamicModel, score: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return ScoreWeightedMBCField(model.respiratory_mbc, score[None])(points[None])[0]


def render_resp_scores_from_cached_points(canonical, query_points: torch.Tensor, model: SourceFirstDynamicModel, scores: torch.Tensor) -> torch.Tensor:
    """Respiratory-only PSF-integrated image using caller-owned cached query points."""
    dvf = _respiratory_field(model, scores, query_points)
    intensity = canonical(query_points + dvf)
    return intensity.mean(-1)


def _dvf_values(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    values = dvf_metrics(predicted, target)
    values["static_to_dynamic_ratio"] = float("nan") if values["dynamic_rms_mm"] <= EPSILON else values["static_rms_mm"] / values["dynamic_rms_mm"]
    values["normalized_epe"] = values["endpoint_error_mm"] / max(values["gt_dynamic_rms_mm"], EPSILON)
    return values


def _score_values(scores: torch.Tensor, timestamps: torch.Tensor, gt_state: torch.Tensor, respiratory_bands: list[tuple[float, float]]) -> dict[str, float]:
    return {"score_temporal_mean_norm": float(scores.mean(dim=0).norm()), "score_temporal_rms": temporal_rms(scores), "score_max_abs_gt_correlation": _safe_correlation(scores, gt_state), **location_spectral_metrics(scores, timestamps, respiratory_bands)}


def _image_values(images: torch.Tensor, target: torch.Tensor, original: torch.Tensor) -> dict[str, float]:
    values = temporal_dynamics_metrics(images, target)
    dynamic = images - images.mean(dim=0, keepdim=True)
    target_dynamic = target - target.mean(dim=0, keepdim=True)
    dynamic_rms = float(dynamic.square().mean().sqrt())
    target_dynamic_rms = float(target_dynamic.square().mean().sqrt())
    values.update({"paired_psf_mse": float((images - target).square().mean()), "image_effect_vs_original_mse": float((images - original).square().mean()), "image_effect_vs_original_rmse": float((images - original).square().mean().sqrt()), "temporal_dynamic_rms": dynamic_rms, "gt_temporal_dynamic_rms": target_dynamic_rms, "temporal_dynamic_rms_ratio": float("nan") if target_dynamic_rms <= EPSILON else dynamic_rms / target_dynamic_rms, "temporal_image_std": float(images.std(unbiased=False))})
    return values


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is non-empty: {path}; pass --overwrite to reuse it")
        for item in path.iterdir():
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _grid(model: SourceFirstDynamicModel, shape: tuple[int, int, int]) -> torch.Tensor:
    return SourceFirstDynamicModel._physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, shape)[0][0]


def run(args: argparse.Namespace) -> dict[str, object]:
    _prepare_output(args.output_dir, args.overwrite)
    audit = prepare_frozen_motion_audit(args)
    model, before = audit.model, parameter_snapshot(audit.model)
    prior = build_synthetic_frequency_prior(audit.motion.respiratory_frequency_hz, audit.motion.cardiac_frequency_hz, {(item.view, item.slice_id) for item in audit.observations})
    with torch.no_grad():
        all_scores = {location: torch.cat([model.film_encoder(item.image.unsqueeze(0), **_geometry_kwargs(item))["resp_scores"] for item in sequence]) for location, sequence in audit.groups.items()}
    global_mean = torch.cat(list(all_scores.values())).mean(dim=0)
    source_seed = int(audit.payload["summary"]["provenance"]["seed"])
    grid = _grid(model, tuple(args.full_grid_shape))
    score_rows: list[dict[str, object]] = []; dvf_rows: list[dict[str, object]] = []; image_rows: list[dict[str, object]] = []
    for location, sequence in sorted(audit.groups.items()):
        view = sequence[0].view
        timestamps = torch.tensor([item.timestamp_s for item in sequence], dtype=torch.float64)
        gt_state = audit.motion.respiratory_score(timestamps.to(torch.float32))
        conditions = {"A_original": all_scores[location], "B_per_location_centered": centered_scores(all_scores[location]), "C_global_centered": centered_scores(all_scores[location], global_mean)}
        for condition, scores in conditions.items():
            if condition == "B_per_location_centered" and not torch.allclose(scores.mean(dim=0), torch.zeros_like(scores[0]), atol=1e-6, rtol=0.0):
                raise RuntimeError("per-location score centering failed")
            score_rows.append({"location": location, "view": view, "condition": condition, "frame_count": len(sequence), **_score_values(scores, timestamps, gt_state, prior.for_location(view, sequence[0].slice_id).respiratory_bands_hz)})
        supports: dict[str, list[torch.Tensor]] = {"full_volume": [grid for _ in sequence], "pixel_center": [SourceFirstDynamicModel._pixel_world(item, full_image_uv(item).to(item.image)) for item in sequence]}
        if not torch.allclose(torch.stack(supports["pixel_center"]), supports["pixel_center"][0].expand(len(sequence), -1, -1), atol=1e-4, rtol=0.0):
            raise ValueError(f"pixel centers differ across time for {location}")
        for support, frames in supports.items():
            target = torch.stack([audit.motion.respiratory_dvf_mm(points[None], points.new_tensor([item.timestamp_s]))[0] for item, points in zip(sequence, frames)])
            for condition, scores in conditions.items():
                predicted = torch.stack([_respiratory_field(model, score, points) for score, points in zip(scores, frames)])
                dvf_rows.append({"location": location, "view": view, "support": support, "condition": condition, "point_count": int(frames[0].shape[0]), **_dvf_values(predicted, target)})
        rendered: dict[str, list[torch.Tensor]] = {name: [] for name in conditions}; target_images: list[torch.Tensor] = []
        for item, score in zip(sequence, conditions["A_original"]):
            uv = full_image_uv(item)
            cache = build_target_psf_offset_cache(item, psf_samples=model.psf.n_samples, seed=source_seed + item.dynamic_frame_id)
            query = matched_psf_sample_world_points(model, item, uv, cache.for_pixel_uv(uv))
            gt_dvf = audit.motion.respiratory_dvf_mm(query, query.new_tensor([item.timestamp_s]))
            target_images.append(audit.model.canonical(query + gt_dvf).mean(-1).reshape(item.image.shape[-2:]))
            for condition, scores in conditions.items():
                rendered[condition].append(render_resp_scores_from_cached_points(audit.model.canonical, query, model, scores[len(rendered[condition])]))
        stacked = {condition: torch.stack(images) for condition, images in rendered.items()}
        target_images_tensor = torch.stack(target_images)
        for condition, images in stacked.items():
            image_rows.append({"location": location, "view": view, "condition": condition, "frame_count": len(sequence), "psf_samples": model.psf.n_samples, **_image_values(images, target_images_tensor, stacked["A_original"])})
    assert_parameters_unchanged(model, before)
    _write_rows(args.output_dir / "score_centering_per_location.csv", score_rows)
    _write_rows(args.output_dir / "dvf_centering_per_location.csv", dvf_rows)
    _write_rows(args.output_dir / "image_centering_per_location.csv", image_rows)
    summary = {"experiment": "exp2c_posthoc_score_centering", "checkpoint": str(args.checkpoint), "source_checkpoint_git_sha": audit.payload["summary"].get("provenance", {}).get("git_sha"), "diagnostic_git_sha": _git_sha(), "source_config": str(args.source_config), "experiment_config": str(args.experiment_config), "manifest": str(args.manifest), "qc_table": str(args.qc_table), "canonical_domain": str(args.canonical_domain), "source_recovery_arrays": str(audit.recovery_arrays), "timestamps_validated": audit.timestamps_validated, "n_locations": len(audit.groups), "n_frames": len(audit.observations), "conditions": {"A": "original frozen respiratory scores", "B": "per-fixed-location temporal score centering", "C": "global score centering"}, "matched_crn": True, "rendering": "respiratory-only analytic target and frozen respiratory field use identical per-frame cached PSF query points", "optimizer_created": False, "parameters_unchanged": True, "online_inference_warning": "Per-location centering uses the whole local temporal sequence and is not an online inference method."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2-C frozen score-centering counterfactual")
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs/source_first_change4_paperaligned.yaml"); parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml")
    parser.add_argument("--device", default="cpu"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50); parser.add_argument("--full-grid-shape", type=lambda value: tuple(int(item) for item in value.split(",")), default=(16, 16, 16))
    args = parser.parse_args()
    if len(args.full_grid_shape) != 3 or min(args.full_grid_shape) < 2:
        parser.error("--full-grid-shape must be three comma-separated values >=2")
    summary = run(args)
    print(f"Exp2-C: {args.output_dir / 'summary.json'} ({summary['n_locations']} locations)")


if __name__ == "__main__":
    main()
