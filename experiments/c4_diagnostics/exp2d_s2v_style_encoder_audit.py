#!/usr/bin/env python3
"""Exp2-D1: frozen audit for the S2V-style single-frame score encoder."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import torch

from cardioresp4d.training.model import SourceFirstDynamicModel
from experiments.c4_diagnostics.exp2_checkpoint_common import assert_parameters_unchanged, full_image_uv, matched_psf_sample_world_points, parameter_snapshot, prepare_frozen_motion_audit
from experiments.c4_diagnostics.exp2c_posthoc_score_centering import EPSILON, _dvf_values, _image_values, _respiratory_field, _score_values, _write_rows, rendered_frame_image, render_resp_scores_from_cached_points
from experiments.c4_diagnostics.exp2c2_dynamic_score_gain_sweep import DEFAULT_GAINS, _aggregate_oracle_rows, _grid, _oracle_metrics, gain_grid_best, gain_scores, render_gain_images_from_cached_points
from experiments.c4_diagnostics.synthetic.metrics import aggregate_location_metrics
from experiments.c4_diagnostics.synthetic.oracle import build_target_psf_offset_cache
from experiments.c4_diagnostics.synthetic.prior import build_synthetic_frequency_prior


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is non-empty: {path}; pass --overwrite to reuse it")
        for item in path.iterdir():
            if item.is_dir():
                import shutil
                shutil.rmtree(item)
            else:
                item.unlink()
    path.mkdir(parents=True, exist_ok=True)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    _prepare_output(args.output_dir, args.overwrite)
    audit = prepare_frozen_motion_audit(args, require_static_collapse=False)
    model, before = audit.model, parameter_snapshot(audit.model)
    if model.film_implementation != "s2v_style_single_frame":
        raise ValueError("Exp2-D1 requires an s2v_style_single_frame checkpoint")
    if any(item.normalized_slice_position is None for item in audit.observations):
        raise ValueError("Exp2-D1 observations require assigned normalized_slice_position")
    gains = [float(value) for value in args.gains]
    prior = build_synthetic_frequency_prior(audit.motion.respiratory_frequency_hz, audit.motion.cardiac_frequency_hz, {(item.view, item.slice_id) for item in audit.observations})
    grid = _grid(model, tuple(args.full_grid_shape))
    source_seed = int(audit.payload["summary"]["provenance"]["seed"])
    score_rows: list[dict[str, object]] = []; dvf_rows: list[dict[str, object]] = []; oracle_rows: list[dict[str, object]] = []; image_rows: list[dict[str, object]] = []
    aggregate_input: dict[str, dict[str, float]] = {}
    for location, sequence in sorted(audit.groups.items()):
        view = sequence[0].view
        scores = torch.cat([model.encode_motion_scores(item)["resp_scores"] for item in sequence])
        timestamps = scores.new_tensor([item.timestamp_s for item in sequence], dtype=torch.float64)
        gt_state = audit.motion.respiratory_score(timestamps.to(torch.float32))
        score_values = _score_values(scores, timestamps, gt_state, prior.for_location(view, sequence[0].slice_id).respiratory_bands_hz)
        score_values["score_static_to_temporal_ratio"] = float("nan") if score_values["score_temporal_rms"] <= EPSILON else score_values["score_temporal_mean_norm"] / score_values["score_temporal_rms"]
        score_rows.append({"location": location, "view": view, "frame_count": len(sequence), "normalized_slice_position": sequence[0].normalized_slice_position, **score_values})
        supports = {"full_volume": [grid for _ in sequence], "pixel_center": [SourceFirstDynamicModel._pixel_world(item, full_image_uv(item).to(item.image)) for item in sequence]}
        centers = torch.stack(supports["pixel_center"])
        if not torch.allclose(centers, centers[0].expand_as(centers), atol=1e-4, rtol=0.0):
            raise ValueError(f"pixel centers differ across time for {location}")
        oracle: dict[str, tuple[float, dict[str, float]]] = {}
        centered = gain_scores(scores, 1.0)
        for support, points_per_frame in supports.items():
            target = torch.stack([audit.motion.respiratory_dvf_mm(points[None], points.new_tensor([item.timestamp_s]))[0] for item, points in zip(sequence, points_per_frame)])
            raw_prediction = torch.stack([_respiratory_field(model, score, points) for score, points in zip(scores, points_per_frame)])
            dvf_rows.append({"location": location, "view": view, "support": support, "condition": "original_scores", **_dvf_values(raw_prediction, target)})
            alpha1_prediction = torch.stack([_respiratory_field(model, score, points) for score, points in zip(centered, points_per_frame)])
            oracle[support] = _oracle_metrics(alpha1_prediction, target)
        images_by_gain: dict[float, list[torch.Tensor]] = {gain: [] for gain in gains}; target_images: list[torch.Tensor] = []
        for frame_index, item in enumerate(sequence):
            uv = full_image_uv(item)
            cache = build_target_psf_offset_cache(item, psf_samples=model.psf.n_samples, seed=source_seed + item.dynamic_frame_id)
            query = matched_psf_sample_world_points(model, item, uv, cache.for_pixel_uv(uv))
            target_images.append(rendered_frame_image(model.canonical(query + audit.motion.respiratory_dvf_mm(query, query.new_tensor([item.timestamp_s]))).mean(-1), item.image.shape[-2:]))
            rendered = render_gain_images_from_cached_points(model.canonical, model, query, {gain: gain_scores(scores, gain)[frame_index] for gain in gains}, item.image.shape[-2:])
            for gain, image in rendered.items():
                images_by_gain[gain].append(image)
        targets = torch.stack(target_images); stacked = {gain: torch.stack(images) for gain, images in images_by_gain.items()}
        mse_by_gain: dict[float, float] = {}
        alpha0, alpha1 = stacked.get(0.0), stacked.get(1.0)
        for gain, images in stacked.items():
            values = _image_values(images, targets, alpha1 if alpha1 is not None else images)
            mse_by_gain[gain] = values["paired_psf_mse"]
            image_rows.append({"location": location, "view": view, "alpha": gain, "paired_psf_rmse": math.sqrt(values["paired_psf_mse"]), "image_effect_vs_alpha0_mse": float((images - alpha0).square().mean()) if alpha0 is not None else float("nan"), **values})
        alpha_image, image_mse, boundary = gain_grid_best(mse_by_gain, gains)
        alpha_full, full = oracle["full_volume"]; alpha_center, center = oracle["pixel_center"]
        oracle_row = {"location": location, "view": view, "alpha_dvf_full_volume": alpha_full, "alpha_dvf_pixel_center": alpha_center, "dynamic_endpoint_error_mm_at_alpha_dvf_full_volume": full["dynamic_endpoint_error_mm"], "dynamic_endpoint_error_mm_at_alpha_dvf_pixel_center": center["dynamic_endpoint_error_mm"], "dynamic_normalized_epe_at_alpha_dvf_full_volume": full["dynamic_normalized_epe"], "dynamic_normalized_epe_at_alpha_dvf_pixel_center": center["dynamic_normalized_epe"], "dynamic_cosine_at_alpha_dvf_full_volume": full["dynamic_cosine"], "dynamic_cosine_at_alpha_dvf_pixel_center": center["dynamic_cosine"], "dynamic_rms_ratio_at_alpha_dvf_full_volume": full["dynamic_rms_ratio"], "dynamic_rms_ratio_at_alpha_dvf_pixel_center": center["dynamic_rms_ratio"], "alpha_image_grid_best": alpha_image, "image_grid_best_mse": image_mse, "image_optimum_at_boundary": boundary, "alpha_image_over_alpha_dvf_full": float("nan") if not math.isfinite(alpha_image) or not math.isfinite(alpha_full) or abs(alpha_full) <= EPSILON else alpha_image / alpha_full, "alpha_image_over_alpha_dvf_center": float("nan") if not math.isfinite(alpha_image) or not math.isfinite(alpha_center) or abs(alpha_center) <= EPSILON else alpha_image / alpha_center}
        oracle_rows.append(oracle_row)
        aggregate_input[location] = {**score_values, **{key: float(value) for key, value in oracle_row.items() if isinstance(value, (int, float))}}
    assert_parameters_unchanged(model, before)
    aggregate = aggregate_location_metrics(aggregate_input)
    _write_rows(args.output_dir / "score_per_location.csv", score_rows)
    _write_rows(args.output_dir / "dvf_per_location.csv", dvf_rows)
    _write_rows(args.output_dir / "gain_oracle_per_location.csv", oracle_rows)
    _write_rows(args.output_dir / "gain_sweep_image_per_location.csv", image_rows)
    _write_rows(args.output_dir / "aggregate.csv", [{"metric": metric, **values} for metric, values in aggregate.items()])
    summary = {"experiment": "exp2d_s2v_style_encoder_audit", "encoder_implementation": model.film_implementation, "encoder_architecture": {"position_bands": model.film_encoder.position_bands, "view_embedding_dim": model.film_encoder.view_embedding.embedding_dim, "condition_dim": model.film_encoder.condition_mlp[0].out_features, "block_channels": [block.out_channels for block in model.film_encoder.blocks]}, "conditioning": "normalized stack-relative physical slice position Fourier features plus learnable view embedding; no center, DICOM orientation/spacing, timestamp, or frame index", "timestamp_is_not_encoder_input": True, "view_mapping": {"SAX": 0, "2CH": 1, "4CH": 2}, "slice_position_definition": "q=dot(center_mm, sign-aligned mean view normal); s=(q-q_mid)/q_half, or 0 for negligible span", "checkpoint": str(args.checkpoint), "source_checkpoint_git_sha": audit.payload["summary"].get("provenance", {}).get("git_sha"), "diagnostic_git_sha": _git_sha(), "source_config": str(args.source_config), "experiment_config": str(args.experiment_config), "manifest": str(args.manifest), "qc_table": str(args.qc_table), "canonical_domain": str(args.canonical_domain), "source_recovery_arrays": str(audit.recovery_arrays), "timestamps_validated": audit.timestamps_validated, "n_locations": len(audit.groups), "n_frames": len(audit.observations), "parameters_unchanged": True, "optimizer_created": False, "matched_crn": True, "gains": gains, "aggregate": aggregate, "gain_oracle_aggregate": _aggregate_oracle_rows(oracle_rows)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2-D1 frozen S2V-style encoder audit")
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs/source_first_change4_s2vstyle.yaml"); parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml")
    parser.add_argument("--device", default="cpu"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50); parser.add_argument("--full-grid-shape", type=lambda value: tuple(int(item) for item in value.split(",")), default=(16, 16, 16)); parser.add_argument("--gains", type=lambda value: [float(item) for item in value.split(",")], default=list(DEFAULT_GAINS))
    args = parser.parse_args()
    if len(args.full_grid_shape) != 3 or min(args.full_grid_shape) < 2:
        parser.error("--full-grid-shape must be three comma-separated values >=2")
    if not args.gains or any(not math.isfinite(value) for value in args.gains):
        parser.error("--gains must be a non-empty comma-separated list of finite values")
    summary = run(args)
    print(f"Exp2-D1: {args.output_dir / 'summary.json'} ({summary['n_locations']} locations)")


if __name__ == "__main__":
    main()
