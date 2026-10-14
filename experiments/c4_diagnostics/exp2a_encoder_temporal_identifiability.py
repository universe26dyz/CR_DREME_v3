#!/usr/bin/env python3
"""Offline FiLM encoder temporal-identifiability audit for Exp2-A."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import torch
import yaml

from cardioresp4d.training.build_model import build_source_first_model
from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, _load_domain, geometry_from_manifest
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom


EPSILON = 1e-12
FEATURE_STAGES = (
    "image_feature_pre_film",
    "image_feature_pre_film_pooled",
    "geometry_encoding",
    "gamma",
    "beta",
    "image_feature_post_film",
    "image_feature_post_film_pooled",
    "head_feature",
    "resp_scores",
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _geometry_kwargs(observation) -> dict[str, torch.Tensor]:
    image = observation.image.unsqueeze(0)
    return {
        "center_mm": observation.center_mm.to(image)[None],
        "row_direction": observation.row_direction.to(image)[None],
        "column_direction": observation.column_direction.to(image)[None],
        "normal": observation.normal.to(image)[None],
        "pixel_spacing_mm": observation.pixel_spacing_mm.to(image)[None],
        "slice_thickness_mm": image.new_tensor([[observation.slice_thickness_mm]]),
    }


def extract_encoder_stages(encoder, image: torch.Tensor, *, center_mm: torch.Tensor, row_direction: torch.Tensor, column_direction: torch.Tensor, normal: torch.Tensor, pixel_spacing_mm: torch.Tensor, slice_thickness_mm: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reproduce the public GeometryFiLMMotionEncoder forward calculation exactly."""
    if image.ndim == 3:
        image = image[:, None]
    geometry = encoder.encode_geometry(center_mm, row_direction, column_direction, normal, pixel_spacing_mm, slice_thickness_mm)
    gamma, beta = encoder.geometry_mlp(geometry).chunk(2, -1)
    pre = encoder.image(image)
    post = encoder.film(pre, 1.0 + gamma, beta)
    head = encoder.head(post)
    return {
        "image_feature_pre_film": pre,
        "image_feature_pre_film_pooled": pre.mean(dim=(-2, -1)),
        "geometry_encoding": geometry,
        "gamma": gamma,
        "beta": beta,
        "image_feature_post_film": post,
        "image_feature_post_film_pooled": post.mean(dim=(-2, -1)),
        "head_feature": head,
        "resp_scores": encoder.resp(head).reshape(image.shape[0], 3, 3),
    }


def validate_constant_location_geometry(observations: list, *, tolerance: float = 1e-4) -> None:
    if not observations:
        raise ValueError("location has no observations")
    reference = observations[0]
    for observation in observations[1:]:
        if not (torch.allclose(observation.center_mm, reference.center_mm, atol=tolerance, rtol=0.0) and torch.allclose(observation.row_direction, reference.row_direction, atol=tolerance, rtol=0.0) and torch.allclose(observation.column_direction, reference.column_direction, atol=tolerance, rtol=0.0) and torch.allclose(observation.normal, reference.normal, atol=tolerance, rtol=0.0) and torch.allclose(observation.pixel_spacing_mm, reference.pixel_spacing_mm, atol=tolerance, rtol=0.0) and abs(observation.slice_thickness_mm - reference.slice_thickness_mm) <= tolerance):
            raise ValueError("geometry changes within a fixed view/slice location")


def _safe_correlation(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    target = reference - reference.mean()
    denominator = centered.square().sum(dim=0).sqrt() * target.square().sum().sqrt()
    correlation = (centered * target[:, None]).sum(dim=0) / denominator.clamp_min(EPSILON)
    return torch.where(denominator > EPSILON, correlation, torch.full_like(correlation, float("nan")))


def feature_temporal_metrics(features: torch.Tensor, gt_state: torch.Tensor) -> dict[str, float]:
    """Metrics for a [time, ...] stage feature and known respiratory state."""
    if features.ndim < 2 or features.shape[0] != gt_state.numel():
        raise ValueError("features must be [time, ...] and align with GT timestamps")
    flat = features.reshape(features.shape[0], -1)
    dynamic = flat - flat.mean(dim=0, keepdim=True)
    temporal_rms = float(dynamic.square().mean().sqrt())
    correlations = _safe_correlation(flat, gt_state)
    finite = correlations[torch.isfinite(correlations)]
    total_energy = dynamic.square().sum()
    if total_energy <= EPSILON or flat.shape[0] < 2:
        pca_fraction = float("nan"); pca_correlation = float("nan")
    else:
        _, singular_values, right = torch.linalg.svd(dynamic, full_matrices=False)
        scores = dynamic @ right[0]
        pca_fraction = float(singular_values[0].square() / total_energy)
        pca_correlation = float(_safe_correlation(scores[:, None], gt_state)[0].abs())
    return {
        "feature_dimension": float(flat.shape[1]),
        "temporal_mean_norm": float(flat.mean(dim=0).norm()),
        "temporal_rms": temporal_rms,
        "temporal_std": float(flat.std(dim=0, unbiased=False).mean()),
        "max_abs_gt_correlation": float(finite.abs().max()) if finite.numel() else float("nan"),
        "pc1_temporal_variance_fraction": pca_fraction,
        "pc1_abs_gt_correlation": pca_correlation,
    }


def static_temporal_metrics(location_mean_features: torch.Tensor, within_location_temporal_rms: torch.Tensor) -> dict[str, float]:
    if location_mean_features.ndim != 2 or within_location_temporal_rms.ndim != 1 or location_mean_features.shape[0] != within_location_temporal_rms.numel():
        raise ValueError("location means and within-location RMS values must align")
    between = float((location_mean_features - location_mean_features.mean(dim=0, keepdim=True)).square().mean().sqrt())
    median_within = float(torch.quantile(within_location_temporal_rms, 0.5))
    return {
        "between_location_mean_rms": between,
        "median_within_location_temporal_rms": median_within,
        "static_to_temporal_ratio": float("nan") if median_within <= EPSILON else between / median_within,
    }


def counterfactual_encoder_inputs(observation, temporal_mean_image: torch.Tensor, reference_geometry_observation) -> dict[str, dict[str, torch.Tensor]]:
    """Diagnostic image/geometry interventions; they are not physical acquisitions."""
    original_geometry = _geometry_kwargs(observation)
    reference_geometry = _geometry_kwargs(reference_geometry_observation)
    original_image = observation.image.unsqueeze(0)
    mean_image = temporal_mean_image.unsqueeze(0) if temporal_mean_image.ndim == 3 else temporal_mean_image
    return {
        "original": {"image": original_image, **original_geometry},
        "temporal_mean_image": {"image": mean_image.to(original_image), **original_geometry},
        "reference_geometry": {"image": original_image, **reference_geometry},
        "temporal_mean_image_reference_geometry": {"image": mean_image.to(original_image), **reference_geometry},
    }


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is non-empty: {path}; pass --overwrite to reuse it")
        for item in path.iterdir():
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _group_by_location(observations: list) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for observation in observations:
        grouped[f"{observation.view}/{observation.slice_id}"].append(observation)
    return {location: sorted(sequence, key=lambda item: item.timestamp_s) for location, sequence in grouped.items()}


def _location_stage_metrics(stages: dict[str, torch.Tensor], gt_state: torch.Tensor) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    metrics: dict[str, dict[str, float]] = {}
    arrays: dict[str, np.ndarray] = {}
    for name in FEATURE_STAGES:
        feature = stages[name]
        flat = feature.reshape(feature.shape[0], -1)
        metrics[name] = feature_temporal_metrics(feature, gt_state)
        arrays[f"{name}_temporal_mean"] = flat.mean(dim=0).cpu().numpy()
        arrays[f"{name}_gt_correlations"] = _safe_correlation(flat, gt_state).cpu().numpy()
    return metrics, arrays


def _counterfactual_metrics(scores: dict[str, torch.Tensor], gt_state: torch.Tensor) -> dict[str, float]:
    values = {name: feature_temporal_metrics(score, gt_state) for name, score in scores.items()}
    rms = {name: value["temporal_rms"] for name, value in values.items()}
    original, mean_image, reference, both = rms["original"], rms["temporal_mean_image"], rms["reference_geometry"], rms["temporal_mean_image_reference_geometry"]
    return {
        "original_resp_score_temporal_rms": original,
        "score_rms_difference_original_minus_temporal_mean_image": original - mean_image,
        "score_rms_difference_original_minus_reference_geometry": original - reference,
        "score_rms_difference_interaction": (original - mean_image) - (reference - both),
        "original_max_abs_gt_correlation": values["original"]["max_abs_gt_correlation"],
        "reference_geometry_max_abs_gt_correlation": values["reference_geometry"]["max_abs_gt_correlation"],
        "temporal_variance_retained_reference_geometry": float("nan") if original <= EPSILON else (reference / original) ** 2,
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    if args.device != "cpu":
        raise ValueError("Exp2-A is an offline CPU diagnostic; use --device cpu")
    _prepare_output(args.output_dir, args.overwrite)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" not in payload or "summary" not in payload:
        raise ValueError("checkpoint must be an Exp1 synthetic_recovery_last.pt payload")
    source_summary = payload["summary"]
    if source_summary.get("mode") != "motion_only" or source_summary.get("failure_signature") != "static_collapse":
        raise ValueError("Exp2-A requires the original Exp1 motion_only static_collapse checkpoint")
    with args.source_config.open(encoding="utf-8") as handle:
        source_config = yaml.safe_load(handle)
    with args.experiment_config.open(encoding="utf-8") as handle:
        synthetic = yaml.safe_load(handle)["synthetic"]
    domain = _load_domain(args.canonical_domain)
    motion = GroundTruthMotion.from_domain(domain, respiratory_frequency_hz=float(synthetic["respiratory_frequency_hz"]), cardiac_frequency_hz=float(synthetic["cardiac_frequency_hz"]), respiratory_amplitude_mm=float(synthetic["respiratory_amplitude_mm"]), cardiac_amplitude_mm=float(synthetic["cardiac_amplitude_mm"]))
    image_shape = tuple(int(value) for value in synthetic["image_shape"])
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=torch.device("cpu"))
    observations = _build_observations(frames, image_shape=image_shape, phantom=AnalyticCanonicalPhantom(domain), motion=motion, psf_samples=int(synthetic["psf_samples"]), seed=int(source_summary["provenance"]["seed"]))
    if len(observations) != int(source_summary["n_frames"]):
        raise ValueError("recreated frame count differs from the source synthetic motion-only run")
    model = build_source_first_model(source_config, domain, n_dynamic_frames=len(observations), device=torch.device("cpu"))
    model.load_state_dict(payload["model"]); model.eval()
    groups = _group_by_location(observations)
    if len(groups) != int(source_summary["n_locations"]):
        raise ValueError("recreated location count differs from the source synthetic motion-only run")
    references = {view: sorted((location for location in groups if location.split("/", 1)[0] == view))[0] for view in ("2CH", "4CH", "SAX")}
    feature_rows: list[dict[str, object]] = []; score_rows: list[dict[str, object]] = []; counter_rows: list[dict[str, object]] = []; arrays: dict[str, np.ndarray] = {}
    location_stage_means: dict[str, list[tuple[str, torch.Tensor, float]]] = defaultdict(list)
    for location, sequence in sorted(groups.items()):
        validate_constant_location_geometry(sequence)
        timestamps = torch.tensor([item.timestamp_s for item in sequence], dtype=torch.float32)
        gt_state = motion.respiratory_score(timestamps)
        stage_values: dict[str, list[torch.Tensor]] = {name: [] for name in FEATURE_STAGES}
        for observation in sequence:
            values = extract_encoder_stages(model.film_encoder, observation.image.unsqueeze(0), **_geometry_kwargs(observation))
            for name in FEATURE_STAGES:
                stage_values[name].append(values[name][0].detach().cpu())
        stacked = {name: torch.stack(values) for name, values in stage_values.items()}
        metrics, location_arrays = _location_stage_metrics(stacked, gt_state)
        view = sequence[0].view
        for stage, values in metrics.items():
            feature_rows.append({"location": location, "view": view, "stage": stage, "frame_count": len(sequence), **values})
            flat = stacked[stage].reshape(len(sequence), -1)
            location_stage_means[stage].append((view, flat.mean(dim=0), values["temporal_rms"]))
        scores = stacked["resp_scores"]
        score_dynamic = scores - scores.mean(dim=0, keepdim=True)
        score_correlations = _safe_correlation(scores.reshape(len(sequence), -1), gt_state).reshape(3, 3)
        for level in range(3):
            for axis in range(3):
                score_rows.append({"location": location, "view": view, "level": level, "axis": axis, "dynamic_rms": float(score_dynamic[:, level, axis].square().mean().sqrt()), "gt_correlation": float(score_correlations[level, axis]), "overall_flattened_dynamic_rms": metrics["resp_scores"]["temporal_rms"], "location_mean_score_norm": float(scores.mean(dim=0).norm())})
        arrays.update({f"{location.replace('/', '_')}_{name}": value for name, value in location_arrays.items()})
        temporal_mean_image = torch.stack([item.image for item in sequence]).mean(dim=0)
        reference = groups[references[view]][0]
        condition_scores: dict[str, list[torch.Tensor]] = {name: [] for name in ("original", "temporal_mean_image", "reference_geometry", "temporal_mean_image_reference_geometry")}
        for observation in sequence:
            for condition, values in counterfactual_encoder_inputs(observation, temporal_mean_image, reference).items():
                condition_scores[condition].append(extract_encoder_stages(model.film_encoder, **values)["resp_scores"][0].detach().cpu())
        counter_rows.append({"location": location, "view": view, "reference_location": references[view], "frame_count": len(sequence), **_counterfactual_metrics({name: torch.stack(values) for name, values in condition_scores.items()}, gt_state)})
    aggregate_rows: list[dict[str, object]] = []
    for stage, values in location_stage_means.items():
        for view in ("overall", "2CH", "4CH", "SAX"):
            selected = values if view == "overall" else [value for value in values if value[0] == view]
            means = torch.stack([value[1] for value in selected]); rms = torch.tensor([value[2] for value in selected])
            aggregate_rows.append({"stage": stage, "scope": view, "n_locations": len(selected), **static_temporal_metrics(means, rms)})
    _write_rows(args.output_dir / "feature_temporal_metrics_per_location.csv", feature_rows)
    _write_rows(args.output_dir / "feature_static_temporal_aggregate.csv", aggregate_rows)
    _write_rows(args.output_dir / "resp_score_metrics_per_location.csv", score_rows)
    _write_rows(args.output_dir / "score_counterfactuals_per_location.csv", counter_rows)
    np.savez_compressed(args.output_dir / "feature_arrays.npz", **arrays)
    summary = {"experiment": "exp2a_encoder_temporal_identifiability", "checkpoint": str(args.checkpoint), "checkpoint_provenance": source_summary.get("provenance", {}), "diagnostic_git_sha": _git_sha(), "source_config": str(args.source_config), "experiment_config": str(args.experiment_config), "manifest": str(args.manifest), "qc_table": str(args.qc_table), "canonical_domain": str(args.canonical_domain), "n_locations": len(groups), "n_frames": len(observations), "reference_location_by_view": references, "gt_respiratory_frequency_hz": motion.respiratory_frequency_hz, "counterfactual_interpretation": "image/geometry mismatches are diagnostic sensitivity interventions, not physical acquisitions"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2-A FiLM encoder temporal identifiability audit")
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs/source_first_change4_paperaligned.yaml")
    parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml")
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--device", default="cpu")
    parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50)
    args = parser.parse_args(); summary = run(args)
    print(f"Exp2-A: {args.output_dir / 'summary.json'} ({summary['n_locations']} locations)")


if __name__ == "__main__":
    main()
