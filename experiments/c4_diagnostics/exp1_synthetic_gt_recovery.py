#!/usr/bin/env python3
"""Experiment 1: isolated synthetic ground-truth recovery diagnostic for C4."""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

from cardioresp4d.geometry.world_geometry import DicomPlane
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField
from cardioresp4d.training.build_model import build_source_first_model
from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler
from cardioresp4d.training.stage_contract import StageSegment, paperaligned_segments
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer

from experiments.c4_diagnostics.synthetic.metrics import aggregate_location_metrics, dvf_metrics, location_spectral_metrics, temporal_dynamics_metrics
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from experiments.c4_diagnostics.synthetic.prior import build_synthetic_frequency_prior
from experiments.c4_diagnostics.synthetic.render import GeometryFrame, PhantomCanonicalAdapter, pixel_centers, render_synthetic_observation


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _frame(view: str, slice_id: str, center: torch.Tensor, timestamp_s: float, *, device: torch.device, rows: int, columns: int) -> GeometryFrame:
    axes = {"SAX": ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), "2CH": ((1., 0., 0.), (0., 0., 1.), (0., -1., 0.)), "4CH": ((0., 1., 0.), (0., 0., 1.), (1., 0., 0.))}[view]
    return GeometryFrame(view, slice_id, center.to(device), torch.tensor(axes[0], device=device), torch.tensor(axes[1], device=device), torch.tensor(axes[2], device=device), torch.tensor((1.5, 1.5), device=device), 6.0, timestamp_s, rows, columns)


def fallback_geometry(*, locations_per_view: int, frames_per_location: int, image_shape: tuple[int, int], motion: GroundTruthMotion, device: torch.device) -> list[GeometryFrame]:
    frames: list[GeometryFrame] = []; cardiac_center = torch.tensor(motion.cardiac_center_mm, device=device)
    for view_index, view in enumerate(("SAX", "2CH", "4CH")):
        for location in range(locations_per_view):
            offset = torch.tensor((float((location - 1) * 6), float((view_index - 1) * 4), float((location - 1) * 5)), device=device)
            for frame_index in range(frames_per_location): frames.append(_frame(view, f"synthetic_{view}_{location:02d}", cardiac_center + offset, frame_index * .16, device=device, rows=image_shape[0], columns=image_shape[1]))
    return frames


def geometry_from_manifest(manifest: Path, qc_table: Path, *, locations_per_view: int, frames_per_location: int, device: torch.device) -> list[GeometryFrame]:
    with manifest.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    with qc_table.open(newline="", encoding="utf-8") as handle: qc_rows = list(csv.DictReader(handle))
    qc = {(row.get("view", "").upper(), row.get("slice_id", ""), row.get("frame_index", "")): str(row.get("qc_valid", "1")).lower() not in {"0", "false"} for row in qc_rows}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        view = row.get("view", "").upper()
        if view in {"SAX", "2CH", "4CH"} and qc.get((view, row.get("slice_id", ""), row.get("frame_index", "")), True): grouped.setdefault((view, row["slice_id"]), []).append(row)
    frames: list[GeometryFrame] = []
    for view in ("SAX", "2CH", "4CH"):
        locations = sorted(key for key in grouped if key[0] == view)[:locations_per_view]
        if not locations: raise ValueError(f"manifest has no QC-valid {view} geometry")
        for _, slice_id in locations:
            selected = sorted(grouped[(view, slice_id)], key=lambda row: float(row["timestamp_s"]))
            for index in np.linspace(0, len(selected) - 1, min(frames_per_location, len(selected))).round().astype(int):
                row = selected[int(index)]; plane = DicomPlane.from_geometry(row)
                frames.append(GeometryFrame(view, slice_id, torch.from_numpy(plane.pixel_to_world((plane.columns - 1) / 2., (plane.rows - 1) / 2.)).float().to(device), torch.from_numpy(plane.row_direction).float().to(device), torch.from_numpy(plane.column_direction).float().to(device), torch.from_numpy(plane.normal).float().to(device), torch.from_numpy(plane.pixel_spacing).float().to(device), float(plane.slice_thickness or row["slice_thickness"]), float(row["timestamp_s"]), plane.rows, plane.columns))
    return frames


def _load_domain(path: Path | None) -> dict[str, object]:
    if path is not None:
        with path.open(encoding="utf-8") as handle: return json.load(handle)
    return {"world_min_mm": [-64., -64., -64.], "world_max_mm": [64., 64., 64.], "cardiac_box": {"min_mm": [-24., -24., -28.], "max_mm": [32., 24., 30.]}}


def _build_observations(frames: list[GeometryFrame], *, image_shape: tuple[int, int], phantom: AnalyticCanonicalPhantom, motion: GroundTruthMotion, psf_samples: int, seed: int) -> list[DynamicObservation]:
    return [render_synthetic_observation(frame, height=image_shape[0], width=image_shape[1], phantom=phantom, motion=motion, psf_samples=psf_samples, seed=seed + index, dynamic_frame_id=index) for index, frame in enumerate(frames)]


def preflight_observations(observations: list[DynamicObservation], motion: GroundTruthMotion) -> dict[str, float | bool]:
    contrast = []; cardiac_views = []
    for observation in observations:
        contrast.append(float(observation.image.std(unbiased=False)))
        if observation.view in {"2CH", "4CH"}:
            height, width = observation.image.shape[-2:]
            frame = GeometryFrame(observation.view, observation.slice_id, observation.center_mm, observation.row_direction, observation.column_direction, observation.normal, observation.pixel_spacing_mm, observation.slice_thickness_mm, observation.timestamp_s, height, width)
            cardiac_views.append(float((motion.cardiac_support(pixel_centers(frame, height, width)) > 1e-4).float().mean()))
    intersection = float(np.mean([value > 0.0 for value in cardiac_views])) if cardiac_views else 0.0
    minimum = min(contrast) if contrast else 0.0
    return {"minimum_slice_contrast_std": minimum, "mean_slice_contrast_std": float(np.mean(contrast)) if contrast else 0.0, "cardiac_view_intersection_fraction": intersection, "valid": bool(minimum > 1e-4 and intersection == 1.0)}


def build_experiment_schedule(mode: str, *, smoke: bool) -> tuple[StageSegment, ...]:
    all_segments = paperaligned_segments()
    if mode == "full_joint": selected = all_segments
    elif mode == "motion_only": selected = tuple(replace(segment, train_canonical=False) for segment in all_segments if segment.stage != "stage1")
    else: raise ValueError("unsupported experiment mode")
    return tuple(replace(segment, steps=1) for segment in selected) if smoke else selected


def training_observations_per_update(config: dict[str, object], *, smoke: bool) -> int:
    return 3 if smoke else int(config["training"]["observations_per_update"])


def _geometry_kwargs(observation: DynamicObservation) -> dict[str, torch.Tensor]:
    image = observation.image.unsqueeze(0)
    return {"center_mm": observation.center_mm[None], "row_direction": observation.row_direction[None], "column_direction": observation.column_direction[None], "normal": observation.normal[None], "pixel_spacing_mm": observation.pixel_spacing_mm[None], "slice_thickness_mm": torch.tensor([[observation.slice_thickness_mm]], device=image.device, dtype=image.dtype)}


def _grid(lower: torch.Tensor, upper: torch.Tensor, points_per_axis: int = 5) -> torch.Tensor:
    return torch.stack(torch.meshgrid(*(torch.linspace(lower[index], upper[index], points_per_axis, device=lower.device) for index in range(3)), indexing="ij"), dim=-1).reshape(1, -1, 3)


def _per_location_metrics(model, observations: list[DynamicObservation], motion: GroundTruthMotion, domain: dict[str, object], prior) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    grouped: dict[str, list[DynamicObservation]] = {}
    for observation in observations: grouped.setdefault(f"{observation.view}/{observation.slice_id}", []).append(observation)
    full_grid = _grid(torch.tensor(domain["world_min_mm"], device=observations[0].image.device), torch.tensor(domain["world_max_mm"], device=observations[0].image.device))
    cardiac = domain["cardiac_box"]; cardiac_grid = _grid(torch.tensor(cardiac["min_mm"], device=full_grid.device), torch.tensor(cardiac["max_mm"], device=full_grid.device))
    model.respiratory_mbc.set_active_levels(3); result: dict[str, dict[str, float]] = {}; arrays: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for key, sequence in grouped.items():
            sequence = sorted(sequence, key=lambda item: item.timestamp_s)
            resp_scores = []; card_scores = []; resp_pred = []; card_pred = []; resp_gt = []; card_gt = []; images = []; targets = []; timestamps = []
            for observation in sequence:
                encoded = model.film_encoder(observation.image.unsqueeze(0), **_geometry_kwargs(observation)); resp_field = ScoreWeightedMBCField(model.respiratory_mbc, encoded["resp_scores"]); card_field = ScoreWeightedMBCField(model.cardiac_mbc, encoded["card_scores"]); timestamp = full_grid.new_tensor([observation.timestamp_s])
                resp_scores.append(encoded["resp_scores"].reshape(-1)); card_scores.append(encoded["card_scores"].reshape(-1)); resp_pred.append(resp_field(full_grid)); card_pred.append(card_field(cardiac_grid)); resp_gt.append(motion.respiratory_dvf_mm(full_grid, timestamp)); card_gt.append(motion.cardiac_dvf_mm(cardiac_grid, timestamp)); timestamps.append(observation.timestamp_s)
                height, width = observation.image.shape[-2:]; uv = torch.stack(torch.meshgrid(torch.arange(width, dtype=observation.image.dtype), torch.arange(height, dtype=observation.image.dtype), indexing="xy"), dim=-1).reshape(-1, 2)
                images.append(model.predict(observation, uv, "stage3b")["predicted_intensity"].reshape(height, width)); targets.append(observation.image[0])
            ts = torch.tensor(timestamps, dtype=torch.float64, device=full_grid.device); local_prior = prior.for_location(sequence[0].view, sequence[0].slice_id)
            values = {**{f"resp_{name}": value for name, value in dvf_metrics(torch.cat(resp_pred), torch.cat(resp_gt)).items()}, **{f"card_{name}": value for name, value in dvf_metrics(torch.cat(card_pred), torch.cat(card_gt)).items()}, **{f"image_{name}": value for name, value in temporal_dynamics_metrics(torch.stack(images), torch.stack(targets)).items()}, **{f"resp_{name}": value for name, value in location_spectral_metrics(torch.stack(resp_scores), ts, local_prior.respiratory_bands_hz).items()}, **{f"card_{name}": value for name, value in location_spectral_metrics(torch.stack(card_scores), ts, local_prior.cardiac_bands_hz).items()}}
            result[key] = values; stem = key.replace("/", "_"); arrays[f"{stem}_timestamps_s"] = ts.cpu().numpy(); arrays[f"{stem}_resp_scores"] = torch.stack(resp_scores).cpu().numpy(); arrays[f"{stem}_card_scores"] = torch.stack(card_scores).cpu().numpy()
    return result, arrays


def _classify(aggregate: dict[str, dict[str, float]], thresholds: dict[str, float]) -> tuple[str, dict[str, bool]]:
    value = lambda name: aggregate[name]["median"]
    frequency = value("resp_correct_band_fraction") >= thresholds["min_correct_band_fraction"] and value("card_correct_band_fraction") >= thresholds["min_correct_band_fraction"] and value("resp_dominant_in_target_band") == 1.0 and value("card_dominant_in_target_band") == 1.0
    amplitude = all(thresholds["min_dynamic_rms_ratio"] <= value(name) <= thresholds["max_dynamic_rms_ratio"] for name in ("resp_dynamic_rms_ratio", "card_dynamic_rms_ratio"))
    spatial = min(value("resp_dynamic_cosine"), value("card_dynamic_cosine")) >= thresholds["min_dynamic_cosine"]
    endpoint = max(value("resp_endpoint_error_mm"), value("card_endpoint_error_mm")) <= thresholds["max_endpoint_error_mm"]
    checks = {"frequency_recovery": frequency, "dynamic_amplitude_recovery": amplitude, "spatial_dvf_agreement": spatial, "endpoint_error": endpoint}
    if all(checks.values()): return "none", checks
    if value("resp_dynamic_rms_ratio") < thresholds["static_collapse_ratio"] and value("card_dynamic_rms_ratio") < thresholds["static_collapse_ratio"]: return "static_collapse", checks
    failed = [name for name, passed in checks.items() if not passed]
    if len(failed) > 1: return "mixed_failure", checks
    return {"frequency_recovery": "wrong_frequency", "dynamic_amplitude_recovery": "low_amplitude", "spatial_dvf_agreement": "wrong_spatial_field", "endpoint_error": "wrong_spatial_field"}[failed[0]], checks


def _save_phantom_png(path: Path, phantom: AnalyticCanonicalPhantom, domain: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import matplotlib.pyplot as plt
    axes = phantom.volume(torch.tensor(domain["world_min_mm"]), torch.tensor(domain["world_max_mm"]), (64, 64, 64)); volume = axes[-1]
    figure, panels = plt.subplots(1, 3, figsize=(9, 3))
    for panel, image in zip(panels, (volume[32], volume[:, 32], volume[:, :, 32])): panel.imshow(image.T, cmap="gray", origin="lower"); panel.axis("off")
    figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure)
    return axes


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite: raise FileExistsError(f"output directory is non-empty: {path}; pass --overwrite to reuse it")
        for item in path.iterdir(): shutil.rmtree(item) if item.is_dir() else item.unlink()
    path.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA requested but unavailable")
    _prepare_output(args.output_dir, args.overwrite); _set_seed(args.seed)
    with args.source_config.open(encoding="utf-8") as handle: config = yaml.safe_load(handle)
    with args.experiment_config.open(encoding="utf-8") as handle: experiment_config = yaml.safe_load(handle)
    synthetic, thresholds = experiment_config["synthetic"], experiment_config["classification"]
    domain = _load_domain(args.canonical_domain); motion = GroundTruthMotion.from_domain(domain, respiratory_frequency_hz=float(synthetic["respiratory_frequency_hz"]), cardiac_frequency_hz=float(synthetic["cardiac_frequency_hz"]), respiratory_amplitude_mm=float(synthetic["respiratory_amplitude_mm"]), cardiac_amplitude_mm=float(synthetic["cardiac_amplitude_mm"]))
    image_shape = tuple(synthetic["image_shape"])
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=device) if args.manifest else fallback_geometry(locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, image_shape=image_shape, motion=motion, device=device)
    phantom = AnalyticCanonicalPhantom(domain); observations = _build_observations(frames, image_shape=image_shape, phantom=phantom, motion=motion, psf_samples=int(synthetic["psf_samples"]), seed=args.seed)
    preflight = preflight_observations(observations, motion)
    if not preflight["valid"]: raise ValueError(f"synthetic geometry preflight failed: {preflight}")
    prior = build_synthetic_frequency_prior(motion.respiratory_frequency_hz, motion.cardiac_frequency_hz, {(item.view, item.slice_id) for item in observations})
    model = build_source_first_model(config, domain, n_dynamic_frames=len(observations), device=device).to(device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["model"] if "model" in payload else payload)
    if args.mode == "motion_only": model.canonical = PhantomCanonicalAdapter(phantom).to(device)
    schedule = build_experiment_schedule(args.mode, smoke=args.smoke); observations_per_update = training_observations_per_update(config, smoke=args.smoke)
    trainer = UnifiedProgressiveTrainer(model, ViewLocationBalancedSampler(observations, seed=args.seed), pixel_samples=min(args.pixel_samples, 4) if args.smoke else args.pixel_samples, observations_per_update=observations_per_update, optimizer_config=config["training"]["optimizer"], loss_weights=config["training"]["loss_weights"], frequency_prior=prior, temporal_every=1, temporal_batch_size=min(50, args.frames_per_location), cardiac_sampling_fraction=float(config["training"]["cardiac_sampling_fraction"]))
    train_reports = [trainer.run_stage(segment.stage, steps=segment.steps, segment=segment) for segment in schedule]
    by_location, arrays = _per_location_metrics(model, observations, motion, domain, prior); aggregate = aggregate_location_metrics(by_location); failure, checks = _classify(aggregate, thresholds); recovery = "pass" if failure == "none" else "fail"
    x, y, z, volume = _save_phantom_png(args.output_dir / "canonical_three_plane.png", phantom, domain); arrays.update({"canonical_x_mm": x, "canonical_y_mm": y, "canonical_z_mm": z, "canonical_intensity": volume})
    summary = {"experiment": "exp1_synthetic_gt_recovery", "mode": args.mode, "motion_only_recovery": recovery if args.mode == "motion_only" else "indeterminate", "full_joint_recovery": recovery if args.mode == "full_joint" else "indeterminate", "failure_signature": failure, "classification_checks": checks, "input": {"source_config": str(args.source_config), "checkpoint": None if args.checkpoint is None else str(args.checkpoint), "manifest": None if args.manifest is None else str(args.manifest), "canonical_domain": None if args.canonical_domain is None else str(args.canonical_domain)}, "n_locations": len(by_location), "n_frames": len(observations), "preflight": preflight, "schedule": [asdict(segment) for segment in schedule], "training": {"observations_per_update": observations_per_update, "reports": train_reports, "frequency_prior": {"source": prior.source, "provenance": prior.provenance}}, "aggregate_metrics": aggregate, "provenance": {"git_sha": _git_sha(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device), "seed": args.seed, "timestamp_utc": datetime.now(timezone.utc).isoformat()}}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "per_location_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["location", "metric", "value"]); writer.writeheader(); writer.writerows({"location": location, "metric": metric, "value": value} for location, metrics in by_location.items() for metric, value in metrics.items())
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "mean", "median", "p10", "p90"]); writer.writeheader(); writer.writerows({"metric": metric, **values} for metric, values in aggregate.items())
    np.savez_compressed(args.output_dir / "recovery_arrays.npz", **arrays); torch.save({"model": model.state_dict(), "summary": summary}, args.output_dir / "synthetic_recovery_last.pt")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="C4 Experiment 1 synthetic ground-truth recovery")
    parser.add_argument("--source-config", type=Path, required=True); parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml")
    parser.add_argument("--manifest", type=Path); parser.add_argument("--qc-table", type=Path); parser.add_argument("--canonical-domain", type=Path); parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mode", choices=("motion_only", "full_joint"), required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=20); parser.add_argument("--pixel-samples", type=int, default=256); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (args.manifest is None) != (args.qc_table is None): parser.error("--manifest and --qc-table must be provided together")
    summary = run(args); recovery = summary["motion_only_recovery"] if args.mode == "motion_only" else summary["full_joint_recovery"]
    print(f"experiment: {summary['experiment']} ({summary['mode']})"); print(f"input checkpoint/config: {summary['input']['checkpoint']} / {summary['input']['source_config']}"); print(f"n_locations / n_frames: {summary['n_locations']} / {summary['n_frames']}"); print(f"recovery / failure: {recovery} / {summary['failure_signature']}"); print(f"outputs: {args.output_dir / 'summary.json'}, {args.output_dir / 'per_location_metrics.csv'}, {args.output_dir / 'metrics.csv'}, {args.output_dir / 'recovery_arrays.npz'}")


if __name__ == "__main__": main()
