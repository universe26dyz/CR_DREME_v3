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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField, SequentialPullbackMotion
from cardioresp4d.geometry.world_geometry import DicomPlane
from cardioresp4d.training.build_model import build_source_first_model
from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer

from experiments.c4_diagnostics.synthetic.metrics import dominant_frequency_hz, dvf_metrics, temporal_dynamics_metrics
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from experiments.c4_diagnostics.synthetic.render import GeometryFrame, PhantomCanonicalAdapter, render_synthetic_observation


@dataclass(frozen=True)
class TinySmokeBatch:
    model: nn.Module
    points_mm: torch.Tensor
    timestamps_s: torch.Tensor
    target_intensity: torch.Tensor


class _TinyPullbackField(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3, bias=False)

    def forward(self, points_mm: torch.Tensor) -> torch.Tensor:
        return .01 * self.linear(points_mm)


class _TinyC4StyleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.respiratory = _TinyPullbackField()
        self.cardiac = _TinyPullbackField()
        self.head = nn.Linear(3, 1)

    def forward(self, points_mm: torch.Tensor, timestamps_s: torch.Tensor) -> torch.Tensor:
        motion = SequentialPullbackMotion(self.respiratory, self.cardiac)
        reference = motion(points_mm)["reference_points_mm"]
        return self.head(reference) + .01 * torch.sin(timestamps_s[:, None, None])


def build_tiny_cpu_smoke_batch(seed: int = 0) -> TinySmokeBatch:
    torch.manual_seed(seed)
    points = torch.tensor([[[-2.0, -1.0, 0.0], [1.0, 2.0, -1.0]], [[-1.0, 2.0, 1.0], [2.0, -2.0, 0.5]], [[0.0, 1.0, 2.0], [-2.0, 0.0, -1.0]]])
    times = torch.tensor([0.0, .15, .30])
    target = AnalyticCanonicalPhantom()(points)
    return TinySmokeBatch(_TinyC4StyleModel(), points, times, target)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _frame(view: str, slice_id: str, center: tuple[float, float, float], timestamp_s: float, *, device: torch.device) -> GeometryFrame:
    axes = {"SAX": ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), "2CH": ((1., 0., 0.), (0., 0., 1.), (0., -1., 0.)), "4CH": ((0., 1., 0.), (0., 0., 1.), (1., 0., 0.))}[view]
    return GeometryFrame(view, slice_id, torch.tensor(center, device=device), torch.tensor(axes[0], device=device), torch.tensor(axes[1], device=device), torch.tensor(axes[2], device=device), torch.tensor((1.5, 1.5), device=device), 6.0, timestamp_s)


def fallback_geometry(*, locations_per_view: int, frames_per_location: int, device: torch.device) -> list[GeometryFrame]:
    frames: list[GeometryFrame] = []
    for view_index, view in enumerate(("SAX", "2CH", "4CH")):
        for location in range(locations_per_view):
            for frame_index in range(frames_per_location):
                frames.append(_frame(view, f"synthetic_{view}_{location:02d}", (float((location - 1) * 10), float((view_index - 1) * 8), float((location - 1) * 7)), frame_index * .16 + location * .01, device=device))
    return frames


def geometry_from_manifest(manifest: Path, qc_table: Path | None, *, locations_per_view: int, frames_per_location: int, device: torch.device) -> list[GeometryFrame]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    qc: dict[tuple[str, str, str], bool] = {}
    if qc_table is not None:
        with qc_table.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                qc[(row.get("view", "").upper(), row.get("slice_id", ""), row.get("frame_index", ""))] = str(row.get("qc_valid", "1")).lower() not in {"0", "false"}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        view = row.get("view", "").upper()
        if view not in {"SAX", "2CH", "4CH"} or not qc.get((view, row.get("slice_id", ""), row.get("frame_index", "")), True):
            continue
        grouped.setdefault((view, row["slice_id"]), []).append(row)
    frames: list[GeometryFrame] = []
    for view in ("SAX", "2CH", "4CH"):
        locations = sorted(key for key in grouped if key[0] == view)[:locations_per_view]
        if not locations:
            raise ValueError(f"manifest has no QC-valid {view} geometry")
        for _, slice_id in locations:
            selected = sorted(grouped[(view, slice_id)], key=lambda row: float(row["timestamp_s"]))
            indices = np.linspace(0, len(selected) - 1, min(frames_per_location, len(selected))).round().astype(int)
            for index in indices:
                row = selected[int(index)]; plane = DicomPlane.from_geometry(row)
                frames.append(GeometryFrame(view, slice_id, torch.from_numpy(plane.pixel_to_world((plane.columns - 1) / 2., (plane.rows - 1) / 2.)).float().to(device), torch.from_numpy(plane.row_direction).float().to(device), torch.from_numpy(plane.column_direction).float().to(device), torch.from_numpy(plane.normal).float().to(device), torch.from_numpy(plane.pixel_spacing).float().to(device), float(plane.slice_thickness or row["slice_thickness"]), float(row["timestamp_s"])))
    return frames


def _load_domain(path: Path | None) -> dict[str, object]:
    if path is not None:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    return {"world_min_mm": [-64., -64., -64.], "world_max_mm": [64., 64., 64.], "cardiac_box": {"min_mm": [-24., -24., -28.], "max_mm": [32., 24., 30.]}}


def _build_observations(frames: list[GeometryFrame], *, image_shape: tuple[int, int], motion: GroundTruthMotion, psf_samples: int, seed: int) -> list[DynamicObservation]:
    phantom = AnalyticCanonicalPhantom()
    return [render_synthetic_observation(frame, height=image_shape[0], width=image_shape[1], phantom=phantom, motion=motion, psf_samples=psf_samples, seed=seed + index, dynamic_frame_id=index) for index, frame in enumerate(frames)]


def _score_and_dvf_metrics(model: nn.Module, observations: list[DynamicObservation], motion: GroundTruthMotion, domain: dict[str, object]) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    lower, upper = (torch.tensor(domain["world_min_mm"], device=observations[0].image.device), torch.tensor(domain["world_max_mm"], device=observations[0].image.device))
    grid = torch.stack(torch.meshgrid(*(torch.linspace(lower[axis], upper[axis], 5, device=lower.device) for axis in range(3)), indexing="ij"), dim=-1).reshape(1, -1, 3)
    predicted_resp, predicted_card, gt_resp, gt_card, resp_scores, card_scores, timestamps = [], [], [], [], [], [], []
    model.respiratory_mbc.set_active_levels(3)
    for observation in observations:
        image = observation.image.unsqueeze(0)
        encoded = model.film_encoder(image, center_mm=observation.center_mm[None], row_direction=observation.row_direction[None], column_direction=observation.column_direction[None], normal=observation.normal[None], pixel_spacing_mm=observation.pixel_spacing_mm[None], slice_thickness_mm=torch.tensor([[observation.slice_thickness_mm]], device=image.device))
        resp_field = ScoreWeightedMBCField(model.respiratory_mbc, encoded["resp_scores"])
        card_field = ScoreWeightedMBCField(model.cardiac_mbc, encoded["card_scores"])
        time = grid.new_tensor([observation.timestamp_s])
        predicted_resp.append(resp_field(grid).detach()); predicted_card.append(card_field(grid).detach())
        gt_resp.append(motion.respiratory_dvf_mm(grid, time)); gt_card.append(motion.cardiac_dvf_mm(grid, time))
        resp_scores.append(encoded["resp_scores"].mean().detach()); card_scores.append(encoded["card_scores"].mean().detach()); timestamps.append(observation.timestamp_s)
    ts = torch.tensor(timestamps)
    metrics = {"resp_score_peak_hz": dominant_frequency_hz(torch.stack(resp_scores), ts), "card_score_peak_hz": dominant_frequency_hz(torch.stack(card_scores), ts), "gt_resp_score_peak_hz": dominant_frequency_hz(motion.respiratory_score(ts), ts), "gt_card_score_peak_hz": dominant_frequency_hz(motion.cardiac_score(ts), ts)}
    metrics.update({f"resp_{key}": value for key, value in dvf_metrics(torch.cat(predicted_resp), torch.cat(gt_resp)).items()})
    metrics.update({f"card_{key}": value for key, value in dvf_metrics(torch.cat(predicted_card), torch.cat(gt_card)).items()})
    arrays = {"timestamps_s": ts.numpy(), "predicted_resp_dvf_mm": torch.cat(predicted_resp).cpu().numpy(), "predicted_card_dvf_mm": torch.cat(predicted_card).cpu().numpy(), "gt_resp_dvf_mm": torch.cat(gt_resp).cpu().numpy(), "gt_card_dvf_mm": torch.cat(gt_card).cpu().numpy(), "predicted_resp_scores": torch.stack(resp_scores).cpu().numpy(), "predicted_card_scores": torch.stack(card_scores).cpu().numpy()}
    return metrics, arrays


def _render_metrics(model: nn.Module, observations: list[DynamicObservation]) -> tuple[dict[str, float], np.ndarray]:
    predicted, target = [], []
    with torch.no_grad():
        for observation in observations:
            height, width = observation.image.shape[-2:]
            uv = torch.stack(torch.meshgrid(torch.arange(width, dtype=observation.image.dtype), torch.arange(height, dtype=observation.image.dtype), indexing="xy"), dim=-1).reshape(-1, 2)
            predicted.append(model.predict(observation, uv, "stage3b")["predicted_intensity"].reshape(height, width))
            target.append(observation.image[0])
    prediction, truth = torch.stack(predicted), torch.stack(target)
    return temporal_dynamics_metrics(prediction, truth), prediction.cpu().numpy()


def _save_phantom_png(path: Path, domain: dict[str, object]) -> None:
    import matplotlib.pyplot as plt
    lower, upper = torch.tensor(domain["world_min_mm"]), torch.tensor(domain["world_max_mm"])
    _, _, _, volume = AnalyticCanonicalPhantom().volume(lower, upper, (64, 64, 64))
    figure, axes = plt.subplots(1, 3, figsize=(9, 3))
    for axis, image in zip(axes, (volume[32], volume[:, 32], volume[:, :, 32])):
        axis.imshow(image.T, cmap="gray", origin="lower"); axis.axis("off")
    figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure)


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is non-empty: {path}; pass --overwrite to reuse it")
        for item in path.iterdir():
            if item.is_dir(): shutil.rmtree(item)
            else: item.unlink()
    path.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    _prepare_output(args.output_dir, args.overwrite); _set_seed(args.seed)
    with args.source_config.open(encoding="utf-8") as handle: config = yaml.safe_load(handle)
    with args.experiment_config.open(encoding="utf-8") as handle: experiment_config = yaml.safe_load(handle)
    synthetic = experiment_config["synthetic"]
    domain = _load_domain(args.canonical_domain)
    motion = GroundTruthMotion(respiratory_frequency_hz=float(synthetic["respiratory_frequency_hz"]), cardiac_frequency_hz=float(synthetic["cardiac_frequency_hz"]), respiratory_amplitude_mm=float(synthetic["respiratory_amplitude_mm"]), cardiac_amplitude_mm=float(synthetic["cardiac_amplitude_mm"]))
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=device) if args.manifest else fallback_geometry(locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=device)
    observations = _build_observations(frames, image_shape=tuple(synthetic["image_shape"]), motion=motion, psf_samples=int(synthetic["psf_samples"]), seed=args.seed)
    model = build_source_first_model(config, domain, n_dynamic_frames=len(observations), device=device).to(device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"] if "model" in payload else payload)
    if args.mode == "motion_only":
        model.canonical = PhantomCanonicalAdapter().to(device)
    trainer = UnifiedProgressiveTrainer(model, ViewLocationBalancedSampler(observations, seed=args.seed), pixel_samples=args.pixel_samples, observations_per_update=3, optimizer_config=config["training"]["optimizer"], loss_weights=config["training"]["loss_weights"], temporal_every=1, temporal_batch_size=min(50, args.frames_per_location), cardiac_sampling_fraction=float(config["training"]["cardiac_sampling_fraction"]))
    train_report = trainer.run_stage("stage3b", steps=args.steps)
    score_metrics, arrays = _score_and_dvf_metrics(model, observations, motion, domain)
    image_metrics, prediction = _render_metrics(model, observations)
    arrays["predicted_images"] = prediction
    arrays["target_images"] = np.stack([item.image.cpu().numpy() for item in observations])
    axis_x, axis_y, axis_z, canonical_volume = AnalyticCanonicalPhantom().volume(torch.tensor(domain["world_min_mm"]), torch.tensor(domain["world_max_mm"]), (64, 64, 64))
    arrays.update({"canonical_x_mm": axis_x, "canonical_y_mm": axis_y, "canonical_z_mm": axis_z, "canonical_intensity": canonical_volume})
    frequency_ok = abs(score_metrics["resp_score_peak_hz"] - motion.respiratory_frequency_hz) < .15 and abs(score_metrics["card_score_peak_hz"] - motion.cardiac_frequency_hz) < .30
    amplitude_ok = .15 <= score_metrics["resp_dynamic_rms_ratio"] <= 4.0 and .15 <= score_metrics["card_dynamic_rms_ratio"] <= 4.0
    recovery = "pass" if frequency_ok and amplitude_ok else "fail"
    failure = "none" if recovery == "pass" else ("wrong_frequency" if not frequency_ok else "low_amplitude")
    summary = {"experiment": "exp1_synthetic_gt_recovery", "mode": args.mode, "motion_only_recovery": recovery if args.mode == "motion_only" else "indeterminate", "full_joint_recovery": recovery if args.mode == "full_joint" else "indeterminate", "failure_signature": failure, "input": {"source_config": str(args.source_config), "checkpoint": None if args.checkpoint is None else str(args.checkpoint), "manifest": None if args.manifest is None else str(args.manifest), "canonical_domain": None if args.canonical_domain is None else str(args.canonical_domain)}, "n_locations": len({(item.view, item.slice_id) for item in observations}), "n_frames": len(observations), "train": train_report, "metrics": {**score_metrics, **image_metrics}, "provenance": {"git_sha": _git_sha(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device), "seed": args.seed, "timestamp_utc": datetime.now(timezone.utc).isoformat()}}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"]); writer.writeheader(); writer.writerows({"metric": key, "value": value} for key, value in summary["metrics"].items())
    np.savez_compressed(args.output_dir / "recovery_arrays.npz", **arrays)
    _save_phantom_png(args.output_dir / "canonical_three_plane.png", domain)
    torch.save({"model": model.state_dict(), "summary": summary}, args.output_dir / "synthetic_recovery_last.pt")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="C4 Experiment 1 synthetic ground-truth recovery")
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml")
    parser.add_argument("--manifest", type=Path); parser.add_argument("--qc-table", type=Path); parser.add_argument("--canonical-domain", type=Path); parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mode", choices=("motion_only", "full_joint"), required=True); parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=16); parser.add_argument("--pixel-samples", type=int, default=256); parser.add_argument("--steps", type=int, default=3000); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (args.manifest is None) != (args.qc_table is None): parser.error("--manifest and --qc-table must be provided together")
    summary = run(args)
    print(f"experiment: {summary['experiment']} ({summary['mode']})")
    print(f"input checkpoint/config: {summary['input']['checkpoint']} / {summary['input']['source_config']}")
    print(f"n_locations / n_frames: {summary['n_locations']} / {summary['n_frames']}")
    print(f"recovery / failure: {summary['motion_only_recovery'] if summary['mode'] == 'motion_only' else summary['full_joint_recovery']} / {summary['failure_signature']}")
    print(f"outputs: {args.output_dir / 'summary.json'}, {args.output_dir / 'metrics.csv'}, {args.output_dir / 'recovery_arrays.npz'}, {args.output_dir / 'canonical_three_plane.png'}")


if __name__ == "__main__":
    main()
