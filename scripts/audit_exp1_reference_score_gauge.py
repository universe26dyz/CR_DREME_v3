#!/usr/bin/env python3
"""Read-only retrospective raw/reference/effective-score audit for Exp1."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_source_first import observations_from_manifest
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import apply_corrected_static_poses, load_corrected_pose_map
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField
from cardioresp4d.visualization.dynamic_audit import (discover_boundary_checkpoints, fixed_physical_grid,
                                                     posthoc_center_scores, reconstruct_exp1_checkpoint,
                                                     select_fixed_location_frames, static_dynamic_rms)
from cardioresp4d.visualization.reference_gauge import ReferenceImageProvider


def _metrics(sequence: torch.Tensor) -> dict[str, float]:
    result = static_dynamic_rms(sequence)
    result.update({"temporal_mean": float(sequence.mean()), "temporal_mean_components": [float(value) for value in sequence.mean(dim=0)], "score_rms": float(sequence.square().mean().sqrt()), "temporal_std": float(sequence.std(unbiased=False)), "per_component_static_rms": [float(value) for value in sequence.mean(dim=0).square().sqrt()], "per_component_dynamic_rms": [float(value) for value in (sequence - sequence.mean(dim=0)).square().mean(dim=0).sqrt()]})
    return result


def _plot_location(output: Path, location: tuple[str, str], frames: list, raw: torch.Tensor, reference: torch.Tensor, effective: torch.Tensor, raw_dvf: torch.Tensor, gauge_dvf: torch.Tensor) -> None:
    view, slice_id = location
    times = [float(frame.timestamp_s) for frame in frames]
    component_count = raw.shape[-1]
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
    for index in range(component_count):
        label = f"c{index + 1}"
        axes[0, 0].plot(times, raw[:, index].detach().cpu(), marker="o", label=label)
        axes[0, 1].plot(times, effective[:, index].detach().cpu(), marker="o", label=label)
        axes[1, 0].plot(times, posthoc_center_scores(raw)[:, index].detach().cpu(), marker="o", label=label)
    for index in range(component_count):
        axes[0, 0].axhline(float(reference[:, index].mean()), color=f"C{index}", linestyle="--", alpha=.55, label=f"reference c{index + 1}")
    axes[0, 0].set_title(f"raw score — {view}/{slice_id}")
    axes[0, 1].set_title("effective raw-reference score")
    axes[1, 0].set_title("POST-HOC DIAGNOSTIC: temporal centering (not used in Exp1 training)")
    axes[1, 1].plot(times, raw_dvf.square().mean(dim=(-1, -2)).sqrt().detach().cpu(), marker="o", label="raw DVF RMS")
    axes[1, 1].plot(times, gauge_dvf.square().mean(dim=(-1, -2)).sqrt().detach().cpu(), marker="o", label="gauge DVF RMS")
    axes[1, 1].set_title("DVF RMS vs time")
    for axis in axes.ravel():
        axis.grid(alpha=.25); axis.legend(fontsize=7)
    axes[1, 0].set_xlabel("timestamp (s)"); axes[1, 1].set_xlabel("timestamp (s)")
    figure.suptitle("Reference-score gauge audit; reference branch uses the same current encoder")
    figure.tight_layout(); figure.savefig(output, dpi=140); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qc-table", type=Path, required=True)
    parser.add_argument("--canonical-domain", type=Path, required=True)
    parser.add_argument("--mean-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    domain = json.loads(args.canonical_domain.read_text())
    boundaries = discover_boundary_checkpoints(args.experiment_root)
    if not boundaries:
        raise FileNotFoundError(f"no Exp1 boundary checkpoints under {args.experiment_root}")
    pose_path = args.experiment_root / "02_static_nesvor" / "corrected_pose_manifest.json"
    pose_map = load_corrected_pose_map(pose_path)
    observations_cache = {}
    selected_cache = {}
    provider_cache = {}
    grid = None
    records = []
    for boundary in boundaries:
        checkpoint = torch.load(boundary.path, map_location="cpu", weights_only=False)
        config = checkpoint["effective_config"]
        normalization = str(config.get("training", {}).get("normalization", {}).get("mode", "per_series"))
        from cardioresp4d.training.build_model import uses_s2v_style_film
        cache_key = (normalization, uses_s2v_style_film(config))
        if cache_key not in observations_cache:
            observations, _, _ = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=normalization, assign_slice_positions=cache_key[1])
            observations_cache[cache_key] = apply_corrected_static_poses(observations, pose_map)
            selected_cache[cache_key] = select_fixed_location_frames(observations_cache[cache_key], frames_per_location=3)
            provider_cache[cache_key] = ReferenceImageProvider.from_mean_artifacts(args.mean_manifest, observations=observations_cache[cache_key])
        observations = observations_cache[cache_key]; selected = selected_cache[cache_key]
        model = reconstruct_exp1_checkpoint(checkpoint, domain, observations, device)
        model.set_reference_score_provider(provider_cache[cache_key])
        current_grid, _ = fixed_physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, (16, 16, 16))
        if grid is None:
            grid = current_grid.detach().cpu().clone()
        elif not torch.allclose(grid.to(current_grid), current_grid.detach(), atol=1e-6, rtol=0.):
            raise ValueError("Exp1 gauge audit grid changed across checkpoints")
        checkpoint_dir = args.output_dir / f"checkpoint_{boundary.label}" / "reference_gauge"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for location, frames in selected.items():
            raw_values, reference_values, effective_values, raw_dvfs, gauge_dvfs = [], [], [], [], []
            with torch.no_grad():
                for frame in frames:
                    raw = model.encode_motion_scores_raw(frame)["resp_scores"][:, :boundary.active_levels].reshape(-1)
                    reference = model.encode_reference_scores(frame)["resp_scores"][:, :boundary.active_levels].reshape(-1)
                    effective = model.encode_motion_scores(frame)["resp_scores"][:, :boundary.active_levels].reshape(-1)
                    raw_values.append(raw); reference_values.append(reference); effective_values.append(effective)
                    raw_dvfs.append(ScoreWeightedMBCField(model.respiratory_mbc, raw[None])(current_grid)[0])
                    gauge_dvfs.append(ScoreWeightedMBCField(model.respiratory_mbc, effective[None])(current_grid)[0])
            raw, reference, effective = torch.stack(raw_values), torch.stack(reference_values), torch.stack(effective_values)
            raw_dvf, gauge_dvf = torch.stack(raw_dvfs), torch.stack(gauge_dvfs)
            item = {"checkpoint": boundary.label, "stage": boundary.stage, "active_levels": boundary.active_levels, "view": location[0], "slice_id": location[1], "frames": len(frames), "raw_score": _metrics(raw), "reference_score": _metrics(reference), "effective_gauge_score": _metrics(effective), "raw_dvf": {**_metrics(raw_dvf), "units": "mm"}, "gauge_dvf": {**_metrics(gauge_dvf), "units": "mm"}}
            records.append(item)
            _plot_location(checkpoint_dir / f"{location[0]}_{location[1]}.png", location, frames, raw, reference, effective, raw_dvf, gauge_dvf)
    pooled = {}
    for checkpoint in sorted({item["checkpoint"] for item in records}):
        subset = [item for item in records if item["checkpoint"] == checkpoint]
        def aggregate(items):
            return {key: float(np.mean([item[key] for item in items])) for key in ("static_rms", "dynamic_rms", "static_dynamic_ratio", "score_rms", "temporal_std")}
        def aggregate_dvf(items):
            return {key: float(np.mean([item[key] for item in items])) for key in ("static_rms", "dynamic_rms", "static_dynamic_ratio")}
        pooled[checkpoint] = {"locations": len(subset), "raw_score": aggregate([item["raw_score"] for item in subset]), "gauge_score": aggregate([item["effective_gauge_score"] for item in subset]), "raw_dvf": aggregate_dvf([item["raw_dvf"] for item in subset]), "gauge_dvf": aggregate_dvf([item["gauge_dvf"] for item in subset]), "by_view": {view: {"locations": sum(item["view"] == view for item in subset), "raw_score": aggregate([item["raw_score"] for item in subset if item["view"] == view]), "gauge_score": aggregate([item["effective_gauge_score"] for item in subset if item["view"] == view]), "raw_dvf": aggregate_dvf([item["raw_dvf"] for item in subset if item["view"] == view]), "gauge_dvf": aggregate_dvf([item["gauge_dvf"] for item in subset if item["view"] == view])} for view in ("SAX", "2CH", "4CH") if any(item["view"] == view for item in subset)}}
    payload = {"scientific_interpretation": "pending", "reference_policy": "authoritative mean_stack[..., stack_index].T; same current encoder, no detached/precomputed scores", "cohort_policy": "one physical middle location per view; early/middle/late QC-valid frames from that fixed location", "fixed_grid_shape": [16, 16, 16], "checkpoints": [item.label for item in boundaries], "pooled": pooled, "locations": records}
    (args.output_dir / "reference_gauge_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output_dir / "reference_gauge_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["checkpoint", "stage", "view", "slice_id", "raw_score_dynamic_rms", "effective_score_dynamic_rms", "raw_dvf_dynamic_rms_mm", "gauge_dvf_dynamic_rms_mm"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for item in records:
            writer.writerow({"checkpoint": item["checkpoint"], "stage": item["stage"], "view": item["view"], "slice_id": item["slice_id"], "raw_score_dynamic_rms": item["raw_score"]["dynamic_rms"], "effective_score_dynamic_rms": item["effective_gauge_score"]["dynamic_rms"], "raw_dvf_dynamic_rms_mm": item["raw_dvf"]["dynamic_rms"], "gauge_dvf_dynamic_rms_mm": item["gauge_dvf"]["dynamic_rms"]})
    print(json.dumps({"status": "audited", "checkpoints": len(boundaries), "locations": len(records), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
