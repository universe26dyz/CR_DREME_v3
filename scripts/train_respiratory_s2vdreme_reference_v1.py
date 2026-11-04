#!/usr/bin/env python3
"""Launch the isolated Static-NeSVoR + S2V-DREME respiratory reference protocol."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_source_first import observations_from_manifest
from cardioresp4d.adapters.source_lock import verify_vendored_source_lock
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import (
    REFERENCE_CHECKPOINT_BOUNDARIES, S2VDREMEReferenceRunner, canonical_init_provenance, joint_stage_audits,
    reference_schedule, strict_canonical_state_from_static_checkpoint, validate_reference_config,
    static_volume_samples, warm_start_canonical_from_static_volume, apply_corrected_static_poses, load_corrected_pose_map, static_reference_diagnostics,
)
from cardioresp4d.training.build_model import build_source_first_model, effective_model_config
from cardioresp4d.training.runtime_state import capture_rng_state, restore_rng_state, set_reproducibility
from cardioresp4d.training.sampler import ViewLocationBalancedSampler, assign_normalized_slice_positions

CHECKPOINT_SCHEMA = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    parent = config.pop("extends", None)
    if parent is None:
        return config
    base = _load_config(path.parent / parent)
    def merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(dict(left))
        for key, value in right.items():
            result[key] = merge(result[key], value) if isinstance(value, Mapping) and isinstance(result.get(key), Mapping) else copy.deepcopy(value)
        return result
    return merge(base, config)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _provenance(config: Mapping[str, Any], args, source_lock: Mapping[str, Any]) -> dict[str, Any]:
    return {"config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(), "git_sha": _git_head(), "manifest_sha256": _sha256(args.manifest), "qc_table_sha256": _sha256(args.qc_table), "canonical_domain_sha256": _sha256(args.canonical_domain), "static_manifest_sha256": _sha256(args.static_manifest), "source_lock": dict(source_lock)}


def resume_next_segment_index(checkpoint: Mapping[str, Any], segments, provenance: Mapping[str, Any]) -> int:
    if checkpoint.get("checkpoint_schema") != CHECKPOINT_SCHEMA or checkpoint.get("provenance") != dict(provenance):
        raise ValueError("resume checkpoint provenance does not exactly match this reference protocol")
    state = checkpoint.get("runner_state")
    if not isinstance(state, Mapping) or not isinstance(state.get("completed"), list):
        raise ValueError("resume checkpoint lacks a completed reference boundary")
    completed = state["completed"]
    for index, item in enumerate(completed):
        if index >= len(segments) or item.get("segment") != segments[index].name or int(item.get("segment_step", -1)) != segments[index].steps:
            raise ValueError("resume checkpoint is not a reference schedule boundary prefix")
    return len(completed)


def _checkpoint(*, model, runner, config, provenance, static_init) -> dict[str, Any]:
    return {"checkpoint_schema": CHECKPOINT_SCHEMA, "model": model.state_dict(), "runner_state": runner.state_dict(), "rng_state": capture_rng_state(), "effective_config": config, "provenance": provenance, "static_initialization": static_init}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, default=ROOT / "configs" / "respiratory_s2vdreme_reference_v1.yaml")
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--qc-table", required=True, type=Path); parser.add_argument("--canonical-domain", required=True, type=Path)
    parser.add_argument("--static-manifest", required=True, type=Path, help="authoritative static_reference_manifest.json")
    parser.add_argument("--static-volume", type=Path, help="Verified static NeSVoR NIfTI override for explicit MSE fallback")
    parser.add_argument("--static-warm-start-steps", type=int, default=0, help="Explicit volume-MSE fallback iterations; never used for exact state admission")
    parser.add_argument("--output-dir", required=True, type=Path); parser.add_argument("--device", default="cuda"); parser.add_argument("--pixel-samples", type=int, default=256); parser.add_argument("--resume", type=Path)
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    config = _load_config(args.experiment_config); validate_reference_config(config)
    source_lock = verify_vendored_source_lock(ROOT); provenance = _provenance(config, args, source_lock)
    static_manifest = json.loads(args.static_manifest.read_text())
    if static_manifest.get("source_manifest_sha256") != _sha256(args.manifest) or static_manifest.get("qc_table_sha256") != _sha256(args.qc_table):
        raise ValueError("static manifest does not bind this dynamic manifest/QC table")
    if static_manifest.get("scientific_contract", {}).get("output_intensity_mean") != 0:
        raise ValueError("static manifest does not prove output_intensity_mean=0")
    static_model = Path(static_manifest["outputs"]["model"]["path"])
    static_volume = args.static_volume or Path(static_manifest["outputs"]["volume"]["path"])
    pose_manifest = Path(static_manifest["corrected_pose_manifest"]["path"])
    if _sha256(static_model) != static_manifest["outputs"]["model"]["sha256"] or _sha256(static_volume) != static_manifest["outputs"]["volume"]["sha256"] or _sha256(pose_manifest) != static_manifest["corrected_pose_manifest"]["sha256"]:
        raise ValueError("static manifest artifact hashes do not match")
    domain = json.loads(args.canonical_domain.read_text()); reproducibility = set_reproducibility(int(config["training"]["seed"]))
    observations, normalization, normalization_groups = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=config["training"]["normalization"]["mode"], assign_slice_positions=False)
    observations = assign_normalized_slice_positions(apply_corrected_static_poses(observations, load_corrected_pose_map(pose_manifest)))
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    model = build_source_first_model(config, domain, n_dynamic_frames=max(item.dynamic_frame_id for item in observations) + 1, device=device).to(device)
    static_payload = torch.load(static_model, map_location="cpu", weights_only=False)
    try:
        bounds = torch.stack((model.canonical_lower_world_mm, model.canonical_upper_world_mm))
        static_state = strict_canonical_state_from_static_checkpoint(static_payload, model.canonical.inr.state_dict(), dynamic_bounds_world_mm=bounds, expected_architecture=model.canonical.args)
    except ValueError:
        if args.static_warm_start_steps <= 0:
            raise ValueError("static INR state is incompatible; pass --static-volume and positive --static-warm-start-steps for the explicit MSE fallback")
        points, targets = static_volume_samples(static_volume, max_points=262144, device=device)
        warm_report = warm_start_canonical_from_static_volume(model, points, targets, steps=args.static_warm_start_steps)
        static_init = {**canonical_init_provenance(static_model, mode="static_volume_mse_warm_start"), "static_volume_path": str(static_volume), "static_volume_sha256": _sha256(static_volume), "warm_start": warm_report, "intensity_statistics": {"min": float(targets.min()), "max": float(targets.max()), "mean": float(targets.mean()), "std": float(targets.std(unbiased=False))}}
    else:
        model.canonical.inr.load_state_dict(static_state, strict=True)
        if not torch.allclose(model.canonical.inr.bounding_box, bounds, atol=1e-5, rtol=0.0): raise RuntimeError("direct static load changed canonical bounding_box")
        static_init = canonical_init_provenance(static_model, mode="direct_exact_state")
    sampler = ViewLocationBalancedSampler(observations, seed=int(config["training"]["seed"]))
    runner = S2VDREMEReferenceRunner(model, sampler, pixel_samples=args.pixel_samples, observations_per_update=int(config["training"]["observations_per_update"]), optimizer_config=config["training"]["optimizer"])
    segments = reference_schedule(); start = 0
    if checkpoint is not None:
        start = resume_next_segment_index(checkpoint, segments, provenance)
        model.load_state_dict(checkpoint["model"]); runner.load_boundary_state(checkpoint["runner_state"]); restore_rng_state(checkpoint["rng_state"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective = {"requested": config, "constructed": {**effective_model_config(model), "optimizer": config["training"]["optimizer"], "schedule": [segment.__dict__ for segment in reference_schedule()], "observations_per_update": runner.observations_per_update, "loss_weights": config["training"]["loss_weights"]}, "source_lock": source_lock}
    (args.output_dir / "effective_config.json").write_text(json.dumps(effective, indent=2) + "\n")
    static_diagnostics = static_reference_diagnostics(model, sampler)
    (args.output_dir / "static_reference_ready.json").write_text(json.dumps({**static_init, **static_diagnostics, "static_manifest": str(args.static_manifest), "static_manifest_sha256": _sha256(args.static_manifest), "canonical_bbox": bounds.detach().cpu().tolist(), "static_pose_map": str(pose_manifest)}, indent=2) + "\n")
    for segment in segments[start:]:
        runner.run_segment(segment)
        payload = _checkpoint(model=model, runner=runner, config=config, provenance=provenance, static_init=static_init)
        torch.save(payload, args.output_dir / f"checkpoint_{segment.name}_end.pt")
        torch.save(payload, args.output_dir / "checkpoint_last.pt")
        (args.output_dir / "boundary_diagnostics.json").write_text(json.dumps(runner.completed, indent=2) + "\n")
    audits = joint_stage_audits(runner.completed)
    (args.output_dir / "joint_stage_audits.json").write_text(json.dumps(audits, indent=2) + "\n")
    report = {"experiment": "respiratory_s2vdreme_reference_v1", "checkpoints": list(REFERENCE_CHECKPOINT_BOUNDARIES) + ["checkpoint_last.pt"], "segments": runner.completed, "joint_stage_audits": audits, "static_initialization": static_init, "provenance": provenance, "normalization_parameters": normalization, "normalization_groups": normalization_groups, "reproducibility": reproducibility, "device": str(device), "resumed_from": None if args.resume is None else str(args.resume)}
    (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
