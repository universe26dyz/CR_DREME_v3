#!/usr/bin/env python3
"""Exp3-B1c: joint low-LR stability from a good B1b staged state."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import torch

import experiments.c4_diagnostics.exp3b1_staged_bootstrap_recovery as old_b1
import experiments.c4_diagnostics.exp3b1b_good_mbc_encoder_recovery as b1b
from experiments.c4_diagnostics.exp2_checkpoint_common import parameter_snapshot
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import _ENCODER_PREFIXES
from experiments.c4_diagnostics.exp3a_joint_fullbatch_reconstruction_recovery import _aggregate_dvf, _metrics

_LEVEL0 = "respiratory_mbc.levels.0."
_PRIMARY = ("full_fov_dynamic_cosine", "full_fov_dynamic_rms_ratio", "full_fov_dynamic_normalized_epe")
_CURVE_METRICS = (*_PRIMARY, "full_fov_dynamic_rms_mm", "full_fov_static_rms_mm", "pixel_center_dynamic_cosine", "pixel_center_dynamic_rms_ratio", "pixel_center_dynamic_normalized_epe")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _formal_args_valid(args) -> bool:
    return (args.phase_c_steps, args.locations_per_view, args.frames_per_location, args.pixel_mode, args.pixel_samples, args.accumulation_chunk_size, args.film_lr, args.respiratory_mbc_lr, args.eval_every, args.checkpoint_every) == (1000, 3, 50, "fixed_grid", 256, 5, 5e-5, 5e-5, 250, 250)


def _lineage_global_step(source_global_step: int, phase_step: int) -> int:
    return source_global_step + phase_step


def _admit_source_checkpoint(checkpoint: dict[str, Any], *, seed: int, observation_audit: dict[str, Any], pixel_protocol: dict[str, Any], formal: bool) -> None:
    if checkpoint.get("phase") != "B":
        raise ValueError("B1c source checkpoint must be a B1b Phase-B checkpoint")
    if checkpoint.get("seed") != seed or checkpoint.get("observation_audit") != observation_audit or checkpoint.get("pixel_protocol") != pixel_protocol:
        raise ValueError("B1c source checkpoint seed, geometry, observation audit, or pixel protocol is incompatible")
    protocol = checkpoint.get("protocol", {}); schedule = protocol.get("phase_schedule", {})
    phase_b = protocol.get("phase_b", {})
    if (phase_b.get("observations_per_update"), phase_b.get("pixel_mode"), phase_b.get("pixel_samples"), phase_b.get("accumulation_chunk_size"), phase_b.get("film_lr"), phase_b.get("objective"), tuple(phase_b.get("trainable_modules", ()))) != (450, "fixed_grid", 256, 5, 5e-4, "reconstruction_image_data_mse_only", tuple(_ENCODER_PREFIXES)):
        raise ValueError("B1c source checkpoint does not carry the required B1b Phase-B protocol")
    if formal:
        if checkpoint.get("global_step") != 2750 or schedule.get("phase_a_steps") != 750 or schedule.get("phase_b_steps") != 2000:
            raise ValueError("formal B1c requires a B1b B2000 checkpoint at global step 2750")
        if "B2000" not in checkpoint.get("phase_metrics", {}):
            raise ValueError("formal B1c source checkpoint is missing B2000 metrics")
        if not any(row.get("phase") == "B" and int(row.get("phase_step", -1)) == 2000 for row in checkpoint.get("curve", [])):
            raise ValueError("formal B1c source checkpoint is missing B2000 curve evidence")
        if (observation_audit.get("n_locations"), observation_audit.get("n_frames"), observation_audit.get("frames_per_location")) != (9, 450, 50) or (pixel_protocol.get("pixel_mode"), pixel_protocol.get("actual_pixel_count")) != ("fixed_grid", 256):
            raise ValueError("formal B1c requires the prescribed B1b 9-location, 450-frame fixed-grid source")


def _metric_agrees(current: dict[str, float], saved: dict[str, float]) -> bool:
    return all(math.isclose(current[name], saved[name], rel_tol=1e-5, abs_tol=1e-6) for name in _PRIMARY)


def _source_provenance(path: Path, checkpoint: dict[str, Any], source_dvf: dict[str, float]) -> dict[str, Any]:
    schedule = checkpoint["protocol"]["phase_schedule"]
    return {"source_checkpoint_path": str(path), "source_checkpoint_sha256": _sha256(path), "source_checkpoint_git_sha": checkpoint.get("git_sha"), "source_checkpoint_phase": checkpoint["phase"], "source_checkpoint_global_step": checkpoint["global_step"], "b1c_lineage_c0_global_step": checkpoint["global_step"], "b1c_lineage_c1000_global_step": _lineage_global_step(int(checkpoint["global_step"]), 1000), "source_checkpoint_seed": checkpoint["seed"], "source_phase_a_steps": schedule["phase_a_steps"], "source_phase_b_steps": schedule["phase_b_steps"], "source_observation_audit": checkpoint["observation_audit"], "source_pixel_protocol": checkpoint["pixel_protocol"], "source_B2000_DVF": source_dvf, "source_escape": b1b._escape(source_dvf), "source_strict": b1b._strict(source_dvf)}


def _curve_entry(*, step: int, source_global_step: int, metrics: dict[str, Any], reconstruction_mse: float | None, model, optimizer=None) -> dict[str, Any]:
    aggregate = metrics["aggregate"]
    values = {name: float(aggregate[name]["mean"]) for name in _CURVE_METRICS}
    return {"phase": "C", "phase_step": step, "global_step": _lineage_global_step(source_global_step, step), "full_batch_reconstruction_mse": reconstruction_mse, **values, "encoder_parameter_norm": old_b1._norm(model, "film_encoder."), "respiratory_mbc_level0_parameter_norm": old_b1._norm(model, _LEVEL0), "encoder_grad_norm": None if optimizer is None else old_b1._norm(model, "film_encoder.", True), "respiratory_mbc_level0_grad_norm": None if optimizer is None else old_b1._norm(model, _LEVEL0, True), "film_lr": None if optimizer is None else optimizer.param_groups[0]["lr"], "respiratory_mbc_lr": None if optimizer is None else optimizer.param_groups[-1]["lr"]}


def _classify_joint_trajectory(source: dict[str, float], c_metrics: list[dict[str, float]]) -> dict[str, Any]:
    source_escape, source_strict = b1b._escape(source), b1b._strict(source)
    escaped = [b1b._escape(value) for value in c_metrics]; strict = [b1b._strict(value) for value in c_metrics]
    ever_lost = source_escape and not all(escaped)
    if not source_escape:
        causal = "invalid_good_staged_source"
    elif not escaped[-1]:
        causal = "joint_unfreeze_regression_from_good_state"
    elif ever_lost:
        causal = "joint_unfreeze_escape_instability"
    elif strict[-1]:
        causal = "joint_unfreeze_reaches_strict_recovery"
    else:
        causal = "joint_unfreeze_preserves_escape"
    return {"causal_classification": causal, "source_escape": source_escape, "source_strict": source_strict, "final_escape": escaped[-1], "final_strict": strict[-1], "ever_lost_escape": ever_lost}


def _first_step(values: list[tuple[int, dict[str, float]]], predicate) -> int | None:
    return next((step for step, metric in values if predicate(metric)), None)


def _trajectory_summary(metrics_by_step: dict[int, dict[str, Any]], curve: list[dict[str, Any]]) -> dict[str, Any]:
    dvf = [(step, _aggregate_dvf(metrics)) for step, metrics in sorted(metrics_by_step.items())]
    c_rows = {int(row["phase_step"]): row for row in curve}
    return {"trajectory_flags": {str(step): {"escape": b1b._escape(metric), "strict": b1b._strict(metric)} for step, metric in dvf}, "first_c_escape_step": _first_step(dvf, b1b._escape), "first_c_strict_step": _first_step(dvf, b1b._strict), "first_c_loss_of_escape_step": _first_step(dvf, lambda metric: not b1b._escape(metric)), "best_cosine": {"step": max(dvf, key=lambda item: item[1]["full_fov_dynamic_cosine"])[0], "value": max(dvf, key=lambda item: item[1]["full_fov_dynamic_cosine"])[1]["full_fov_dynamic_cosine"]}, "worst_cosine": {"step": min(dvf, key=lambda item: item[1]["full_fov_dynamic_cosine"])[0], "value": min(dvf, key=lambda item: item[1]["full_fov_dynamic_cosine"])[1]["full_fov_dynamic_cosine"]}, "best_nepe": {"step": min(dvf, key=lambda item: item[1]["full_fov_dynamic_normalized_epe"])[0], "value": min(dvf, key=lambda item: item[1]["full_fov_dynamic_normalized_epe"])[1]["full_fov_dynamic_normalized_epe"]}, "worst_nepe": {"step": max(dvf, key=lambda item: item[1]["full_fov_dynamic_normalized_epe"])[0], "value": max(dvf, key=lambda item: item[1]["full_fov_dynamic_normalized_epe"])[1]["full_fov_dynamic_normalized_epe"]}, "lowest_reconstruction_mse": min((row for row in c_rows.values() if row["full_batch_reconstruction_mse"] is not None), key=lambda row: row["full_batch_reconstruction_mse"], default=None)}


def _inherited_exposure(checkpoint: dict[str, Any]) -> dict[str, Any]:
    schedule = checkpoint["protocol"]["phase_schedule"]; curve = checkpoint.get("curve", [])
    return b1b._executed_exposure_accounting(phase_a_steps=b1b._completed_curve_phase_steps(curve, "A", phase_a_steps=schedule["phase_a_steps"]), phase_b_steps_executed=b1b._completed_curve_phase_steps(curve, "B", phase_a_steps=schedule["phase_a_steps"]))


def _save_checkpoint(path: Path, **state: Any) -> None:
    old_b1._save(path, **state)


def _checkpoint_state(*, model, optimizer, step: int, source_global_step: int, protocol: dict[str, Any], provenance: dict[str, Any], observation_audit: dict[str, Any], pixel_protocol: dict[str, Any], curve: list[dict[str, Any]], phase_metrics: dict[str, Any], snapshots: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "Exp3-B1c", "model": model.state_dict(), "optimizer": optimizer.state_dict(), "phase_step": step, "global_step": _lineage_global_step(source_global_step, step), "phase": "C", "protocol": protocol, "source_provenance": provenance, "observation_audit": observation_audit, "pixel_protocol": pixel_protocol, "curve": curve, "phase_metrics": phase_metrics, "snapshots": snapshots}


def _write_outputs(args, metrics_by_step: dict[int, dict[str, Any]], curve: list[dict[str, Any]], source_provenance: dict[str, Any], stage_audit: dict[str, Any], factor: dict[str, Any], summary: dict[str, Any]) -> None:
    for step, metrics in metrics_by_step.items():
        _write_json(args.output_dir / f"metrics_c_{step}.json", metrics)
    _write_json(args.output_dir / "metrics_final.json", metrics_by_step[max(metrics_by_step)])
    _write_json(args.output_dir / "source_checkpoint_provenance.json", source_provenance)
    _write_json(args.output_dir / "stage_transition_audit.json", stage_audit)
    _write_json(args.output_dir / "factor_diagnostics.json", factor)
    if curve:
        with (args.output_dir / "training_curve.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curve[0])); writer.writeheader(); writer.writerows(curve)
    arrays = {f"source_c0_dvf_metrics" if step == 0 else f"c{step}_dvf_metrics": np.asarray(list(_aggregate_dvf(metrics).values())) for step, metrics in metrics_by_step.items()}
    np.savez_compressed(args.output_dir / "recovery_arrays.npz", **arrays)
    _write_json(args.output_dir / "summary.json", summary)


def run(args) -> dict[str, Any]:
    formal = not args.smoke
    if formal and (not _formal_args_valid(args) or args.manifest is None or args.qc_table is None or args.canonical_domain is None):
        raise ValueError("formal B1c requires the prescribed B2000 source, C1000 protocol, geometry, fixed grid, and low learning rates")
    if args.resume_from is None:
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError("output directory is non-empty")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.source_checkpoint.is_file():
        raise FileNotFoundError(args.source_checkpoint)
    _, motion, observations, groups, observation_audit, pixel_protocol, model = b1b._build_runtime(args)
    source_checkpoint = old_b1._load_checkpoint(args.source_checkpoint)
    _admit_source_checkpoint(source_checkpoint, seed=args.seed, observation_audit=observation_audit, pixel_protocol=pixel_protocol, formal=formal)
    source_global_step = int(source_checkpoint["global_step"])
    model.load_state_dict(source_checkpoint["model"]); old_b1._restore_rng(source_checkpoint["rng_state"])
    source_metrics, _source_arrays = _metrics(model, groups, motion, grid_shape=tuple(args.evaluation_grid_shape)); source_dvf = _aggregate_dvf(source_metrics)
    source_key = f"B{source_checkpoint['protocol']['phase_schedule']['phase_b_steps']}"
    saved_source_metrics = source_checkpoint["phase_metrics"]["B2000"] if formal else source_checkpoint.get("phase_metrics", {}).get(source_key)
    if saved_source_metrics is not None and not _metric_agrees(source_dvf, _aggregate_dvf(saved_source_metrics)):
        raise ValueError("source_checkpoint_metric_mismatch")
    provenance = _source_provenance(args.source_checkpoint, source_checkpoint, source_dvf); _write_json(args.output_dir / "source_checkpoint_provenance.json", provenance)
    protocol = {"phase_c_steps": args.phase_c_steps, "observations_per_update": 450, "pixel_mode": "fixed_grid", "pixel_samples": 256, "accumulation_chunk_size": args.accumulation_chunk_size, "film_lr": args.film_lr, "respiratory_mbc_lr": args.respiratory_mbc_lr, "objective": "reconstruction_image_data_mse_only", "trainable_modules": list(old_b1._prefixes("C")), "gt_respiratory_score_for_training": False}
    if args.resume_from is None and not b1b._escape(source_dvf):
        classification = _classify_joint_trajectory(source_dvf, [source_dvf])
        summary = {"experiment": "Exp3-B1c Good-Staged-State Joint Low-LR Unfreeze", "git_sha": old_b1._git_sha(), "scientific_protocol": protocol, "source_checkpoint_provenance": provenance, "phase_c_entered": False, "phase_c": None, "exposure_accounting": {"inherited_source": _inherited_exposure(source_checkpoint), "phase_c_incremental": b1b._exposure_accounting(steps=0, observations_per_update=450, pixels_per_observation=256), "cumulative_observation_exposures": _inherited_exposure(source_checkpoint)["total_observation_exposures"]}, **classification}
        _write_outputs(args, {0: source_metrics}, [_curve_entry(step=0, source_global_step=source_global_step, metrics=source_metrics, reconstruction_mse=None, model=model)], provenance, {}, {}, summary)
        return summary
    resume = old_b1._load_checkpoint(args.resume_from) if args.resume_from is not None else None
    if resume is not None:
        if resume.get("kind") != "Exp3-B1c" or resume.get("phase") != "C" or resume.get("source_provenance", {}).get("source_checkpoint_sha256") != provenance["source_checkpoint_sha256"] or resume.get("observation_audit") != observation_audit or resume.get("pixel_protocol") != pixel_protocol:
            raise ValueError("incompatible B1c resume checkpoint")
        saved_protocol = resume.get("protocol", {})
        if formal and saved_protocol != protocol:
            raise ValueError("incompatible formal B1c resume protocol")
        try:
            start = int(resume["phase_step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("incompatible B1c resume step provenance") from exc
        if resume.get("global_step") != _lineage_global_step(source_global_step, start):
            raise ValueError("incompatible B1c resume step provenance")
        if start < 0 or start > args.phase_c_steps:
            raise ValueError("B1c resume checkpoint exceeds requested C target")
        model.load_state_dict(resume["model"]); old_b1._restore_rng(resume["rng_state"])
        curve, phase_metrics, snapshots = resume["curve"], resume["phase_metrics"], resume["snapshots"]
    else:
        start = 0; curve = [_curve_entry(step=0, source_global_step=source_global_step, metrics=source_metrics, reconstruction_mse=None, model=model)]; phase_metrics = {"C0": source_metrics}; snapshots = {"source": parameter_snapshot(model)}
    optimizer = old_b1._configure_phase(model, "C", film_lr=args.film_lr, respiratory_mbc_lr=args.respiratory_mbc_lr)
    fresh_optimizer_state_empty = len(optimizer.state) == 0
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
    calls = {"encoder_calls": 0, "respiratory_mbc_calls": 0, "renderer_calls": 0, "cardiac_mbc_calls": 0, "uncertainty_calls": 0}
    hooks = [model.film_encoder.register_forward_hook(lambda *_: calls.__setitem__("encoder_calls", calls["encoder_calls"] + 1)), model.respiratory_mbc.register_forward_hook(lambda *_: calls.__setitem__("respiratory_mbc_calls", calls["respiratory_mbc_calls"] + 1)), model.psf.register_forward_hook(lambda *_: calls.__setitem__("renderer_calls", calls["renderer_calls"] + 1)), model.cardiac_mbc.register_forward_hook(lambda *_: calls.__setitem__("cardiac_mbc_calls", calls["cardiac_mbc_calls"] + 1)), model.uncertainty.register_forward_hook(lambda *_: calls.__setitem__("uncertainty_calls", calls["uncertainty_calls"] + 1))]
    gradient_connectivity_audit = False
    try:
        for step in range(start + 1, args.phase_c_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            batch = old_b1._accumulate_full_batch(model, observations, phase="C", motion=None, pixel_samples=args.pixel_samples, accumulation_chunk_size=args.accumulation_chunk_size)
            frozen_grads = [name for name, parameter in model.named_parameters() if not name.startswith(old_b1._prefixes("C")) and parameter.grad is not None]
            if batch["observation_ids"] != [observation.dynamic_frame_id for observation in observations] or frozen_grads:
                raise RuntimeError("B1c Phase-C full-batch or frozen-gradient audit failed")
            if step == start + 1:
                if not old_b1._assert_expected_gradients(model, "C"):
                    raise RuntimeError("B1c Phase-C expected-gradient connectivity audit failed")
                gradient_connectivity_audit = True
            optimizer.step()
            if step % args.eval_every == 0 or step == args.phase_c_steps:
                metrics, _arrays = _metrics(model, groups, motion, grid_shape=tuple(args.evaluation_grid_shape)); phase_metrics[f"C{step}"] = metrics
                curve.append(_curve_entry(step=step, source_global_step=source_global_step, metrics=metrics, reconstruction_mse=float(batch["full_batch_reconstruction_mse"]), model=model, optimizer=optimizer))
                _write_json(args.output_dir / f"metrics_c_{step}.json", metrics)
                snapshots["C"] = parameter_snapshot(model)
                state = _checkpoint_state(model=model, optimizer=optimizer, step=step, source_global_step=source_global_step, protocol=protocol, provenance=provenance, observation_audit=observation_audit, pixel_protocol=pixel_protocol, curve=curve, phase_metrics=phase_metrics, snapshots=snapshots)
                _save_checkpoint(args.output_dir / f"checkpoint_c_{step}.pt", **state); _save_checkpoint(args.output_dir / "checkpoint_last.pt", **state)
    finally:
        for hook in hooks:
            hook.remove()
    if not gradient_connectivity_audit or calls["encoder_calls"] <= 0 or calls["respiratory_mbc_calls"] <= 0 or calls["renderer_calls"] <= 0 or calls["cardiac_mbc_calls"] != 0 or calls["uncertainty_calls"] != 0:
        raise RuntimeError("B1c runtime or gradient-connectivity audit failed")
    snapshots["C"] = parameter_snapshot(model)
    stage_audit = {"C": old_b1._snapshot_changed(snapshots["source"], snapshots["C"], [name for name in snapshots["source"] if name.startswith(old_b1._prefixes("C"))])}
    encoder_changed = any(name.startswith(_ENCODER_PREFIXES) for name in stage_audit["C"]["changed"]); mbc_changed = any(name.startswith(_LEVEL0) for name in stage_audit["C"]["changed"])
    if stage_audit["C"]["unexpected"] or not encoder_changed or not mbc_changed:
        raise RuntimeError("B1c Phase-C parameter-change audit failed")
    metrics_by_step = {int(key[1:]): value for key, value in phase_metrics.items() if key.startswith("C")}
    final_step = max(metrics_by_step); final_dvf = _aggregate_dvf(metrics_by_step[final_step]); trajectory = _trajectory_summary(metrics_by_step, curve); classification = _classify_joint_trajectory(source_dvf, [_aggregate_dvf(metrics_by_step[step]) for step in sorted(metrics_by_step) if step > 0])
    source_snapshot, final_snapshot = snapshots["source"], snapshots["C"]
    frozen = {"film_encoder_card_unchanged": all(torch.equal(source_snapshot[name], final_snapshot[name]) for name in source_snapshot if name.startswith("film_encoder.card.")), "canonical_unchanged": all(torch.equal(source_snapshot[name], final_snapshot[name]) for name in source_snapshot if name.startswith("canonical.")), "respiratory_mbc_higher_levels_unchanged": all(torch.equal(source_snapshot[name], final_snapshot[name]) for name in source_snapshot if name.startswith("respiratory_mbc.levels.") and not name.startswith(_LEVEL0)), "cardiac_mbc_unchanged": all(torch.equal(source_snapshot[name], final_snapshot[name]) for name in source_snapshot if name.startswith("cardiac_mbc.")), "uncertainty_unchanged": all(torch.equal(source_snapshot[name], final_snapshot[name]) for name in source_snapshot if name.startswith("uncertainty.")), "encoder_changed_during_c": encoder_changed, "respiratory_mbc_level0_changed_during_c": mbc_changed}
    if not all(value is True for value in frozen.values()):
        raise RuntimeError("B1c frozen-module audit failed")
    source_encoder_norm, source_mbc_norm = b1b._state_norm(source_snapshot, _ENCODER_PREFIXES), b1b._state_norm(source_snapshot, _LEVEL0)
    final_encoder_norm, final_mbc_norm = b1b._state_norm(final_snapshot, _ENCODER_PREFIXES), b1b._state_norm(final_snapshot, _LEVEL0)
    factor = {"source_encoder_parameter_norm": source_encoder_norm, "source_respiratory_mbc_level0_parameter_norm": source_mbc_norm, "c1000_encoder_parameter_norm": final_encoder_norm, "c1000_respiratory_mbc_level0_parameter_norm": final_mbc_norm, "encoder_relative_norm_change": final_encoder_norm / source_encoder_norm - 1 if source_encoder_norm else None, "respiratory_mbc_level0_relative_norm_change": final_mbc_norm / source_mbc_norm - 1 if source_mbc_norm else None}
    inherited = _inherited_exposure(source_checkpoint); incremental = b1b._exposure_accounting(steps=final_step, observations_per_update=450, pixels_per_observation=256)
    source_aggregate, final_aggregate = source_metrics["aggregate"], metrics_by_step[final_step]["aggregate"]
    deltas = {f"delta_{name}": float(final_aggregate[name]["mean"] - source_aggregate[name]["mean"]) for name in (*_PRIMARY, "full_fov_static_rms_mm")}
    summary = {"experiment": "Exp3-B1c Good-Staged-State Joint Low-LR Unfreeze", "git_sha": old_b1._git_sha(), "scientific_protocol": protocol, "source_checkpoint_provenance": provenance, "source_dvf": source_dvf, "phase_c_entered": True, "phase_c": {"final_dvf": final_dvf, "escape": b1b._escape(final_dvf), "strict": b1b._strict(final_dvf)}, "fresh_optimizer_state_empty": fresh_optimizer_state_empty, "gradient_connectivity_audit": gradient_connectivity_audit, "runtime_audit": calls, "frozen_module_audit": frozen, "stage_parameter_change_audit": stage_audit, "factor_diagnostics": factor, "exposure_accounting": {"inherited_source": inherited, "phase_c_incremental": incremental, "cumulative_observation_exposures": inherited["total_observation_exposures"] + incremental["observation_exposures"]}, **trajectory, **deltas, **classification}
    _write_outputs(args, metrics_by_step, curve, provenance, stage_audit, factor, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs/source_first_change4_paperaligned.yaml"); parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml")
    parser.add_argument("--manifest", type=Path); parser.add_argument("--qc-table", type=Path); parser.add_argument("--canonical-domain", type=Path); parser.add_argument("--source-checkpoint", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50)
    parser.add_argument("--phase-c-steps", type=int, default=1000); parser.add_argument("--pixel-mode", default="fixed_grid"); parser.add_argument("--pixel-samples", type=int, default=256); parser.add_argument("--accumulation-chunk-size", type=int, default=5); parser.add_argument("--film-lr", type=float, default=5e-5); parser.add_argument("--respiratory-mbc-lr", type=float, default=5e-5); parser.add_argument("--eval-every", type=int, default=250); parser.add_argument("--checkpoint-every", type=int, default=250); parser.add_argument("--resume-from", type=Path); parser.add_argument("--evaluation-grid-shape", type=lambda value: tuple(map(int, value.split(","))), default=(8, 8, 8)); parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
