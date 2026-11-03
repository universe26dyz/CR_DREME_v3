#!/usr/bin/env python3
"""Exp3-B1b: good-MBC bootstrap followed by encoder-only recovery."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import torch
import yaml

from cardioresp4d.training.sampler import ViewLocationBalancedSampler
from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, _load_domain, fallback_geometry, geometry_from_manifest
from experiments.c4_diagnostics.exp2_checkpoint_common import parameter_snapshot
from experiments.c4_diagnostics.exp2c4a_gt_score_mbc_only import _assert_only_level0_changed, _build_intervention_model, _configure_level0_only, _evaluate, _intervention_loss_terms, _respiratory_only_motion, _set_seed, _validate_observations
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import _ENCODER_PREFIXES
from experiments.c4_diagnostics.exp2c4h_oracle_field_fullbatch_reconstruction import _fixed_pixel_grid
import experiments.c4_diagnostics.exp3b1_staged_bootstrap_recovery as old_b1
from experiments.c4_diagnostics.exp3a_joint_fullbatch_reconstruction_recovery import _aggregate_dvf, _metrics
from experiments.c4_diagnostics.synthetic.metrics import aggregate_location_metrics
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom

_LEVEL0 = "respiratory_mbc.levels.0."
_PRIMARY = ("full_fov_dynamic_cosine", "full_fov_dynamic_rms_ratio", "full_fov_dynamic_normalized_epe")
_PAIRED_METRICS = (*_PRIMARY, "full_fov_dynamic_rms_mm", "full_fov_static_rms_mm", "pixel_center_dynamic_cosine", "pixel_center_dynamic_rms_ratio", "pixel_center_dynamic_normalized_epe", "full_batch_reconstruction_mse")


def _exposure_accounting(*, steps: int, observations_per_update: int, pixels_per_observation: int) -> dict[str, int]:
    return {"optimizer_steps": steps, "observations_per_update": observations_per_update, "pixels_per_observation": pixels_per_observation, "observation_exposures": steps * observations_per_update, "pixel_observation_exposures": steps * observations_per_update * pixels_per_observation}


def _phase_a_protocol() -> dict[str, Any]:
    return {"steps": 750, "observations_per_update": 32, "pixel_mode": "fixed_grid", "pixel_samples": 256, "objective": "data_only", "trainable_modules": [_LEVEL0], "gt_respiratory_score_for_training": True, "exposure_accounting": _exposure_accounting(steps=750, observations_per_update=32, pixels_per_observation=256)}


def _phase_b_protocol() -> dict[str, Any]:
    return {"steps": 1000, "observations_per_update": 450, "pixel_mode": "fixed_grid", "pixel_samples": 256, "accumulation_chunk_size": 5, "film_lr": 5e-4, "objective": "reconstruction_image_data_mse_only", "trainable_modules": list(_ENCODER_PREFIXES), "gt_respiratory_score_for_training": False, "exposure_accounting": _exposure_accounting(steps=1000, observations_per_update=450, pixels_per_observation=256)}


def _phase_b_eval_steps(steps: int) -> tuple[int, ...]:
    if steps == 1000:
        return (0, 250, 500, 750, 1000)
    return tuple(range(0, steps + 1, max(1, steps // 4))) if steps else (0,)


def _escape(metric: dict[str, float]) -> bool:
    return metric["full_fov_dynamic_cosine"] >= .90 and .5 <= metric["full_fov_dynamic_rms_ratio"] <= 1.5 and metric["full_fov_dynamic_normalized_epe"] <= .75


def _strict(metric: dict[str, float]) -> bool:
    return metric["full_fov_dynamic_cosine"] >= .95 and .8 <= metric["full_fov_dynamic_rms_ratio"] <= 1.2 and metric["full_fov_dynamic_normalized_epe"] <= .50


def _classify(phase_a: dict[str, float], phase_b: dict[str, float] | None) -> dict[str, Any]:
    if not _escape(phase_a):
        return {"causal_classification": "good_mbc_bootstrap_failure", "phase_a_escape": False, "phase_b_escape": None, "phase_b_strict": None}
    if phase_b is None or not _escape(phase_b):
        return {"causal_classification": "encoder_given_good_mbc_failure", "phase_a_escape": True, "phase_b_escape": False, "phase_b_strict": False}
    if _strict(phase_b):
        return {"causal_classification": "good_mbc_enables_encoder_strict_recovery", "phase_a_escape": True, "phase_b_escape": True, "phase_b_strict": True}
    return {"causal_classification": "good_mbc_enables_encoder_escape", "phase_a_escape": True, "phase_b_escape": True, "phase_b_strict": False}


def _parse_existing_b1_phase_b_at_step(curve: Path, phase_step: int = 1000) -> dict[str, float | None]:
    with curve.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        has_phase_step = "phase_step" in (reader.fieldnames or [])
        for row in reader:
            if row.get("phase") != "B":
                continue
            is_target = int(row.get("phase_step", -1)) == phase_step if has_phase_step else int(row.get("global_step", -1)) == 3000
            if not is_target:
                continue
            try:
                parsed = {name: float(row[name]) for name in _PRIMARY}
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("existing B1 Phase-B@1000 curve row is missing primary DVF metrics") from exc
            if not all(math.isfinite(value) for value in parsed.values()):
                raise ValueError("existing B1 Phase-B@1000 curve row has non-finite primary DVF metrics")
            for name in _PAIRED_METRICS[len(_PRIMARY):]:
                try:
                    value = float(row[name])
                except (KeyError, TypeError, ValueError):
                    parsed[name] = None
                else:
                    parsed[name] = value if math.isfinite(value) else None
            return parsed
    selector = f"phase_step={phase_step}" if has_phase_step else "global_step=3000 without phase_step column"
    raise ValueError(f"existing B1 curve has no Phase-B@1000 row ({selector})")


def _write_paired_comparison(output_dir: Path, new_metrics: dict[str, Any], new_reconstruction_mse: float | None, old_metrics: dict[str, float | None]) -> dict[str, Any]:
    aggregate = new_metrics["aggregate"]
    new_values = {name: float(aggregate[name]["mean"]) for name in _PAIRED_METRICS[:-1]}
    new_values["full_batch_reconstruction_mse"] = new_reconstruction_mse
    comparison = {"old_b1_phase_b_phase_step": 1000, "new_b1b_phase_b_phase_step": 1000, "metrics": {name: {"old_b1_phase_b_1000": old_metrics.get(name), "new_b1b_phase_b_1000": new_values[name]} for name in _PAIRED_METRICS}}
    _write_json(output_dir / "paired_phase_b_1000_comparison.json", comparison)
    with (output_dir / "paired_phase_b_1000_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "old_b1_phase_b_1000", "new_b1b_phase_b_1000"]); writer.writeheader()
        writer.writerows({"metric": name, **values} for name, values in comparison["metrics"].items())
    return comparison


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _phase_a_metric(rows: dict[str, dict[str, float]]) -> dict[str, Any]:
    return {"per_location": rows, "aggregate": aggregate_location_metrics(rows)}


def _state_norm(state: dict[str, torch.Tensor], prefixes: str | tuple[str, ...]) -> float:
    values = [value.reshape(-1) for name, value in state.items() if name.startswith(prefixes)]
    return float(torch.cat(values).norm()) if values else 0.


def _formal_args_valid(args) -> bool:
    return (args.phase_a_steps, args.phase_b_steps, args.phase_a_observations_per_update, args.pixel_mode, args.pixel_samples, args.locations_per_view, args.frames_per_location, args.accumulation_chunk_size) == (750, 1000, 32, "fixed_grid", 256, 3, 50, 5)


def _checkpoint_state(*, model, optimizer, global_step: int, phase: str, protocol: dict[str, Any], pixel_protocol: dict[str, Any], observation_audit: dict[str, Any], curve: list[dict[str, Any]], initial_metrics: dict[str, Any], phase_metrics: dict[str, Any], snapshots: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "global_step": global_step, "phase": phase, "protocol": protocol, "pixel_protocol": pixel_protocol, "observation_audit": observation_audit, "curve": curve, "initial_metrics": initial_metrics, "phase_metrics": phase_metrics, "snapshots": snapshots}


def _save_checkpoint(path: Path, **state: Any) -> None:
    old_b1._save(path, **state)


def _build_runtime(args):
    device = torch.device(args.device); _set_seed(args.seed)
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); synthetic = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))["synthetic"]
    domain = _load_domain(args.canonical_domain)
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(domain, respiratory_frequency_hz=float(synthetic["respiratory_frequency_hz"]), cardiac_frequency_hz=float(synthetic["cardiac_frequency_hz"]), respiratory_amplitude_mm=float(synthetic["respiratory_amplitude_mm"]), cardiac_amplitude_mm=float(synthetic["cardiac_amplitude_mm"])))
    shape, psf = ((8, 8), 1) if args.smoke else (tuple(synthetic["image_shape"]), int(synthetic["psf_samples"]))
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=device) if args.manifest else fallback_geometry(locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, image_shape=shape, motion=motion, device=device)
    observations = _build_observations(frames, image_shape=shape, phantom=AnalyticCanonicalPhantom(domain).to(device), motion=motion, psf_samples=psf, seed=args.seed)
    groups = _validate_observations(observations, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, smoke=args.smoke)
    observation_audit = old_b1._observation_audit(observations); fixed_uv = _fixed_pixel_grid(observations[0], count=args.pixel_samples)
    pixel_protocol = {"pixel_mode": "fixed_grid", "requested_pixel_samples": args.pixel_samples, "actual_pixel_count": len(fixed_uv), "fixed_uv": fixed_uv.long().cpu().tolist()}
    model = _build_intervention_model(config, domain, n_dynamic_frames=len(observations), init_mode="fresh", checkpoint=None, seed=args.seed, device=device)
    return config, motion, observations, groups, observation_audit, pixel_protocol, model


def run(args) -> dict[str, Any]:
    if args.resume_from is not None:
        return _resume(args)
    if not args.smoke and (not _formal_args_valid(args) or args.manifest is None or args.qc_table is None or args.canonical_domain is None or args.eval_every != 250):
        raise ValueError("formal Exp3-B1b requires the prescribed 750/1000, stochastic32/fullbatch450, fixed-grid geometry, and 250-step evaluation protocol")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError("output directory is non-empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config, motion, observations, groups, observation_audit, pixel_protocol, model = _build_runtime(args)
    phase_a_protocol, phase_b_protocol = _phase_a_protocol(), _phase_b_protocol()
    protocol = {"phase_a": phase_a_protocol, "phase_b": phase_b_protocol, "phase_schedule": {"phase_a_steps": args.phase_a_steps, "phase_b_steps": args.phase_b_steps}, "no_phase_c": True}
    snapshots = {"start": parameter_snapshot(model)}
    initial_rows, initial_arrays, _ = _evaluate(model, groups, motion, grid_shape=tuple(args.evaluation_grid_shape))
    initial_metrics = _phase_a_metric(initial_rows)
    phase_metrics: dict[str, Any] = {}; phase_arrays: dict[str, dict[str, np.ndarray]] = {"initial": initial_arrays}; curve: list[dict[str, Any]] = []
    module_calls = {"phase_a_encoder_calls": 0, "phase_b_encoder_calls": 0, "renderer_calls": 0, "respiratory_mbc_calls": 0, "cardiac_mbc_calls": 0, "uncertainty_calls": 0}; active_phase = ["A"]

    def _count_encoder(_module, _args, _output): module_calls["phase_a_encoder_calls" if active_phase[0] == "A" else "phase_b_encoder_calls"] += 1
    def _count_renderer(_module, _args, _output): module_calls["renderer_calls"] += 1
    def _count_mbc(_module, _args, _output): module_calls["respiratory_mbc_calls"] += 1
    def _count_card(_module, _args, _output): module_calls["cardiac_mbc_calls"] += 1
    def _count_unc(_module, _args, _output): module_calls["uncertainty_calls"] += 1

    hooks = [model.film_encoder.register_forward_hook(_count_encoder), model.psf.register_forward_hook(_count_renderer), model.respiratory_mbc.register_forward_hook(_count_mbc), model.cardiac_mbc.register_forward_hook(_count_card), model.uncertainty.register_forward_hook(_count_unc)]
    try:
        optimizer = _configure_level0_only(model, learning_rate=float(config["training"]["optimizer"]["stage2"]["respiratory_mbc_lr"]))
        sampler = ViewLocationBalancedSampler(observations, seed=args.seed)
        for phase_step in range(1, args.phase_a_steps + 1):
            optimizer.zero_grad(set_to_none=True); selected = sampler.sample_step(observations_per_update=args.phase_a_observations_per_update)
            losses = [_intervention_loss_terms(model, observation, motion, pixel_uv=_fixed_pixel_grid(observation, count=args.pixel_samples), objective="data_only", normalization_weight=0.) for observation in selected]
            loss = torch.stack([item["data_mse"] for item in losses]).mean(); loss.backward()
            frozen_grads = [name for name, parameter in model.named_parameters() if not name.startswith(_LEVEL0) and parameter.grad is not None]
            expected_grad = any(name.startswith(_LEVEL0) and parameter.grad is not None and torch.isfinite(parameter.grad).all() and float(parameter.grad.norm()) > 0 for name, parameter in model.named_parameters())
            if frozen_grads or (phase_step == 1 and not expected_grad):
                raise RuntimeError("B1b Phase-A gradient ownership audit failed")
            optimizer.step(); curve.append({"global_step": phase_step, "phase": "A", "phase_step": phase_step, "full_batch_reconstruction_mse": float(loss.detach())})
        phase_a_rows, phase_a_arrays, _ = _evaluate(model, groups, motion, grid_shape=tuple(args.evaluation_grid_shape)); phase_a_metrics = _phase_a_metric(phase_a_rows); phase_arrays["phase_a"] = phase_a_arrays; phase_metrics["A"] = phase_a_metrics; a_dvf = _aggregate_dvf(phase_a_metrics)
        _assert_only_level0_changed(model, snapshots["start"]); snapshots["A"] = parameter_snapshot(model); stage_audit: dict[str, Any] = {"A": old_b1._snapshot_changed(snapshots["start"], snapshots["A"], [name for name in snapshots["start"] if name.startswith(_LEVEL0)])}
        _write_json(args.output_dir / "metrics_initial.json", initial_metrics); _write_json(args.output_dir / "metrics_phase_a_end.json", phase_a_metrics)
        state = _checkpoint_state(model=model, optimizer=optimizer, global_step=args.phase_a_steps, phase="A", protocol=protocol, pixel_protocol=pixel_protocol, observation_audit=observation_audit, curve=curve, initial_metrics=initial_metrics, phase_metrics=phase_metrics, snapshots=snapshots)
        _save_checkpoint(args.output_dir / "checkpoint_phase_a_end.pt", **state); _save_checkpoint(args.output_dir / "checkpoint_last.pt", **state)
        if not _escape(a_dvf):
            return _finish(args, initial_metrics, phase_metrics, phase_arrays, snapshots, stage_audit, module_calls, a_dvf, None, curve, protocol, observation_audit, pixel_protocol, optimizer)
        active_phase[0] = "B"; optimizer = old_b1._configure_phase(model, "B", film_lr=5e-4, respiratory_mbc_lr=5e-4)
        phase_b_initial, phase_b_initial_arrays = _metrics(model, groups, motion, grid_shape=tuple(args.evaluation_grid_shape)); phase_metrics["B0"] = phase_b_initial; phase_arrays["phase_b_0"] = phase_b_initial_arrays; _write_json(args.output_dir / "metrics_phase_b_0.json", phase_b_initial); curve.append({"global_step": args.phase_a_steps, "phase": "B", "phase_step": 0, **_aggregate_dvf(phase_b_initial)})
        for phase_step in range(1, args.phase_b_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            batch = old_b1._accumulate_full_batch(model, observations, phase="B", motion=None, pixel_samples=args.pixel_samples, accumulation_chunk_size=args.accumulation_chunk_size)
            frozen_grads = [name for name, parameter in model.named_parameters() if not name.startswith(old_b1._prefixes("B")) and parameter.grad is not None]
            if batch["observation_ids"] != [observation.dynamic_frame_id for observation in observations] or frozen_grads:
                raise RuntimeError("B1b Phase-B full-batch or frozen-gradient audit failed")
            if phase_step == 1 and not old_b1._assert_expected_gradients(model, "B"):
                raise RuntimeError("B1b Phase-B expected-gradient connectivity audit failed")
            optimizer.step(); global_step = args.phase_a_steps + phase_step
            if phase_step % args.eval_every == 0 or phase_step == args.phase_b_steps:
                metrics, arrays = _metrics(model, groups, motion, grid_shape=tuple(args.evaluation_grid_shape)); phase_metrics[f"B{phase_step}"] = metrics; phase_arrays[f"phase_b_{phase_step}"] = arrays; dvf = _aggregate_dvf(metrics)
                curve.append({"global_step": global_step, "phase": "B", "phase_step": phase_step, "full_batch_reconstruction_mse": float(batch["full_batch_reconstruction_mse"]), **dvf})
                _write_json(args.output_dir / f"metrics_phase_b_{phase_step}.json", metrics)
                state = _checkpoint_state(model=model, optimizer=optimizer, global_step=global_step, phase="B", protocol=protocol, pixel_protocol=pixel_protocol, observation_audit=observation_audit, curve=curve, initial_metrics=initial_metrics, phase_metrics=phase_metrics, snapshots=snapshots)
                _save_checkpoint(args.output_dir / f"checkpoint_phase_b_{phase_step}.pt", **state); _save_checkpoint(args.output_dir / "checkpoint_last.pt", **state)
        snapshots["B"] = parameter_snapshot(model); stage_audit["B"] = old_b1._snapshot_changed(snapshots["A"], snapshots["B"], [name for name in snapshots["A"] if name.startswith(_ENCODER_PREFIXES)])
        b_dvf = _aggregate_dvf(phase_metrics[f"B{args.phase_b_steps}"])
        return _finish(args, initial_metrics, phase_metrics, phase_arrays, snapshots, stage_audit, module_calls, a_dvf, b_dvf, curve, protocol, observation_audit, pixel_protocol, optimizer)
    finally:
        for hook in hooks: hook.remove()


def _resume(args) -> dict[str, Any]:
    if not args.resume_from.is_file():
        raise FileNotFoundError(args.resume_from)
    if not args.smoke and (not _formal_args_valid(args) or args.manifest is None or args.qc_table is None or args.canonical_domain is None or args.eval_every != 250):
        raise ValueError("formal Exp3-B1b resume requires the prescribed protocol")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = old_b1._load_checkpoint(args.resume_from)
    config, motion, observations, groups, observation_audit, pixel_protocol, model = _build_runtime(args)
    expected = {"phase_a": _phase_a_protocol(), "phase_b": _phase_b_protocol(), "no_phase_c": True}
    saved_protocol = dict(checkpoint["protocol"]); saved_schedule = saved_protocol.pop("phase_schedule", {})
    if saved_protocol != expected or saved_schedule.get("phase_a_steps") != args.phase_a_steps or int(saved_schedule.get("phase_b_steps", -1)) > args.phase_b_steps or checkpoint["observation_audit"] != observation_audit or checkpoint["pixel_protocol"] != pixel_protocol:
        raise ValueError("incompatible Exp3-B1b resume checkpoint")
    if checkpoint["phase"] not in {"A", "B"}:
        raise ValueError("B1b resume checkpoint must be from Phase A or B")
    model.load_state_dict(checkpoint["model"]); old_b1._restore_rng(checkpoint["rng_state"])
    phase_step_start = 0 if checkpoint["phase"] == "A" else int(checkpoint["global_step"]) - args.phase_a_steps
    if phase_step_start < 0 or phase_step_start > args.phase_b_steps:
        raise ValueError("resume checkpoint global step is incompatible with the requested Phase-B length")
    snapshots, curve, initial_metrics, phase_metrics = checkpoint["snapshots"], checkpoint["curve"], checkpoint["initial_metrics"], checkpoint["phase_metrics"]
    a_dvf = _aggregate_dvf(phase_metrics["A"]); stage_audit = {"A": old_b1._snapshot_changed(snapshots["start"], snapshots["A"], [name for name in snapshots["start"] if name.startswith(_LEVEL0)])}; phase_arrays: dict[str, dict[str, np.ndarray]] = {}
    module_calls = {"phase_a_encoder_calls": 0, "phase_b_encoder_calls": 0, "renderer_calls": 0, "respiratory_mbc_calls": 0, "cardiac_mbc_calls": 0, "uncertainty_calls": 0}

    def _count_encoder(_module, _args, _output): module_calls["phase_b_encoder_calls"] += 1
    def _count_renderer(_module, _args, _output): module_calls["renderer_calls"] += 1
    def _count_mbc(_module, _args, _output): module_calls["respiratory_mbc_calls"] += 1
    def _count_card(_module, _args, _output): module_calls["cardiac_mbc_calls"] += 1
    def _count_unc(_module, _args, _output): module_calls["uncertainty_calls"] += 1

    hooks = [model.film_encoder.register_forward_hook(_count_encoder), model.psf.register_forward_hook(_count_renderer), model.respiratory_mbc.register_forward_hook(_count_mbc), model.cardiac_mbc.register_forward_hook(_count_card), model.uncertainty.register_forward_hook(_count_unc)]
    try:
        optimizer = old_b1._configure_phase(model, "B", film_lr=5e-4, respiratory_mbc_lr=5e-4)
        if checkpoint["phase"] == "B":
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "B0" not in phase_metrics:
            phase_b_initial, arrays = _metrics(model, groups, motion, grid_shape=tuple(args.evaluation_grid_shape)); phase_metrics["B0"] = phase_b_initial; phase_arrays["phase_b_0"] = arrays; curve.append({"global_step": args.phase_a_steps, "phase": "B", "phase_step": 0, **_aggregate_dvf(phase_b_initial)}); _write_json(args.output_dir / "metrics_phase_b_0.json", phase_b_initial)
        for phase_step in range(phase_step_start + 1, args.phase_b_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            batch = old_b1._accumulate_full_batch(model, observations, phase="B", motion=None, pixel_samples=args.pixel_samples, accumulation_chunk_size=args.accumulation_chunk_size)
            frozen_grads = [name for name, parameter in model.named_parameters() if not name.startswith(old_b1._prefixes("B")) and parameter.grad is not None]
            if batch["observation_ids"] != [observation.dynamic_frame_id for observation in observations] or frozen_grads:
                raise RuntimeError("B1b Phase-B full-batch or frozen-gradient audit failed")
            if phase_step == phase_step_start + 1 and not old_b1._assert_expected_gradients(model, "B"):
                raise RuntimeError("B1b Phase-B expected-gradient connectivity audit failed")
            optimizer.step(); global_step = args.phase_a_steps + phase_step
            if phase_step % args.eval_every == 0 or phase_step == args.phase_b_steps:
                metrics, arrays = _metrics(model, groups, motion, grid_shape=tuple(args.evaluation_grid_shape)); phase_metrics[f"B{phase_step}"] = metrics; phase_arrays[f"phase_b_{phase_step}"] = arrays; dvf = _aggregate_dvf(metrics)
                curve.append({"global_step": global_step, "phase": "B", "phase_step": phase_step, "full_batch_reconstruction_mse": float(batch["full_batch_reconstruction_mse"]), **dvf}); _write_json(args.output_dir / f"metrics_phase_b_{phase_step}.json", metrics)
                state = _checkpoint_state(model=model, optimizer=optimizer, global_step=global_step, phase="B", protocol=checkpoint["protocol"], pixel_protocol=pixel_protocol, observation_audit=observation_audit, curve=curve, initial_metrics=initial_metrics, phase_metrics=phase_metrics, snapshots=snapshots)
                _save_checkpoint(args.output_dir / f"checkpoint_phase_b_{phase_step}.pt", **state); _save_checkpoint(args.output_dir / "checkpoint_last.pt", **state)
        snapshots["B"] = parameter_snapshot(model); stage_audit["B"] = old_b1._snapshot_changed(snapshots["A"], snapshots["B"], [name for name in snapshots["A"] if name.startswith(_ENCODER_PREFIXES)])
        b_dvf = _aggregate_dvf(phase_metrics[f"B{args.phase_b_steps}"])
        return _finish(args, initial_metrics, phase_metrics, phase_arrays, snapshots, stage_audit, module_calls, a_dvf, b_dvf, curve, checkpoint["protocol"], observation_audit, pixel_protocol, optimizer)
    finally:
        for hook in hooks: hook.remove()


def _finish(args, initial_metrics, phase_metrics, phase_arrays, snapshots, stage_audit, module_calls, a_dvf, b_dvf, curve, protocol, observation_audit, pixel_protocol, optimizer):
    phase_b_entered = b_dvf is not None
    if module_calls["phase_a_encoder_calls"] != 0 or module_calls["cardiac_mbc_calls"] != 0 or module_calls["uncertainty_calls"] != 0 or module_calls["renderer_calls"] <= 0 or (phase_b_entered and module_calls["phase_b_encoder_calls"] <= 0):
        raise RuntimeError("B1b runtime module-call audit failed")
    final_snapshot = snapshots["B" if phase_b_entered else "A"]
    frozen = {"film_encoder_card_unchanged": all(torch.equal(snapshots["start"][name], final_snapshot[name]) for name in snapshots["start"] if name.startswith("film_encoder.card.")), "canonical_unchanged": all(torch.equal(snapshots["start"][name], final_snapshot[name]) for name in snapshots["start"] if name.startswith("canonical.")), "respiratory_mbc_unchanged_during_b": None if not phase_b_entered else all(torch.equal(snapshots["A"][name], snapshots["B"][name]) for name in snapshots["A"] if name.startswith("respiratory_mbc.")), "cardiac_mbc_unchanged": all(torch.equal(snapshots["start"][name], final_snapshot[name]) for name in snapshots["start"] if name.startswith("cardiac_mbc.")), "uncertainty_unchanged": all(torch.equal(snapshots["start"][name], final_snapshot[name]) for name in snapshots["start"] if name.startswith("uncertainty."))}
    classification = _classify(a_dvf, b_dvf); final_metrics = phase_metrics.get(f"B{args.phase_b_steps}", phase_metrics["A"])
    _write_json(args.output_dir / "metrics_final.json", final_metrics); _write_json(args.output_dir / "stage_transition_audit.json", stage_audit)
    arrays = {"initial_dvf_metrics": np.asarray(list(_aggregate_dvf(initial_metrics).values())), "phase_a_dvf_metrics": np.asarray(list(a_dvf.values()))}
    if b_dvf is not None: arrays["phase_b_dvf_metrics"] = np.asarray(list(b_dvf.values()))
    np.savez_compressed(args.output_dir / "recovery_arrays.npz", **arrays)
    if curve:
        with (args.output_dir / "training_curve.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = sorted({key for row in curve for key in row}); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(curve)
    phase_b_rows = [row for row in curve if row.get("phase") == "B" and all(name in row for name in _PRIMARY)]
    first_escape = next((int(row["phase_step"]) for row in phase_b_rows if _escape(row)), None)
    first_strict = next((int(row["phase_step"]) for row in phase_b_rows if _strict(row)), None)
    historical_pair_available = False
    if phase_b_entered and args.existing_b1_curve is not None and args.phase_b_steps == 1000:
        old_metrics = _parse_existing_b1_phase_b_at_step(args.existing_b1_curve)
        new_reconstruction_mse = next((float(row["full_batch_reconstruction_mse"]) for row in reversed(phase_b_rows) if row.get("full_batch_reconstruction_mse") is not None), None)
        _write_paired_comparison(args.output_dir, final_metrics, new_reconstruction_mse, old_metrics)
        historical_pair_available = True
    factor = {"encoder_parameter_norm": _state_norm(final_snapshot, _ENCODER_PREFIXES), "respiratory_mbc_level0_parameter_norm": _state_norm(final_snapshot, _LEVEL0), "basis_field_metric_supported": False}
    _write_json(args.output_dir / "factor_diagnostics.json", factor)
    summary = {"experiment": "Exp3-B1b Good-MBC to Encoder Recovery", "output_dir": str(args.output_dir), "seed": args.seed, "scientific_protocol": protocol, "exposure_accounting": {"phase_a": _phase_a_protocol()["exposure_accounting"], "phase_b": _phase_b_protocol()["exposure_accounting"], "total_observation_exposures": 474000}, "observation_audit": observation_audit, "pixel_protocol": pixel_protocol, "phase_a": {"final_dvf": a_dvf, "escape": _escape(a_dvf)}, "phase_b": None if b_dvf is None else {"final_dvf": b_dvf, "escape": _escape(b_dvf), "strict": _strict(b_dvf), "first_phase_b_escape_step": first_escape, "first_phase_b_strict_step": first_strict}, "phase_b_entered": phase_b_entered, "historical_pair_available": historical_pair_available, "stage_parameter_change_audit": stage_audit, "frozen_module_audit": frozen, "runtime_audit": module_calls, **classification}
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs/source_first_change4_paperaligned.yaml"); parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml")
    parser.add_argument("--manifest", type=Path); parser.add_argument("--qc-table", type=Path); parser.add_argument("--canonical-domain", type=Path); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--existing-b1-curve", type=Path)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50)
    parser.add_argument("--phase-a-steps", type=int, default=750); parser.add_argument("--phase-b-steps", type=int, default=1000); parser.add_argument("--phase-a-observations-per-update", type=int, default=32); parser.add_argument("--pixel-mode", default="fixed_grid"); parser.add_argument("--pixel-samples", type=int, default=256); parser.add_argument("--accumulation-chunk-size", type=int, default=5); parser.add_argument("--eval-every", type=int, default=250); parser.add_argument("--checkpoint-every", type=int, default=250); parser.add_argument("--resume-from", type=Path); parser.add_argument("--evaluation-grid-shape", type=lambda value: tuple(map(int, value.split(","))), default=(8, 8, 8)); parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
