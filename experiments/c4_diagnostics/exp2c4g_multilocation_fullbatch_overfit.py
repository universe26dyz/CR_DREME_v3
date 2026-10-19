#!/usr/bin/env python3
"""Exp2-C4g: 9-location exact-full-batch direct score overfit probe."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]

import numpy as np
import torch
import yaml

from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, _load_domain, fallback_geometry, geometry_from_manifest
from experiments.c4_diagnostics.exp2_checkpoint_common import parameter_snapshot
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import _ENCODER_PREFIXES, _git_sha, _group_by_location, _prepare_output, _respiratory_only_motion, _set_seed, _validate_observations
from experiments.c4_diagnostics.exp2c4d_supervised_score_encoder_probe import _build_model, _configure_encoder_only, _direct_score_loss, _evaluate, _parameter_audit
from experiments.c4_diagnostics.synthetic.metrics import aggregate_location_metrics
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom


def _observation_audit(observations: list) -> dict[str, Any]:
    groups = _group_by_location(observations)
    counts = {location: len(sequence) for location, sequence in groups.items()}
    return {
        "n_locations": len(groups),
        "n_frames": len(observations),
        "frames_per_location": next(iter(counts.values())) if len(set(counts.values())) == 1 else None,
        "location_frame_counts": counts,
        "timestamps_strictly_increasing": all(all(second.timestamp_s > first.timestamp_s for first, second in zip(sequence, sequence[1:])) for sequence in groups.values()),
    }


def _accumulate_full_batch(model, observations: list, motion: GroundTruthMotion, *, accumulation_chunk_size: int) -> dict[str, Any]:
    """Backpropagate the exact mean score loss once across every observation."""
    if accumulation_chunk_size <= 0:
        raise ValueError("C4g accumulation chunk size must be positive")
    if not observations:
        raise ValueError("C4g full batch cannot be empty")
    total = observations[0].image.new_zeros(())
    observation_ids: list[int] = []
    scale = 1.0 / len(observations)
    for start in range(0, len(observations), accumulation_chunk_size):
        chunk_losses: list[torch.Tensor] = []
        for observation in observations[start : start + accumulation_chunk_size]:
            loss, _ = _direct_score_loss(model, observation, motion)
            chunk_losses.append(loss)
            total = total + loss.detach()
            observation_ids.append(int(observation.dynamic_frame_id))
        (torch.stack(chunk_losses).sum() * scale).backward()
    return {"full_batch_score_mse": total / len(observations), "observation_ids": observation_ids}


def _parameter_norm(model) -> float:
    parameters = [parameter.detach().reshape(-1) for name, parameter in model.named_parameters() if name.startswith(_ENCODER_PREFIXES)]
    return float(torch.cat(parameters).norm())


def _frozen_module_audit(changed: list[str]) -> dict[str, bool]:
    return {
        "film_encoder_card_unchanged": not any(name.startswith("film_encoder.card.") for name in changed),
        "canonical_unchanged": not any(name.startswith("canonical.") for name in changed),
        "respiratory_mbc_unchanged": not any(name.startswith("respiratory_mbc.") for name in changed),
        "cardiac_mbc_unchanged": not any(name.startswith("cardiac_mbc.") for name in changed),
        "uncertainty_unchanged": not any(name.startswith("uncertainty.") for name in changed),
    }


def _classify(metrics: dict[str, float]) -> dict[str, bool | str]:
    criteria = {
        "correlation_pass": metrics["gt_correlation"] >= .95,
        "temporal_rms_ratio_pass": .8 <= metrics["temporal_rms_ratio"] <= 1.2,
        "score_mse_pass": metrics["score_mse"] <= 2.0,
    }
    return {**criteria, "classification": "multilocation_fullbatch_recovery" if all(criteria.values()) else "multilocation_fullbatch_failure"}


def _empty_thresholds() -> dict[str, int | None]:
    return {name: None for name in (
        "first_corr_ge_0p90_step", "first_corr_ge_0p95_step", "first_corr_ge_0p99_step", "first_rms_ratio_ge_0p1_step", "first_rms_ratio_in_0p5_1p5_step", "first_rms_ratio_in_0p9_1p1_step", "first_score_mse_le_10_step", "first_score_mse_le_5_step", "first_score_mse_le_1_step", "first_score_mse_le_0p1_step", "first_joint_near_recovery_step",
    )}


def _update_thresholds(thresholds: dict[str, int | None] | None, metrics: dict[str, float], *, step: int) -> dict[str, int | None]:
    result = _empty_thresholds() if thresholds is None else dict(thresholds)
    conditions = {
        "first_corr_ge_0p90_step": metrics["gt_correlation"] >= .90,
        "first_corr_ge_0p95_step": metrics["gt_correlation"] >= .95,
        "first_corr_ge_0p99_step": metrics["gt_correlation"] >= .99,
        "first_rms_ratio_ge_0p1_step": metrics["temporal_rms_ratio"] >= .1,
        "first_rms_ratio_in_0p5_1p5_step": .5 <= metrics["temporal_rms_ratio"] <= 1.5,
        "first_rms_ratio_in_0p9_1p1_step": .9 <= metrics["temporal_rms_ratio"] <= 1.1,
        "first_score_mse_le_10_step": metrics["score_mse"] <= 10.,
        "first_score_mse_le_5_step": metrics["score_mse"] <= 5.,
        "first_score_mse_le_1_step": metrics["score_mse"] <= 1.,
        "first_score_mse_le_0p1_step": metrics["score_mse"] <= .1,
        "first_joint_near_recovery_step": metrics["gt_correlation"] >= .95 and .8 <= metrics["temporal_rms_ratio"] <= 1.2 and metrics["score_mse"] <= 2.,
    }
    for name, reached in conditions.items():
        if reached and result[name] is None:
            result[name] = step
    return result


def _exposure_accounting(*, steps: int, n_observations: int) -> dict[str, float | int]:
    return {
        "optimizer_steps": steps,
        "observations_per_optimizer_step": n_observations,
        "total_observation_exposures": steps * n_observations,
        "per_frame_exposures": steps,
        "c4d_nominal_comparison": {"steps": 750, "observations_per_update": 32, "total_observation_exposures": 24000, "average_per_frame_exposure": 24000 / 450},
    }


def _curve_row(step: int, metrics: dict[str, float], *, full_batch_score_mse: float, encoder_parameter_norm: float) -> dict[str, float | int]:
    names = ("gt_correlation", "temporal_rms_ratio", "predicted_score_temporal_rms", "gt_score_temporal_rms", "predicted_score_static_rms", "static_to_temporal_rms_ratio")
    return {"step": step, "full_batch_score_mse": full_batch_score_mse, **{name: metrics[name] for name in names}, "encoder_parameter_norm": encoder_parameter_norm}


def _write_curve(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _checkpoint_state(*, model, optimizer, step: int, protocol: dict[str, Any], observation_audit: dict[str, Any], thresholds: dict[str, int | None], curve: list[dict[str, float | int]], initial: dict[str, Any], initial_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "git_sha": _git_sha(), "protocol": protocol, "observation_audit": observation_audit, "thresholds": thresholds, "curve": curve, "initial_metrics": initial, "initial_arrays": {name: torch.from_numpy(value) for name, value in initial_arrays.items()}}


def _save_checkpoint(path: Path, **kwargs: Any) -> None:
    torch.save(_checkpoint_state(**kwargs), path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    checkpoint_every = int(getattr(args, "checkpoint_every", 1000)); resume_from = getattr(args, "resume_from", None)
    if args.steps <= 0 or args.eval_every <= 0 or args.accumulation_chunk_size <= 0 or checkpoint_every <= 0:
        raise ValueError("C4g steps, eval interval, accumulation chunk size, and checkpoint interval must be positive")
    if not args.smoke and (args.manifest is None or args.qc_table is None or args.canonical_domain is None):
        raise ValueError("formal C4g requires --manifest, --qc-table, and --canonical-domain")
    if resume_from is None:
        _prepare_output(args.output_dir, args.overwrite)
    elif args.overwrite or not args.output_dir.is_dir():
        raise ValueError("C4g resume requires an existing output directory without --overwrite")
    _set_seed(args.seed)
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); synthetic = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))["synthetic"]; domain = _load_domain(args.canonical_domain)
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(domain, respiratory_frequency_hz=float(synthetic["respiratory_frequency_hz"]), cardiac_frequency_hz=float(synthetic["cardiac_frequency_hz"]), respiratory_amplitude_mm=float(synthetic["respiratory_amplitude_mm"]), cardiac_amplitude_mm=float(synthetic["cardiac_amplitude_mm"])))
    if motion.cardiac_amplitude_mm != 0.0:
        raise RuntimeError("C4g target must have exactly zero cardiac amplitude")
    image_shape = (8, 8) if args.smoke else tuple(int(value) for value in synthetic["image_shape"]); psf_samples = 1 if args.smoke else int(synthetic["psf_samples"])
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=device) if args.manifest is not None else fallback_geometry(locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, image_shape=image_shape, motion=motion, device=device)
    observations = _build_observations(frames, image_shape=image_shape, phantom=AnalyticCanonicalPhantom(domain).to(device), motion=motion, psf_samples=psf_samples, seed=args.seed)
    groups = _validate_observations(observations, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, smoke=args.smoke)
    observation_audit = _observation_audit(observations)
    if not observation_audit["timestamps_strictly_increasing"]:
        raise RuntimeError("C4g requires strictly increasing timestamps within every location")
    if not args.smoke and (observation_audit["n_locations"] != 9 or observation_audit["n_frames"] != 450 or observation_audit["frames_per_location"] != 50):
        raise RuntimeError("formal C4g observation audit failed")
    model = _build_model(config, domain, n_dynamic_frames=len(observations), seed=args.seed, device=device)
    if model.film_implementation != "full_geometry_single_film":
        raise RuntimeError("C4g requires production GeometryFiLMMotionEncoder")
    learning_rate = float(config["training"]["optimizer"]["stage2"]["film_lr"])
    protocol = {"exact_full_batch": True, "accumulation_chunk_size": args.accumulation_chunk_size, "optimizer_step_after_all_observations": True, "uses_sampler": False, "direct_gt_score_mse": True, "original_geometry_preserved": True}
    optimizer = _configure_encoder_only(model, learning_rate=learning_rate)
    checkpoint: dict[str, Any] | None = None
    if resume_from is not None:
        checkpoint = torch.load(Path(resume_from), map_location=device, weights_only=True)
        saved_protocol = checkpoint.get("protocol", {})
        if checkpoint.get("observation_audit") != observation_audit or not all(saved_protocol.get(name) == protocol[name] for name in ("exact_full_batch", "optimizer_step_after_all_observations", "uses_sampler", "direct_gt_score_mse", "original_geometry_preserved")):
            raise ValueError("C4g resume checkpoint is incompatible with this observation protocol")
        if not 0 <= int(checkpoint.get("step", -1)) < args.steps:
            raise ValueError("C4g resume checkpoint step must be smaller than requested --steps")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
    before = parameter_snapshot(model)
    renderer_calls = [0]; mbc_calls = [0]
    renderer_hook = model.psf.register_forward_hook(lambda *_: renderer_calls.__setitem__(0, renderer_calls[0] + 1)); mbc_hook = model.respiratory_mbc.register_forward_hook(lambda *_: mbc_calls.__setitem__(0, mbc_calls[0] + 1))
    try:
        if checkpoint is None:
            initial_rows, initial_arrays, initial_by_view, initial_overall = _evaluate(model, groups, motion)
            initial = {"overall": initial_overall, "per_location": initial_rows, "by_location": aggregate_location_metrics(initial_rows), "by_view": initial_by_view}
            thresholds = _update_thresholds(None, initial_overall, step=0)
            curve = [_curve_row(0, initial_overall, full_batch_score_mse=initial_overall["score_mse"], encoder_parameter_norm=_parameter_norm(model))]
            _write_curve(args.output_dir / "training_curve.csv", curve)
            (args.output_dir / "metrics_initial.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8")
            start_step = 0
        else:
            initial = checkpoint["initial_metrics"]
            initial_arrays = {name: value.cpu().numpy() for name, value in checkpoint["initial_arrays"].items()}
            thresholds = checkpoint["thresholds"]; curve = checkpoint["curve"]; start_step = int(checkpoint["step"])
        for step in range(start_step + 1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            batch = _accumulate_full_batch(model, observations, motion, accumulation_chunk_size=args.accumulation_chunk_size)
            if batch["observation_ids"] != [observation.dynamic_frame_id for observation in observations]:
                raise RuntimeError("C4g full-batch pass did not use every observation exactly once")
            if step == start_step + 1:
                for prefix in _ENCODER_PREFIXES:
                    if not any(name.startswith(prefix) and parameter.grad is not None and torch.isfinite(parameter.grad).all() and float(parameter.grad.norm()) > 0.0 for name, parameter in model.named_parameters()):
                        raise RuntimeError(f"C4g direct score loss did not reach {prefix}")
                if any(parameter.grad is not None for name, parameter in model.named_parameters() if not name.startswith(_ENCODER_PREFIXES)):
                    raise RuntimeError("C4g direct score loss reached frozen parameters")
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                rows, _, by_view, overall = _evaluate(model, groups, motion)
                thresholds = _update_thresholds(thresholds, overall, step=step)
                curve.append(_curve_row(step, overall, full_batch_score_mse=float(batch["full_batch_score_mse"]), encoder_parameter_norm=_parameter_norm(model)))
                _write_curve(args.output_dir / "training_curve.csv", curve)
            if step % checkpoint_every == 0:
                checkpoint_path = args.output_dir / f"checkpoint_{step}.pt"
                checkpoint_kwargs = {"model": model, "optimizer": optimizer, "step": step, "protocol": protocol, "observation_audit": observation_audit, "thresholds": thresholds, "curve": curve, "initial": initial, "initial_arrays": initial_arrays}
                _save_checkpoint(checkpoint_path, **checkpoint_kwargs); _save_checkpoint(args.output_dir / "checkpoint_last.pt", **checkpoint_kwargs)
        final_rows, final_arrays, final_by_view, final_overall = _evaluate(model, groups, motion)
    finally:
        renderer_hook.remove(); mbc_hook.remove()
    if renderer_calls[0] != 0 or mbc_calls[0] != 0:
        raise RuntimeError("C4g used renderer or learned respiratory MBC")
    changed = _parameter_audit(model, before); frozen_audit = _frozen_module_audit(changed)
    if not all(frozen_audit.values()):
        raise RuntimeError("C4g frozen module audit failed")
    final = {"overall": final_overall, "per_location": final_rows, "by_location": aggregate_location_metrics(final_rows), "by_view": final_by_view}
    classification = _classify(final_overall); exposure_accounting = _exposure_accounting(steps=args.steps, n_observations=len(observations))
    _write_curve(args.output_dir / "training_curve.csv", curve)
    (args.output_dir / "metrics_initial.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "metrics_final.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output_dir / "recovery_arrays.npz", **initial_arrays, **{f"final_{name}": value for name, value in final_arrays.items()})
    _save_checkpoint(args.output_dir / "checkpoint_last.pt", model=model, optimizer=optimizer, step=args.steps, protocol=protocol, observation_audit=observation_audit, thresholds=thresholds, curve=curve, initial=initial, initial_arrays=initial_arrays)
    summary = {
        "experiment": "Exp2-C4g multi-location exact-full-batch encoder overfit", "git_sha": _git_sha(), "seed": args.seed, "supervision": "direct_gt_respiratory_score", "model": "GeometryFiLMMotionEncoder", "film_implementation": model.film_implementation,
        "observation_audit": observation_audit, "original_geometry_preserved": True, "multi_location_images_preserved": True, "multi_location_timestamps_preserved": True, "multi_location_gt_scores_preserved": True,
        "trainable_modules": list(_ENCODER_PREFIXES), "film_lr": learning_rate, "steps": args.steps, "eval_every": args.eval_every, "checkpoint_every": checkpoint_every, "resumed_from": None if resume_from is None else str(resume_from), "full_batch_protocol": protocol, "exposure_accounting": exposure_accounting,
        "renderer_used": False, "mbc_used": False, "renderer_calls": renderer_calls[0], "learned_respiratory_mbc_calls": mbc_calls[0], "parameters_changed": changed, "frozen_module_audit": frozen_audit,
        "threshold_steps": thresholds, **classification, "initial_metrics": initial, "final_metrics": final,
        "interpretation_contract": "Full-batch recovery supports optimization-budget/sampling dynamics as the main C4d/C4f confound; failure after 10000 exact full-batch steps supports shared multi-location representation or optimization conflict as a primary blocker.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2-C4g multi-location exact-full-batch encoder overfit")
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs/source_first_change4_paperaligned.yaml"); parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml"); parser.add_argument("--manifest", type=Path); parser.add_argument("--qc-table", type=Path); parser.add_argument("--canonical-domain", type=Path); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50); parser.add_argument("--steps", type=int, default=10000); parser.add_argument("--eval-every", type=int, default=100); parser.add_argument("--accumulation-chunk-size", type=int, default=25); parser.add_argument("--checkpoint-every", type=int, default=1000); parser.add_argument("--resume-from", type=Path); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); summary = run(args); print(f"Exp2-C4g: {args.output_dir / 'summary.json'} ({summary['observation_audit']['n_locations']} locations)")


if __name__ == "__main__":
    main()
