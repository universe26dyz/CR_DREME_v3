#!/usr/bin/env python3
"""Exp2-C4h: oracle-field, exact-full-batch reconstruction encoder probe."""
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

from cardioresp4d.training.model import SourceFirstDynamicModel
from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, _load_domain, fallback_geometry, geometry_from_manifest
from experiments.c4_diagnostics.exp2_checkpoint_common import parameter_snapshot
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import _ENCODER_PREFIXES, _OracleRespiratoryMBC, _assert_only_encoder_changed, _assert_oracle_matches_gt, _build_intervention_model, _configure_encoder_only, _evaluate, _full_image_uv, _git_sha, _group_by_location, _intervention_loss_terms, _prepare_output, _respiratory_only_motion, _set_seed, _validate_observations
from experiments.c4_diagnostics.exp2c4d_supervised_score_encoder_probe import _pooled_score_values
from experiments.c4_diagnostics.synthetic.metrics import aggregate_location_metrics
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom


def _observation_audit(observations: list) -> dict[str, Any]:
    groups = _group_by_location(observations)
    counts = {location: len(sequence) for location, sequence in groups.items()}
    return {"n_locations": len(groups), "n_frames": len(observations), "frames_per_location": next(iter(counts.values())) if len(set(counts.values())) == 1 else None, "location_frame_counts": counts, "timestamps_strictly_increasing": all(all(second.timestamp_s > first.timestamp_s for first, second in zip(sequence, sequence[1:])) for sequence in groups.values())}


def _fixed_pixel_grid(observation, *, count: int) -> torch.Tensor:
    """A deterministic square grid spanning the acquired image FOV."""
    height, width = observation.image.shape[-2:]
    effective_count = min(count, height * width)
    side = math.isqrt(effective_count)
    if count <= 0 or side * side != effective_count:
        raise ValueError("C4h fixed pixel grid requires a positive square pixel count after smoke adaptation")
    x = torch.linspace(0, width - 1, side, device=observation.image.device).round().long()
    y = torch.linspace(0, height - 1, side, device=observation.image.device).round().long()
    uv = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1).reshape(-1, 2)
    if uv.shape[0] != effective_count or torch.unique(uv, dim=0).shape[0] != effective_count:
        raise RuntimeError("C4h fixed grid must contain unique FOV-spanning pixels")
    return uv.to(observation.image.dtype)


def _pixel_uv(observation, *, pixel_mode: str, pixel_samples: int) -> torch.Tensor:
    if pixel_mode == "fixed_grid":
        return _fixed_pixel_grid(observation, count=pixel_samples)
    if pixel_mode == "full_image":
        return _full_image_uv(observation)
    raise ValueError(f"unsupported C4h pixel mode {pixel_mode!r}")


def _pixel_protocol(observation, *, pixel_mode: str, pixel_samples: int) -> dict[str, Any]:
    uv = _pixel_uv(observation, pixel_mode=pixel_mode, pixel_samples=pixel_samples)
    return {"pixel_mode": pixel_mode, "requested_pixel_samples": pixel_samples, "actual_pixel_count": int(uv.shape[0]), "fixed_uv": uv.detach().cpu().long().tolist()}


def _accumulate_full_batch(model, observations: list, oracle: _OracleRespiratoryMBC, *, pixel_mode: str, pixel_samples: int, accumulation_chunk_size: int) -> dict[str, Any]:
    if accumulation_chunk_size <= 0 or not observations:
        raise ValueError("C4h requires a positive accumulation chunk size and non-empty observations")
    total = observations[0].image.new_zeros(())
    observation_ids: list[int] = []
    scale = 1.0 / len(observations)
    for start in range(0, len(observations), accumulation_chunk_size):
        chunk_losses: list[torch.Tensor] = []
        for observation in observations[start : start + accumulation_chunk_size]:
            terms = _intervention_loss_terms(model, observation, oracle, pixel_uv=_pixel_uv(observation, pixel_mode=pixel_mode, pixel_samples=pixel_samples))
            chunk_losses.append(terms["data_mse"]); total = total + terms["data_mse"].detach(); observation_ids.append(int(observation.dynamic_frame_id))
        (torch.stack(chunk_losses).sum() * scale).backward()
    return {"full_batch_reconstruction_mse": total / len(observations), "observation_ids": observation_ids}


@torch.inference_mode()
def _full_batch_reconstruction_mse(model, observations: list, oracle: _OracleRespiratoryMBC, *, pixel_mode: str, pixel_samples: int) -> float:
    return float(torch.stack([_intervention_loss_terms(model, observation, oracle, pixel_uv=_pixel_uv(observation, pixel_mode=pixel_mode, pixel_samples=pixel_samples))["data_mse"] for observation in observations]).mean())


def _score_pooling(groups: dict[str, list], arrays: dict[str, np.ndarray]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    all_scores: list[tuple[torch.Tensor, torch.Tensor]] = []; by_view: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for location, sequence in groups.items():
        stem = location.replace("/", "_")
        scores = (torch.from_numpy(arrays[f"{stem}_predicted_scores"]), torch.from_numpy(arrays[f"{stem}_gt_scores"]))
        all_scores.append(scores); by_view.setdefault(sequence[0].view, []).append(scores)
    return _pooled_score_values(all_scores), {view: _pooled_score_values(scores) for view, scores in by_view.items()}


def _metrics(model, groups: dict[str, list], motion: GroundTruthMotion, oracle: _OracleRespiratoryMBC, *, grid_shape: tuple[int, int, int]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    rows, arrays = _evaluate(model, groups, motion, oracle, grid_shape=grid_shape)
    overall, by_view = _score_pooling(groups, arrays)
    return {"per_location": rows, "aggregate": aggregate_location_metrics(rows), "overall_score_metrics": overall, "by_view_score_metrics": by_view}, arrays


def _parameter_norm(model) -> float:
    return float(torch.cat([parameter.detach().reshape(-1) for name, parameter in model.named_parameters() if name.startswith(_ENCODER_PREFIXES)]).norm())


def _frozen_module_audit(changed: list[str]) -> dict[str, bool]:
    return {"film_encoder_card_unchanged": not any(name.startswith("film_encoder.card.") for name in changed), "canonical_unchanged": not any(name.startswith("canonical.") for name in changed), "respiratory_mbc_unchanged": not any(name.startswith("respiratory_mbc.") for name in changed), "cardiac_mbc_unchanged": not any(name.startswith("cardiac_mbc.") for name in changed), "uncertainty_unchanged": not any(name.startswith("uncertainty.") for name in changed)}


def _classify(metrics: dict[str, float]) -> dict[str, bool | str]:
    criteria = {"correlation_pass": metrics["gt_correlation"] >= .95, "temporal_rms_ratio_pass": .8 <= metrics["temporal_rms_ratio"] <= 1.2, "score_mse_pass": metrics["score_mse"] <= 2.0}
    return {**criteria, "classification": "reconstruction_fullbatch_recovery" if all(criteria.values()) else "reconstruction_fullbatch_failure"}


def _empty_thresholds() -> dict[str, int | None]:
    return {name: None for name in ("first_corr_ge_0p90_step", "first_corr_ge_0p95_step", "first_corr_ge_0p99_step", "first_rms_ratio_ge_0p1_step", "first_rms_ratio_in_0p5_1p5_step", "first_rms_ratio_in_0p9_1p1_step", "first_score_mse_le_10_step", "first_score_mse_le_5_step", "first_score_mse_le_2_step", "first_score_mse_le_1_step", "first_joint_near_recovery_step")}


def _update_thresholds(thresholds: dict[str, int | None] | None, metrics: dict[str, float], *, step: int) -> dict[str, int | None]:
    result = _empty_thresholds() if thresholds is None else dict(thresholds)
    conditions = {"first_corr_ge_0p90_step": metrics["gt_correlation"] >= .90, "first_corr_ge_0p95_step": metrics["gt_correlation"] >= .95, "first_corr_ge_0p99_step": metrics["gt_correlation"] >= .99, "first_rms_ratio_ge_0p1_step": metrics["temporal_rms_ratio"] >= .1, "first_rms_ratio_in_0p5_1p5_step": .5 <= metrics["temporal_rms_ratio"] <= 1.5, "first_rms_ratio_in_0p9_1p1_step": .9 <= metrics["temporal_rms_ratio"] <= 1.1, "first_score_mse_le_10_step": metrics["score_mse"] <= 10., "first_score_mse_le_5_step": metrics["score_mse"] <= 5., "first_score_mse_le_2_step": metrics["score_mse"] <= 2., "first_score_mse_le_1_step": metrics["score_mse"] <= 1., "first_joint_near_recovery_step": metrics["gt_correlation"] >= .95 and .8 <= metrics["temporal_rms_ratio"] <= 1.2 and metrics["score_mse"] <= 2.}
    for name, reached in conditions.items():
        if reached and result[name] is None:
            result[name] = step
    return result


def _exposure_accounting(*, steps: int, n_observations: int, pixels_per_observation: int) -> dict[str, float | int]:
    return {"optimizer_steps": steps, "observations_per_optimizer_step": n_observations, "pixels_per_observation": pixels_per_observation, "observation_exposures": steps * n_observations, "pixel_observation_exposures": steps * n_observations * pixels_per_observation, "per_frame_exposures": steps, "c4b_nominal_comparison": {"steps": 750, "observations_per_update": 32, "pixel_samples": 256, "total_observation_exposures": 24000, "average_frame_exposure": 24000 / 450}}


def _curve_row(step: int, metrics: dict[str, float], *, full_batch_reconstruction_mse: float, encoder_parameter_norm: float) -> dict[str, float | int]:
    names = ("gt_correlation", "temporal_rms_ratio", "predicted_score_temporal_rms", "gt_score_temporal_rms", "predicted_score_static_rms", "static_to_temporal_rms_ratio", "score_mse")
    return {"step": step, "full_batch_reconstruction_mse": full_batch_reconstruction_mse, **{name: metrics[name] for name in names}, "encoder_parameter_norm": encoder_parameter_norm}


def _write_curve(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _checkpoint_state(*, model, optimizer, step: int, protocol: dict[str, Any], pixel_protocol: dict[str, Any], observation_audit: dict[str, Any], thresholds: dict[str, int | None], curve: list[dict[str, float | int]], initial: dict[str, Any], initial_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    rng_state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "git_sha": _git_sha(), "protocol": protocol, "pixel_protocol": pixel_protocol, "observation_audit": observation_audit, "thresholds": thresholds, "curve": curve, "initial_metrics": initial, "initial_arrays": {name: torch.from_numpy(value) for name, value in initial_arrays.items()}, "rng_state": rng_state}


def _save_checkpoint(path: Path, **kwargs: Any) -> None:
    torch.save(_checkpoint_state(**kwargs), path)


def _restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device); checkpoint_every = int(getattr(args, "checkpoint_every", 1000)); resume_from = getattr(args, "resume_from", None)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if args.steps <= 0 or args.eval_every <= 0 or args.accumulation_chunk_size <= 0 or args.pixel_samples <= 0 or checkpoint_every <= 0:
        raise ValueError("C4h training, pixel, accumulation, and checkpoint arguments must be positive")
    if not args.smoke and (args.manifest is None or args.qc_table is None or args.canonical_domain is None):
        raise ValueError("formal C4h requires --manifest, --qc-table, and --canonical-domain")
    if not args.smoke and (args.pixel_mode != "fixed_grid" or args.pixel_samples != 256):
        raise ValueError("formal C4h primary protocol requires --pixel-mode fixed_grid --pixel-samples 256")
    if resume_from is None:
        _prepare_output(args.output_dir, args.overwrite)
    elif args.overwrite or not args.output_dir.is_dir():
        raise ValueError("C4h resume requires an existing output directory without --overwrite")
    _set_seed(args.seed)
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); synthetic = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))["synthetic"]; domain = _load_domain(args.canonical_domain)
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(domain, respiratory_frequency_hz=float(synthetic["respiratory_frequency_hz"]), cardiac_frequency_hz=float(synthetic["cardiac_frequency_hz"]), respiratory_amplitude_mm=float(synthetic["respiratory_amplitude_mm"]), cardiac_amplitude_mm=float(synthetic["cardiac_amplitude_mm"])))
    if motion.cardiac_amplitude_mm != 0.0:
        raise RuntimeError("C4h target must have exactly zero cardiac amplitude")
    image_shape = (8, 8) if args.smoke else tuple(int(value) for value in synthetic["image_shape"]); psf_samples = 1 if args.smoke else int(synthetic["psf_samples"])
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=device) if args.manifest is not None else fallback_geometry(locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, image_shape=image_shape, motion=motion, device=device)
    observations = _build_observations(frames, image_shape=image_shape, phantom=AnalyticCanonicalPhantom(domain).to(device), motion=motion, psf_samples=psf_samples, seed=args.seed)
    groups = _validate_observations(observations, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, smoke=args.smoke); observation_audit = _observation_audit(observations)
    if not observation_audit["timestamps_strictly_increasing"] or (not args.smoke and (observation_audit["n_locations"] != 9 or observation_audit["n_frames"] != 450 or observation_audit["frames_per_location"] != 50)):
        raise RuntimeError("C4h formal observation audit failed")
    pixel_protocol = _pixel_protocol(observations[0], pixel_mode=args.pixel_mode, pixel_samples=args.pixel_samples)
    model = _build_intervention_model(config, domain, n_dynamic_frames=len(observations), seed=args.seed, device=device)
    if model.film_implementation != "full_geometry_single_film":
        raise RuntimeError("C4h requires production GeometryFiLMMotionEncoder")
    oracle = _OracleRespiratoryMBC(motion).to(device); oracle_grid, _ = SourceFirstDynamicModel._physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, (3, 3, 3)); _assert_oracle_matches_gt(oracle, motion, oracle_grid)
    learning_rate = float(config["training"]["optimizer"]["stage2"]["film_lr"]); optimizer = _configure_encoder_only(model, learning_rate=learning_rate)
    protocol = {"exact_full_batch": True, "optimizer_step_after_all_observations": True, "uses_sampler": False, "objective": "reconstruction_data_mse", "oracle_field": "c4b_gt_respiratory"}
    checkpoint: dict[str, Any] | None = None
    if resume_from is not None:
        checkpoint = torch.load(Path(resume_from), map_location=device, weights_only=True)
        if checkpoint.get("observation_audit") != observation_audit or checkpoint.get("pixel_protocol") != pixel_protocol or checkpoint.get("protocol") != protocol:
            raise ValueError("C4h resume checkpoint is incompatible with observations, fixed pixels, oracle, or full-batch protocol")
        if not 0 <= int(checkpoint.get("step", -1)) < args.steps:
            raise ValueError("C4h resume checkpoint step must be smaller than requested --steps")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"]); _restore_rng_state(checkpoint["rng_state"])
    before = parameter_snapshot(model); renderer_calls = [0]; training_renderer_calls = [0]; mbc_calls = [0]
    renderer_hook = model.psf.register_forward_hook(lambda *_: renderer_calls.__setitem__(0, renderer_calls[0] + 1)); mbc_hook = model.respiratory_mbc.register_forward_hook(lambda *_: mbc_calls.__setitem__(0, mbc_calls[0] + 1))
    try:
        if checkpoint is None:
            initial, initial_arrays = _metrics(model, groups, motion, oracle, grid_shape=tuple(args.evaluation_grid_shape)); thresholds = _update_thresholds(None, initial["overall_score_metrics"], step=0)
            curve = [_curve_row(0, initial["overall_score_metrics"], full_batch_reconstruction_mse=_full_batch_reconstruction_mse(model, observations, oracle, pixel_mode=args.pixel_mode, pixel_samples=args.pixel_samples), encoder_parameter_norm=_parameter_norm(model))]
            _write_curve(args.output_dir / "training_curve.csv", curve); (args.output_dir / "metrics_initial.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8"); start_step = 0
        else:
            initial = checkpoint["initial_metrics"]; initial_arrays = {name: value.cpu().numpy() for name, value in checkpoint["initial_arrays"].items()}; thresholds = checkpoint["thresholds"]; curve = checkpoint["curve"]; start_step = int(checkpoint["step"])
        for step in range(start_step + 1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True); renderer_calls_before_batch = renderer_calls[0]; batch = _accumulate_full_batch(model, observations, oracle, pixel_mode=args.pixel_mode, pixel_samples=args.pixel_samples, accumulation_chunk_size=args.accumulation_chunk_size); training_renderer_calls[0] += renderer_calls[0] - renderer_calls_before_batch
            if batch["observation_ids"] != [observation.dynamic_frame_id for observation in observations]:
                raise RuntimeError("C4h full-batch pass did not use every observation exactly once")
            if step == start_step + 1:
                for prefix in _ENCODER_PREFIXES:
                    if not any(name.startswith(prefix) and parameter.grad is not None and torch.isfinite(parameter.grad).all() and float(parameter.grad.norm()) > 0.0 for name, parameter in model.named_parameters()):
                        raise RuntimeError(f"C4h reconstruction loss did not reach {prefix}")
                if any(parameter.grad is not None for name, parameter in model.named_parameters() if not name.startswith(_ENCODER_PREFIXES)):
                    raise RuntimeError("C4h reconstruction backward reached frozen parameters")
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                evaluated, _, = _metrics(model, groups, motion, oracle, grid_shape=tuple(args.evaluation_grid_shape)); thresholds = _update_thresholds(thresholds, evaluated["overall_score_metrics"], step=step)
                curve.append(_curve_row(step, evaluated["overall_score_metrics"], full_batch_reconstruction_mse=float(batch["full_batch_reconstruction_mse"]), encoder_parameter_norm=_parameter_norm(model))); _write_curve(args.output_dir / "training_curve.csv", curve)
            if step % checkpoint_every == 0:
                state = {"model": model, "optimizer": optimizer, "step": step, "protocol": protocol, "pixel_protocol": pixel_protocol, "observation_audit": observation_audit, "thresholds": thresholds, "curve": curve, "initial": initial, "initial_arrays": initial_arrays}
                _save_checkpoint(args.output_dir / f"checkpoint_{step}.pt", **state); _save_checkpoint(args.output_dir / "checkpoint_last.pt", **state)
        final, final_arrays = _metrics(model, groups, motion, oracle, grid_shape=tuple(args.evaluation_grid_shape))
    finally:
        renderer_hook.remove(); mbc_hook.remove()
    if renderer_calls[0] <= 0 or training_renderer_calls[0] <= 0 or mbc_calls[0] != 0:
        raise RuntimeError("C4h renderer/oracle MBC audit failed")
    changed, unchanged = _assert_only_encoder_changed(model, before); frozen_audit = _frozen_module_audit(changed)
    if not all(frozen_audit.values()):
        raise RuntimeError("C4h frozen module audit failed")
    _write_curve(args.output_dir / "training_curve.csv", curve); (args.output_dir / "metrics_initial.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8"); (args.output_dir / "metrics_final.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output_dir / "recovery_arrays.npz", **initial_arrays, **{f"final_{name}": value for name, value in final_arrays.items()})
    _save_checkpoint(args.output_dir / "checkpoint_last.pt", model=model, optimizer=optimizer, step=args.steps, protocol=protocol, pixel_protocol=pixel_protocol, observation_audit=observation_audit, thresholds=thresholds, curve=curve, initial=initial, initial_arrays=initial_arrays)
    classification = _classify(final["overall_score_metrics"])
    summary = {"experiment": "Exp2-C4h oracle-field full-batch reconstruction encoder probe", "git_sha": _git_sha(), "seed": args.seed, "observation_audit": observation_audit, "oracle_spatial_field": "C4b analytic GT respiratory basis", "oracle_validation": "ScoreWeightedMBCField(oracle, gt_score) == analytic respiratory_dvf_mm", "objective": "reconstruction_image_data_mse_only", "model": "GeometryFiLMMotionEncoder", "film_implementation": model.film_implementation, "pixel_protocol": pixel_protocol, "trainable_modules": list(_ENCODER_PREFIXES), "film_lr": learning_rate, "steps": args.steps, "eval_every": args.eval_every, "accumulation_chunk_size": args.accumulation_chunk_size, "checkpoint_every": checkpoint_every, "resumed_from": None if resume_from is None else str(resume_from), "full_batch_protocol": protocol, "exposure_accounting": _exposure_accounting(steps=args.steps, n_observations=len(observations), pixels_per_observation=pixel_protocol["actual_pixel_count"]), "renderer_calls": renderer_calls[0], "training_renderer_calls": training_renderer_calls[0], "learned_respiratory_mbc_calls": mbc_calls[0], "parameters_changed": changed, "parameters_unchanged": unchanged, "frozen_module_audit": frozen_audit, "threshold_steps": thresholds, **classification, "initial_metrics": initial, "final_metrics": final, "interpretation_contract": "PASS supports reconstruction-driven encoder recovery under long deterministic full-batch optimization; FAIL with C4g recovery isolates reconstruction-loss-to-encoder optimization conditioning rather than encoder capacity."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2-C4h oracle-field exact-full-batch reconstruction encoder probe")
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs/source_first_change4_paperaligned.yaml"); parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml"); parser.add_argument("--manifest", type=Path); parser.add_argument("--qc-table", type=Path); parser.add_argument("--canonical-domain", type=Path); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50); parser.add_argument("--steps", type=int, default=10000); parser.add_argument("--eval-every", type=int, default=250); parser.add_argument("--pixel-mode", choices=("fixed_grid", "full_image"), default="fixed_grid"); parser.add_argument("--pixel-samples", type=int, default=256); parser.add_argument("--accumulation-chunk-size", type=int, default=5); parser.add_argument("--checkpoint-every", type=int, default=1000); parser.add_argument("--resume-from", type=Path); parser.add_argument("--evaluation-grid-shape", type=lambda value: tuple(int(item) for item in value.split(",")), default=(8, 8, 8)); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); summary = run(args); print(f"Exp2-C4h: {args.output_dir / 'summary.json'} ({summary['observation_audit']['n_locations']} locations)")


if __name__ == "__main__":
    main()
