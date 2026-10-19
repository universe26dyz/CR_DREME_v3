#!/usr/bin/env python3
"""Exp2-C4e: single-location direct GT-score overfit probe.

PASS means the baseline GeometryFiLMMotionEncoder can fit the respiratory
mapping at one fixed location; paired with C4d failure, that implicates
multi-location sharing/geometry conditioning rather than image-path capacity.
FAIL means this direct full-batch overfit probe cannot fit even one location.
"""
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
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import _ENCODER_PREFIXES, _git_sha, _prepare_output, _respiratory_only_motion, _score_values, _set_seed, _validate_observations
from experiments.c4_diagnostics.exp2c4d_supervised_score_encoder_probe import _build_model, _component_values, _configure_encoder_only, _direct_score_loss, _gt_score, _parameter_audit
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom


def _select_location(observations: list, *, view: str, slice_id: str | None, frames_per_location: int) -> tuple[list, dict[str, Any]]:
    candidates: dict[tuple[str, str], list] = {}
    for observation in observations:
        if observation.view == view and (slice_id is None or observation.slice_id == slice_id):
            candidates.setdefault((observation.view, observation.slice_id), []).append(observation)
    if not candidates:
        raise ValueError(f"C4e found no location for view={view!r}, slice_id={slice_id!r}")
    selected_key = sorted(candidates)[0]
    sequence = sorted(candidates[selected_key], key=lambda item: item.timestamp_s)
    if len(sequence) != frames_per_location or any(second.timestamp_s <= first.timestamp_s for first, second in zip(sequence, sequence[1:])):
        raise ValueError("C4e selected location does not have the requested strictly increasing frame sequence")
    reference = sequence[0]
    tensor_fields = ("center_mm", "row_direction", "column_direction", "normal", "pixel_spacing_mm")
    fields_constant = {name: all(torch.allclose(getattr(item, name), getattr(reference, name), atol=1e-6, rtol=0.) for item in sequence[1:]) for name in tensor_fields}
    fields_constant["slice_thickness_mm"] = all(abs(item.slice_thickness_mm - reference.slice_thickness_mm) <= 1e-6 for item in sequence[1:])
    if not all(fields_constant.values()):
        raise ValueError("C4e selected location geometry is not constant")
    return sequence, {"selected_view": selected_key[0], "selected_slice_id": selected_key[1], "n_frames": len(sequence), "timestamp_min_s": sequence[0].timestamp_s, "timestamp_max_s": sequence[-1].timestamp_s, "geometry_constant": True, "geometry_fields_constant": fields_constant}


@torch.inference_mode()
def _metrics(model, sequence: list, motion: GroundTruthMotion) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    prediction = torch.cat([model.encode_motion_scores(observation)["resp_scores"][:, :1, :] for observation in sequence])
    target = torch.cat([_gt_score(motion, observation.timestamp_s, device=prediction.device, dtype=prediction.dtype) for observation in sequence])
    values = {**_score_values(prediction, target), **_component_values(prediction, target), "prediction_min": float(prediction.min()), "prediction_max": float(prediction.max()), "target_min": float(target.min()), "target_max": float(target.max()), "prediction_peak_to_peak": float(prediction.max() - prediction.min()), "target_peak_to_peak": float(target.max() - target.min())}
    return values, {"timestamps_s": np.asarray([observation.timestamp_s for observation in sequence]), "predicted_scores": prediction.cpu().numpy(), "gt_scores": target.cpu().numpy()}


def _classify(metrics: dict[str, float]) -> dict[str, bool | str]:
    criteria = {"correlation_pass": metrics["gt_correlation"] >= .99, "temporal_rms_ratio_pass": .90 <= metrics["temporal_rms_ratio"] <= 1.10, "score_mse_pass": metrics["score_mse"] <= .01}
    return {**criteria, "classification": "single_location_overfit_pass" if all(criteria.values()) else "single_location_overfit_fail"}


def _curve_row(step: int, metrics: dict[str, float], parameter_norm: float, raw_prediction_rms: float) -> dict[str, float | int]:
    names = ("score_mse", "gt_correlation", "predicted_score_temporal_rms", "gt_score_temporal_rms", "temporal_rms_ratio", "predicted_score_static_rms", "static_to_temporal_rms_ratio", "x_gt_correlation", "y_gt_correlation", "z_gt_correlation", "x_temporal_rms_ratio", "y_temporal_rms_ratio", "z_temporal_rms_ratio")
    return {"step": step, **{name: metrics[name] for name in names}, "encoder_parameter_norm": parameter_norm, "raw_prediction_rms": raw_prediction_rms}


def _write_curve(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if not args.smoke and (args.manifest is None or args.qc_table is None or args.canonical_domain is None):
        raise ValueError("formal C4e requires --manifest, --qc-table, and --canonical-domain")
    if args.steps <= 0 or args.eval_every <= 0:
        raise ValueError("C4e steps and eval interval must be positive")
    _prepare_output(args.output_dir, args.overwrite); _set_seed(args.seed)
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); synthetic = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))["synthetic"]; domain = _load_domain(args.canonical_domain)
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(domain, respiratory_frequency_hz=float(synthetic["respiratory_frequency_hz"]), cardiac_frequency_hz=float(synthetic["cardiac_frequency_hz"]), respiratory_amplitude_mm=float(synthetic["respiratory_amplitude_mm"]), cardiac_amplitude_mm=float(synthetic["cardiac_amplitude_mm"])))
    if motion.cardiac_amplitude_mm != 0.0:
        raise RuntimeError("C4e target must have exactly zero cardiac amplitude")
    image_shape = (8, 8) if args.smoke else tuple(int(value) for value in synthetic["image_shape"]); psf_samples = 1 if args.smoke else int(synthetic["psf_samples"])
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=device) if args.manifest is not None else fallback_geometry(locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, image_shape=image_shape, motion=motion, device=device)
    observations = _build_observations(frames, image_shape=image_shape, phantom=AnalyticCanonicalPhantom(domain).to(device), motion=motion, psf_samples=psf_samples, seed=args.seed)
    _validate_observations(observations, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, smoke=args.smoke)
    sequence, geometry_audit = _select_location(observations, view=args.view, slice_id=args.slice_id, frames_per_location=args.frames_per_location)
    model = _build_model(config, domain, n_dynamic_frames=len(observations), seed=args.seed, device=device)
    if model.film_implementation != "full_geometry_single_film":
        raise RuntimeError("C4e requires production GeometryFiLMMotionEncoder")
    learning_rate = float(config["training"]["optimizer"]["stage2"]["film_lr"]) if args.film_lr is None else args.film_lr
    optimizer = _configure_encoder_only(model, learning_rate=learning_rate); before = parameter_snapshot(model)
    renderer_calls = [0]; mbc_calls = [0]; renderer_hook = model.psf.register_forward_hook(lambda *_: renderer_calls.__setitem__(0, renderer_calls[0] + 1)); mbc_hook = model.respiratory_mbc.register_forward_hook(lambda *_: mbc_calls.__setitem__(0, mbc_calls[0] + 1))
    thresholds: dict[str, int | None] = {"first_corr_ge_0p99_step": None, "first_rms_ratio_in_0p9_1p1_step": None, "first_score_mse_le_0p01_step": None, "first_joint_success_step": None}
    try:
        initial_metrics, initial_arrays = _metrics(model, sequence, motion); curve: list[dict[str, float | int]] = []
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True); losses_and_scores = [_direct_score_loss(model, observation, motion) for observation in sequence]; loss = torch.stack([item[0] for item in losses_and_scores]).mean(); loss.backward()
            if step == 1:
                for prefix in _ENCODER_PREFIXES:
                    if not any(name.startswith(prefix) and parameter.grad is not None and torch.isfinite(parameter.grad).all() and float(parameter.grad.norm()) > 0.0 for name, parameter in model.named_parameters()):
                        raise RuntimeError(f"C4e direct score loss did not reach {prefix}")
                if any(parameter.grad is not None for name, parameter in model.named_parameters() if not name.startswith(_ENCODER_PREFIXES)):
                    raise RuntimeError("C4e direct score loss reached frozen parameters")
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                metrics, _ = _metrics(model, sequence, motion); status = _classify(metrics)
                if status["correlation_pass"] and thresholds["first_corr_ge_0p99_step"] is None: thresholds["first_corr_ge_0p99_step"] = step
                if status["temporal_rms_ratio_pass"] and thresholds["first_rms_ratio_in_0p9_1p1_step"] is None: thresholds["first_rms_ratio_in_0p9_1p1_step"] = step
                if status["score_mse_pass"] and thresholds["first_score_mse_le_0p01_step"] is None: thresholds["first_score_mse_le_0p01_step"] = step
                if status["classification"] == "single_location_overfit_pass" and thresholds["first_joint_success_step"] is None: thresholds["first_joint_success_step"] = step
                norm = torch.cat([parameter.detach().reshape(-1) for name, parameter in model.named_parameters() if name.startswith(_ENCODER_PREFIXES)]).norm(); raw = torch.cat([item[1].detach().reshape(-1) for item in losses_and_scores]).square().mean().sqrt()
                curve.append(_curve_row(step, metrics, float(norm), float(raw)))
        final_metrics, final_arrays = _metrics(model, sequence, motion)
    finally:
        renderer_hook.remove(); mbc_hook.remove()
    if renderer_calls[0] != 0 or mbc_calls[0] != 0:
        raise RuntimeError("C4e used renderer or learned respiratory MBC")
    changed = _parameter_audit(model, before)
    frozen_audit = {"film_encoder_card_unchanged": not any(name.startswith("film_encoder.card.") for name in changed), "canonical_unchanged": not any(name.startswith("canonical.") for name in changed), "respiratory_mbc_unchanged": not any(name.startswith("respiratory_mbc.") for name in changed), "cardiac_mbc_unchanged": not any(name.startswith("cardiac_mbc.") for name in changed), "uncertainty_unchanged": not any(name.startswith("uncertainty.") for name in changed)}
    if not all(frozen_audit.values()):
        raise RuntimeError("C4e frozen-module audit failed")
    classification = _classify(final_metrics)
    _write_curve(args.output_dir / "training_curve.csv", curve); (args.output_dir / "metrics_initial.json").write_text(json.dumps(initial_metrics, indent=2) + "\n", encoding="utf-8"); (args.output_dir / "metrics_final.json").write_text(json.dumps(final_metrics, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output_dir / "recovery_arrays.npz", **{f"initial_{name}": value for name, value in initial_arrays.items()}, **{f"final_{name}": value for name, value in final_arrays.items()}, timestamps_s=final_arrays["timestamps_s"], gt_scores=final_arrays["gt_scores"])
    torch.save({"model": model.state_dict(), "selected_location": geometry_audit, "step": args.steps, "git_sha": _git_sha()}, args.output_dir / "checkpoint_last.pt")
    summary = {"experiment": "Exp2-C4e single-location direct GT-score overfit", "git_sha": _git_sha(), "seed": args.seed, **geometry_audit, "supervision": "direct_gt_respiratory_score", "renderer_used": False, "mbc_used": False, "trainable_modules": list(_ENCODER_PREFIXES), "film_lr": learning_rate, "steps": args.steps, "eval_every": args.eval_every, "renderer_calls": renderer_calls[0], "learned_respiratory_mbc_calls": mbc_calls[0], "parameters_changed": changed, "frozen_module_audit": frozen_audit, "threshold_steps": thresholds, **classification, "initial_metrics": initial_metrics, "final_metrics": final_metrics, "interpretation_contract": "PASS supports single-location encoder capacity; C4d failure then implicates multi-location sharing/geometry conflict. FAIL supports the encoder/FiLM/pooling/head pathway as the primary blocker."}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2-C4e single-location direct GT-score overfit probe")
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs/source_first_change4_paperaligned.yaml"); parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml"); parser.add_argument("--manifest", type=Path); parser.add_argument("--qc-table", type=Path); parser.add_argument("--canonical-domain", type=Path); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--view", default="2CH"); parser.add_argument("--slice-id"); parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50); parser.add_argument("--steps", type=int, default=10000); parser.add_argument("--film-lr", type=float); parser.add_argument("--eval-every", type=int, default=100); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); summary = run(args); print(f"Exp2-C4e: {args.output_dir / 'summary.json'} ({summary['selected_view']}/{summary['selected_slice_id']})")


if __name__ == "__main__":
    main()
