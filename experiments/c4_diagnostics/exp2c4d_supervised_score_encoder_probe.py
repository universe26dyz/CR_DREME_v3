#!/usr/bin/env python3
"""Exp2-C4d: direct GT respiratory-score supervision capacity probe."""
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

from cardioresp4d.training.sampler import ViewLocationBalancedSampler
from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, _load_domain, fallback_geometry, geometry_from_manifest
from experiments.c4_diagnostics.exp2_checkpoint_common import parameter_snapshot
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import _ENCODER_PREFIXES, _build_intervention_model, _git_sha, _prepare_output, _respiratory_only_motion, _score_values, _set_seed, _validate_observations
from experiments.c4_diagnostics.synthetic.metrics import aggregate_location_metrics
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom


EPSILON = 1e-12


def _build_model(config: dict[str, Any], domain: dict[str, object], *, n_dynamic_frames: int, seed: int, device: torch.device | None = None):
    return _build_intervention_model(config, domain, n_dynamic_frames=n_dynamic_frames, seed=seed, device=device)


def _gt_score(motion: GroundTruthMotion, timestamp_s: float | torch.Tensor, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    timestamp = torch.as_tensor(timestamp_s, device=device, dtype=dtype).reshape(-1)
    value = motion.respiratory_amplitude_mm * motion.respiratory_score(timestamp)
    score = value[:, None, None].expand(-1, 1, 3)
    if score.shape != (timestamp.numel(), 1, 3) or not torch.isfinite(score).all():
        raise RuntimeError("C4d GT respiratory score must be finite [B,1,3]")
    return score


def _configure_encoder_only(model, *, learning_rate: float) -> torch.optim.Adam:
    if learning_rate <= 0:
        raise ValueError("film learning rate must be positive")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = [(name, parameter) for name, parameter in model.named_parameters() if name.startswith(_ENCODER_PREFIXES)]
    if not selected:
        raise RuntimeError("C4d found no intended encoder parameters")
    for _, parameter in selected:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam([parameter for _, parameter in selected], lr=learning_rate)
    owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if owned != {id(parameter) for _, parameter in selected} or any(parameter.requires_grad != (id(parameter) in owned) for parameter in model.parameters()):
        raise RuntimeError("C4d optimizer ownership is not intended-encoder-only")
    return optimizer


def _direct_score_loss(model, observation, motion: GroundTruthMotion) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = model.encode_motion_scores(observation)["resp_scores"][:, :1, :]
    target = _gt_score(motion, observation.timestamp_s, device=prediction.device, dtype=prediction.dtype)
    if prediction.shape != target.shape:
        raise RuntimeError("C4d prediction and GT score shape mismatch")
    return (prediction - target).square().mean(), prediction


def _component_values(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, name in enumerate(("x", "y", "z")):
        pred, truth = prediction[..., index], target[..., index]
        dynamic_pred, dynamic_truth = pred - pred.mean(), truth - truth.mean()
        result[f"{name}_gt_correlation"] = float((dynamic_pred * dynamic_truth).sum() / (dynamic_pred.norm() * dynamic_truth.norm()).clamp_min(EPSILON))
        result[f"{name}_temporal_rms_ratio"] = float(dynamic_pred.square().mean().sqrt() / dynamic_truth.square().mean().sqrt().clamp_min(EPSILON))
    return result


@torch.inference_mode()
def _evaluate(model, groups: dict[str, list], motion: GroundTruthMotion) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, dict[str, float]], dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}; arrays: dict[str, np.ndarray] = {}; all_predicted: list[torch.Tensor] = []; all_target: list[torch.Tensor] = []; by_view_scores: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for location, sequence in groups.items():
        predicted = torch.cat([model.encode_motion_scores(observation)["resp_scores"][:, :1, :] for observation in sequence])
        target = torch.cat([_gt_score(motion, observation.timestamp_s, device=predicted.device, dtype=predicted.dtype) for observation in sequence])
        rows[location] = {**_score_values(predicted, target), **_component_values(predicted, target)}
        all_predicted.append(predicted); all_target.append(target); by_view_scores.setdefault(sequence[0].view, []).append((predicted, target))
        stem = location.replace("/", "_"); arrays[f"{stem}_timestamps_s"] = np.asarray([observation.timestamp_s for observation in sequence]); arrays[f"{stem}_predicted_scores"] = predicted.cpu().numpy(); arrays[f"{stem}_gt_scores"] = target.cpu().numpy()
    by_view: dict[str, dict[str, float]] = {}
    for view, values in by_view_scores.items():
        predicted, target = torch.cat([item[0] for item in values]), torch.cat([item[1] for item in values])
        by_view[view] = {**_score_values(predicted, target), **_component_values(predicted, target)}
    overall_prediction, overall_target = torch.cat(all_predicted), torch.cat(all_target)
    return rows, arrays, by_view, {**_score_values(overall_prediction, overall_target), **_component_values(overall_prediction, overall_target)}


def _parameter_audit(model, before: dict[str, torch.Tensor], *, require_change: bool = True) -> list[str]:
    after = dict(model.named_parameters())
    if set(after) != set(before):
        raise RuntimeError("C4d model parameter set changed")
    changed = [name for name, parameter in after.items() if not torch.equal(before[name], parameter.detach().cpu())]
    if set(changed) - {name for name in after if name.startswith(_ENCODER_PREFIXES)} or (require_change and not changed):
        raise RuntimeError("C4d frozen-module parameter audit failed")
    return changed


def _write_curve(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "score_mse", "encoder_parameter_norm", "sampled_raw_score_rms"]); writer.writeheader(); writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if not args.smoke and (args.manifest is None or args.qc_table is None or args.canonical_domain is None):
        raise ValueError("formal C4d requires --manifest, --qc-table, and --canonical-domain")
    _prepare_output(args.output_dir, args.overwrite); _set_seed(args.seed)
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); synthetic = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))["synthetic"]; domain = _load_domain(args.canonical_domain)
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(domain, respiratory_frequency_hz=float(synthetic["respiratory_frequency_hz"]), cardiac_frequency_hz=float(synthetic["cardiac_frequency_hz"]), respiratory_amplitude_mm=float(synthetic["respiratory_amplitude_mm"]), cardiac_amplitude_mm=float(synthetic["cardiac_amplitude_mm"])))
    if motion.cardiac_amplitude_mm != 0.0:
        raise RuntimeError("C4d target must have exactly zero cardiac amplitude")
    image_shape = (8, 8) if args.smoke else tuple(int(value) for value in synthetic["image_shape"]); psf_samples = 1 if args.smoke else int(synthetic["psf_samples"])
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=device) if args.manifest is not None else fallback_geometry(locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, image_shape=image_shape, motion=motion, device=device)
    observations = _build_observations(frames, image_shape=image_shape, phantom=AnalyticCanonicalPhantom(domain).to(device), motion=motion, psf_samples=psf_samples, seed=args.seed)
    groups = _validate_observations(observations, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, smoke=args.smoke)
    model = _build_model(config, domain, n_dynamic_frames=len(observations), seed=args.seed, device=device); optimizer = _configure_encoder_only(model, learning_rate=float(config["training"]["optimizer"]["stage2"]["film_lr"])); before = parameter_snapshot(model)
    renderer_calls = [0]; mbc_calls = [0]; renderer_hook = model.psf.register_forward_hook(lambda *_: renderer_calls.__setitem__(0, renderer_calls[0] + 1)); mbc_hook = model.respiratory_mbc.register_forward_hook(lambda *_: mbc_calls.__setitem__(0, mbc_calls[0] + 1))
    try:
        initial_rows, initial_arrays, initial_by_view, initial_overall = _evaluate(model, groups, motion); sampler = ViewLocationBalancedSampler(observations, seed=args.seed); curve: list[dict[str, float | int]] = []
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True); losses_and_scores = [_direct_score_loss(model, observation, motion) for observation in sampler.sample_step(observations_per_update=args.observations_per_update)]; loss = torch.stack([item[0] for item in losses_and_scores]).mean(); loss.backward()
            if step == 1:
                for prefix in _ENCODER_PREFIXES:
                    if not any(name.startswith(prefix) and parameter.grad is not None and torch.isfinite(parameter.grad).all() and float(parameter.grad.norm()) > 0.0 for name, parameter in model.named_parameters()):
                        raise RuntimeError(f"C4d direct score loss did not reach {prefix}")
                if any(parameter.grad is not None for name, parameter in model.named_parameters() if not name.startswith(_ENCODER_PREFIXES)):
                    raise RuntimeError("C4d direct score loss reached frozen parameters")
            optimizer.step(); parameters = [parameter.detach().reshape(-1) for name, parameter in model.named_parameters() if name.startswith(_ENCODER_PREFIXES)]; sampled = torch.cat([item[1].detach().reshape(-1) for item in losses_and_scores])
            curve.append({"step": step, "score_mse": float(loss.detach()), "encoder_parameter_norm": float(torch.cat(parameters).norm()), "sampled_raw_score_rms": float(sampled.square().mean().sqrt())})
        final_rows, final_arrays, final_by_view, final_overall = _evaluate(model, groups, motion)
    finally:
        renderer_hook.remove(); mbc_hook.remove()
    if renderer_calls[0] != 0 or mbc_calls[0] != 0:
        raise RuntimeError("C4d used renderer or learned respiratory MBC")
    changed = _parameter_audit(model, before)
    frozen_audit = {"film_encoder_card_unchanged": not any(name.startswith("film_encoder.card.") for name in changed), "canonical_unchanged": not any(name.startswith("canonical.") for name in changed), "respiratory_mbc_unchanged": not any(name.startswith("respiratory_mbc.") for name in changed), "cardiac_mbc_unchanged": not any(name.startswith("cardiac_mbc.") for name in changed), "uncertainty_unchanged": not any(name.startswith("uncertainty.") for name in changed)}
    if not all(frozen_audit.values()):
        raise RuntimeError("C4d frozen module audit failed")
    initial = {"overall": initial_overall, "per_location": initial_rows, "by_location": aggregate_location_metrics(initial_rows), "by_view": initial_by_view}; final = {"overall": final_overall, "per_location": final_rows, "by_location": aggregate_location_metrics(final_rows), "by_view": final_by_view}
    _write_curve(args.output_dir / "training_curve.csv", curve); (args.output_dir / "metrics_initial.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8"); (args.output_dir / "metrics_final.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8"); np.savez_compressed(args.output_dir / "recovery_arrays.npz", **initial_arrays, **{f"final_{name}": value for name, value in final_arrays.items()})
    summary = {"experiment": "Exp2-C4d supervised GT-score encoder probe", "git_sha": _git_sha(), "seed": args.seed, "n_locations": len(groups), "n_frames": len(observations), "supervision": "direct_gt_respiratory_score", "renderer_used": False, "mbc_used": False, "trainable_modules": list(_ENCODER_PREFIXES), "film_lr": float(config["training"]["optimizer"]["stage2"]["film_lr"]), "steps": args.steps, "observations_per_update": args.observations_per_update, "parameters_changed": changed, "parameter_change_audit": bool(changed), "frozen_module_audit": frozen_audit, "renderer_calls": renderer_calls[0], "learned_respiratory_mbc_calls": mbc_calls[0], "initial_metrics": initial, "final_metrics": final}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2-C4d direct GT-score encoder capacity probe")
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs/source_first_change4_paperaligned.yaml"); parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml"); parser.add_argument("--manifest", type=Path); parser.add_argument("--qc-table", type=Path); parser.add_argument("--canonical-domain", type=Path); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50); parser.add_argument("--steps", type=int, default=750); parser.add_argument("--observations-per-update", type=int, default=32); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.observations_per_update < 3:
        parser.error("invalid C4d training arguments")
    summary = run(args); print(f"Exp2-C4d: {args.output_dir / 'summary.json'} ({summary['n_locations']} locations)")


if __name__ == "__main__":
    main()
