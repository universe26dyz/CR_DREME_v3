#!/usr/bin/env python3
"""Exp2-C4f: direct GT-score encoder probe with constant GeometryFiLM input."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
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
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import _ENCODER_PREFIXES, _git_sha, _group_by_location, _prepare_output, _respiratory_only_motion, _set_seed, _validate_observations
from experiments.c4_diagnostics.exp2c4d_supervised_score_encoder_probe import _build_model, _configure_encoder_only, _direct_score_loss, _evaluate, _parameter_audit, _write_curve
from experiments.c4_diagnostics.synthetic.metrics import aggregate_location_metrics
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom


_GEOMETRY_FIELDS = ("center_mm", "row_direction", "column_direction", "normal", "pixel_spacing_mm", "slice_thickness_mm")


def _select_reference(observations: list, *, view: str, slice_id: str | None, smoke: bool):
    """Select the requested reference location; smoke runs may fall back by view."""
    if not smoke and slice_id is None:
        raise ValueError("C4f formal runs require an exact reference location")
    requested = [item for item in observations if item.view == view and (slice_id is None or item.slice_id == slice_id)]
    if requested:
        return min(requested, key=lambda item: (item.timestamp_s, item.dynamic_frame_id))
    if not smoke:
        raise ValueError(f"C4f reference location {view}/{slice_id} is absent")
    fallback = [item for item in observations if item.view == view]
    if not fallback:
        raise ValueError(f"C4f smoke reference view {view!r} is absent")
    return min(fallback, key=lambda item: (item.slice_id, item.timestamp_s, item.dynamic_frame_id))


def _with_reference_geometry(observation, reference):
    """Diagnostic-only observation copy retaining all non-GeometryFiLM metadata."""
    return replace(
        observation,
        center_mm=reference.center_mm.detach().clone(),
        row_direction=reference.row_direction.detach().clone(),
        column_direction=reference.column_direction.detach().clone(),
        normal=reference.normal.detach().clone(),
        pixel_spacing_mm=reference.pixel_spacing_mm.detach().clone(),
        slice_thickness_mm=float(reference.slice_thickness_mm),
    )


def _equal_geometry(field: str, first, second) -> bool:
    if field == "slice_thickness_mm":
        return float(first) == float(second)
    return bool(torch.equal(first.detach().cpu(), second.detach().cpu()))


def _geometry_audit(original: list, clamped: list, reference) -> dict[str, Any]:
    if len(original) != len(clamped):
        raise ValueError("C4f geometry audit needs aligned original and clamped observations")
    locations: dict[tuple[str, str], object] = {}
    for observation in original:
        locations.setdefault((observation.view, observation.slice_id), observation)
    result: dict[str, Any] = {
        "reference_view": reference.view,
        "reference_slice_id": reference.slice_id,
        "reference_timestamp_s": float(reference.timestamp_s),
        "reference_geometry": {
            field: (float(getattr(reference, field)) if field == "slice_thickness_mm" else getattr(reference, field).detach().cpu().tolist())
            for field in _GEOMETRY_FIELDS
        },
        "original_unique_location_count": len(locations),
    }
    for field in _GEOMETRY_FIELDS:
        values = [getattr(observation, field) for observation in locations.values()]
        result[f"{field}_original_varies"] = any(not _equal_geometry(field, values[0], value) for value in values[1:])
        clamped_values = [getattr(observation, field) for observation in clamped]
        result[f"{field}_clamped_constant"] = all(_equal_geometry(field, clamped_values[0], value) for value in clamped_values[1:])
        result[f"{field}_equals_reference"] = all(_equal_geometry(field, getattr(reference, field), value) for value in clamped_values)
    result["clamped_geometry_constant"] = all(result[f"{field}_clamped_constant"] and result[f"{field}_equals_reference"] for field in _GEOMETRY_FIELDS)
    return result


def _frozen_module_audit(changed: list[str]) -> dict[str, bool]:
    return {
        "film_encoder_card_unchanged": not any(name.startswith("film_encoder.card.") for name in changed),
        "canonical_unchanged": not any(name.startswith("canonical.") for name in changed),
        "respiratory_mbc_unchanged": not any(name.startswith("respiratory_mbc.") for name in changed),
        "cardiac_mbc_unchanged": not any(name.startswith("cardiac_mbc.") for name in changed),
        "uncertainty_unchanged": not any(name.startswith("uncertainty.") for name in changed),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if not args.smoke and (args.manifest is None or args.qc_table is None or args.canonical_domain is None):
        raise ValueError("formal C4f requires --manifest, --qc-table, and --canonical-domain")
    _prepare_output(args.output_dir, args.overwrite); _set_seed(args.seed)
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); synthetic = yaml.safe_load(args.experiment_config.read_text(encoding="utf-8"))["synthetic"]; domain = _load_domain(args.canonical_domain)
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(domain, respiratory_frequency_hz=float(synthetic["respiratory_frequency_hz"]), cardiac_frequency_hz=float(synthetic["cardiac_frequency_hz"]), respiratory_amplitude_mm=float(synthetic["respiratory_amplitude_mm"]), cardiac_amplitude_mm=float(synthetic["cardiac_amplitude_mm"])))
    if motion.cardiac_amplitude_mm != 0.0:
        raise RuntimeError("C4f target must have exactly zero cardiac amplitude")
    image_shape = (8, 8) if args.smoke else tuple(int(value) for value in synthetic["image_shape"]); psf_samples = 1 if args.smoke else int(synthetic["psf_samples"])
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=device) if args.manifest is not None else fallback_geometry(locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, image_shape=image_shape, motion=motion, device=device)
    observations = _build_observations(frames, image_shape=image_shape, phantom=AnalyticCanonicalPhantom(domain).to(device), motion=motion, psf_samples=psf_samples, seed=args.seed)
    _validate_observations(observations, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, smoke=args.smoke)
    reference = _select_reference(observations, view=args.reference_view, slice_id=args.reference_slice_id, smoke=args.smoke)
    clamped_observations = [_with_reference_geometry(observation, reference) for observation in observations]
    groups = _group_by_location(clamped_observations)
    geometry_audit = _geometry_audit(observations, clamped_observations, reference)
    if not geometry_audit["clamped_geometry_constant"]:
        raise RuntimeError("C4f clamped GeometryFiLM inputs are not constant")
    if not args.smoke and geometry_audit["original_unique_location_count"] != 9:
        raise RuntimeError("formal C4f requires exactly nine original locations")
    model = _build_model(config, domain, n_dynamic_frames=len(observations), seed=args.seed, device=device)
    if model.film_implementation != "full_geometry_single_film":
        raise RuntimeError("C4f requires full_geometry_single_film")
    optimizer = _configure_encoder_only(model, learning_rate=float(config["training"]["optimizer"]["stage2"]["film_lr"])); before = parameter_snapshot(model)
    renderer_calls = [0]; mbc_calls = [0]
    renderer_hook = model.psf.register_forward_hook(lambda *_: renderer_calls.__setitem__(0, renderer_calls[0] + 1)); mbc_hook = model.respiratory_mbc.register_forward_hook(lambda *_: mbc_calls.__setitem__(0, mbc_calls[0] + 1))
    try:
        initial_rows, initial_arrays, initial_by_view, initial_overall = _evaluate(model, groups, motion)
        sampler = ViewLocationBalancedSampler(clamped_observations, seed=args.seed); curve: list[dict[str, float | int]] = []
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            losses_and_scores = [_direct_score_loss(model, observation, motion) for observation in sampler.sample_step(observations_per_update=args.observations_per_update)]
            loss = torch.stack([item[0] for item in losses_and_scores]).mean(); loss.backward()
            if step == 1:
                for prefix in _ENCODER_PREFIXES:
                    if not any(name.startswith(prefix) and parameter.grad is not None and torch.isfinite(parameter.grad).all() and float(parameter.grad.norm()) > 0.0 for name, parameter in model.named_parameters()):
                        raise RuntimeError(f"C4f direct score loss did not reach {prefix}")
                if any(parameter.grad is not None for name, parameter in model.named_parameters() if not name.startswith(_ENCODER_PREFIXES)):
                    raise RuntimeError("C4f direct score loss reached frozen parameters")
            optimizer.step()
            parameters = [parameter.detach().reshape(-1) for name, parameter in model.named_parameters() if name.startswith(_ENCODER_PREFIXES)]
            sampled = torch.cat([item[1].detach().reshape(-1) for item in losses_and_scores])
            curve.append({"step": step, "score_mse": float(loss.detach()), "encoder_parameter_norm": float(torch.cat(parameters).norm()), "sampled_raw_score_rms": float(sampled.square().mean().sqrt())})
        final_rows, final_arrays, final_by_view, final_overall = _evaluate(model, groups, motion)
    finally:
        renderer_hook.remove(); mbc_hook.remove()
    if renderer_calls[0] != 0 or mbc_calls[0] != 0:
        raise RuntimeError("C4f used renderer or learned respiratory MBC")
    changed = _parameter_audit(model, before); frozen_audit = _frozen_module_audit(changed)
    if not all(frozen_audit.values()):
        raise RuntimeError("C4f frozen module audit failed")
    initial = {"overall": initial_overall, "per_location": initial_rows, "by_location": aggregate_location_metrics(initial_rows), "by_view": initial_by_view}
    final = {"overall": final_overall, "per_location": final_rows, "by_location": aggregate_location_metrics(final_rows), "by_view": final_by_view}
    _write_curve(args.output_dir / "training_curve.csv", curve)
    (args.output_dir / "metrics_initial.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "metrics_final.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output_dir / "recovery_arrays.npz", **initial_arrays, **{f"final_{name}": value for name, value in final_arrays.items()})
    summary = {
        "experiment": "Exp2-C4f multi-location constant-geometry probe", "git_sha": _git_sha(), "seed": args.seed, "n_locations": len(groups), "n_frames": len(observations),
        "intervention": "constant_geometry_conditioning", "comparison_target": "Exp2-C4d", "multi_location_images_preserved": True, "multi_location_timestamps_preserved": True, "multi_location_gt_scores_preserved": True, "location_geometry_conditioning_preserved": False,
        "reference_view": reference.view, "reference_slice_id": reference.slice_id, "reference_timestamp_s": float(reference.timestamp_s), "geometry_audit": geometry_audit,
        "supervision": "direct_gt_respiratory_score", "renderer_used": False, "mbc_used": False, "trainable_modules": list(_ENCODER_PREFIXES), "film_lr": float(config["training"]["optimizer"]["stage2"]["film_lr"]), "steps": args.steps, "observations_per_update": args.observations_per_update,
        "parameters_changed": changed, "parameter_change_audit": bool(changed), "frozen_module_audit": frozen_audit, "renderer_calls": renderer_calls[0], "learned_respiratory_mbc_calls": mbc_calls[0], "initial_metrics": initial, "final_metrics": final,
        "interpretation": "Report whether constant GeometryFiLM conditioning changes score recovery; no outcome is prespecified.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2-C4f multi-location constant-geometry encoder probe")
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs/source_first_change4_paperaligned.yaml"); parser.add_argument("--experiment-config", type=Path, default=PROJECT_ROOT / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml"); parser.add_argument("--manifest", type=Path); parser.add_argument("--qc-table", type=Path); parser.add_argument("--canonical-domain", type=Path); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--reference-view", default="2CH"); parser.add_argument("--reference-slice-id", default="2CH_s001"); parser.add_argument("--locations-per-view", type=int, default=3); parser.add_argument("--frames-per-location", type=int, default=50); parser.add_argument("--steps", type=int, default=750); parser.add_argument("--observations-per-update", type=int, default=32); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.observations_per_update < 3:
        parser.error("invalid C4f training arguments")
    summary = run(args); print(f"Exp2-C4f: {args.output_dir / 'summary.json'} ({summary['n_locations']} locations)")


if __name__ == "__main__":
    main()
