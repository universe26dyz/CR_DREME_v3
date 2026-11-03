#!/usr/bin/env python3
"""Exp3-B1a: controlled C4a-to-B1 Phase-A pixel/batch protocol bridge."""
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
from experiments.c4_diagnostics.exp2c4a_gt_score_mbc_only import _assert_only_level0_changed, _build_intervention_model, _configure_level0_only, _evaluate, _intervention_loss_terms, _prepare_output, _respiratory_only_motion, _sample_pixels, _set_seed, _validate_observations
from experiments.c4_diagnostics.exp2c4h_oracle_field_fullbatch_reconstruction import _fixed_pixel_grid
from experiments.c4_diagnostics.synthetic.metrics import aggregate_location_metrics
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from cardioresp4d.training.sampler import ViewLocationBalancedSampler


def _protocol(pixel_mode: str) -> dict[str, Any]:
    if pixel_mode not in {"random", "fixed_grid"}: raise ValueError("pixel mode must be random or fixed_grid")
    return {"experiment": "Exp3-B1a C4a bridge", "pixel_mode": pixel_mode, "steps": 750, "observations_per_update": 32, "pixel_samples": 256, "objective": "data_only", "active_respiratory_levels": 1, "gt_score": True, "optimizer": "Adam stage2.respiratory_mbc_lr", "sampler": "ViewLocationBalancedSampler"}


def _pixel_uv(observation, pixel_mode: str, count: int) -> torch.Tensor:
    return _sample_pixels(observation, count) if pixel_mode == "random" else _fixed_pixel_grid(observation, count=count)


def _exposure_accounting(*, steps: int, observations_per_update: int, pixels_per_observation: int) -> dict[str, int]:
    return {"optimizer_steps": steps, "observations_per_update": observations_per_update, "pixels_per_observation": pixels_per_observation, "observation_exposures": steps * observations_per_update, "pixel_observation_exposures": steps * observations_per_update * pixels_per_observation}


def _dvf(metrics: dict[str, Any]) -> dict[str, float]:
    return {name: float(metrics["aggregate"][name]["mean"]) for name in ("full_fov_dynamic_cosine", "full_fov_dynamic_rms_ratio", "full_fov_dynamic_normalized_epe")}


def _passes(metric: dict[str, float]) -> bool:
    return metric["full_fov_dynamic_cosine"] >= .90 and .5 <= metric["full_fov_dynamic_rms_ratio"] <= 1.5 and metric["full_fov_dynamic_normalized_epe"] <= .75


def _classify_bridge(r0: dict[str, float], r1: dict[str, float], b1: dict[str, float]) -> str:
    if not _passes(r0): return "c4a_reproduction_failure"
    if not _passes(r1): return "fixed_pixel_sampling_degrades_mbc_recovery"
    if not _passes(b1): return "exact_fullbatch_optimization_is_primary_bridge_difference"
    return "bridge_difference_not_explained_by_pixel_sampling_alone"


def _existing_b1_at_step(summary: Path, curve: Path | None, step: int = 750) -> dict[str, float]:
    if curve is not None:
        with curve.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row.get("global_step", row.get("step", -1))) == step:
                    return {name: float(row[name]) for name in ("full_fov_dynamic_cosine", "full_fov_dynamic_rms_ratio", "full_fov_dynamic_normalized_epe")}
        raise ValueError(f"existing B1 curve has no step {step}")
    payload = json.loads(summary.read_text(encoding="utf-8")); value = payload.get("phase_a", {}).get("final_dvf")
    if not isinstance(value, dict): raise ValueError("existing B1 summary has no phase_a.final_dvf")
    return {name: float(value[name]) for name in ("full_fov_dynamic_cosine", "full_fov_dynamic_rms_ratio", "full_fov_dynamic_normalized_epe")}


def _run_arm(args, pixel_mode: str) -> dict[str, Any]:
    protocol = _protocol(pixel_mode); output = args.output_dir / pixel_mode; _prepare_output(output, False); _set_seed(args.seed)
    config = yaml.safe_load(args.source_config.read_text()); synthetic = yaml.safe_load(args.experiment_config.read_text())["synthetic"]; domain = _load_domain(args.canonical_domain)
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(domain, respiratory_frequency_hz=float(synthetic["respiratory_frequency_hz"]), cardiac_frequency_hz=float(synthetic["cardiac_frequency_hz"]), respiratory_amplitude_mm=float(synthetic["respiratory_amplitude_mm"]), cardiac_amplitude_mm=float(synthetic["cardiac_amplitude_mm"])))
    shape, psf = ((8, 8), 1) if args.smoke else (tuple(synthetic["image_shape"]), int(synthetic["psf_samples"]))
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=torch.device(args.device)) if args.manifest else fallback_geometry(locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, image_shape=shape, motion=motion, device=torch.device(args.device))
    observations = _build_observations(frames, image_shape=shape, phantom=AnalyticCanonicalPhantom(domain).to(args.device), motion=motion, psf_samples=psf, seed=args.seed); groups = _validate_observations(observations, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, smoke=args.smoke)
    model = _build_intervention_model(config, domain, n_dynamic_frames=len(observations), init_mode="fresh", checkpoint=None, seed=args.seed, device=torch.device(args.device)); optimizer = _configure_level0_only(model, learning_rate=float(config["training"]["optimizer"]["stage2"]["respiratory_mbc_lr"])); before = parameter_snapshot(model); initial_rows, initial_arrays, _ = _evaluate(model, groups, motion, grid_shape=tuple(args.evaluation_grid_shape)); sampler = ViewLocationBalancedSampler(observations, seed=args.seed); curve=[]
    encoder_calls = 0

    def _count_encoder_calls(_module, _args, _output):
        nonlocal encoder_calls
        encoder_calls += 1

    hook = model.film_encoder.register_forward_hook(_count_encoder_calls)
    try:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True); selected = sampler.sample_step(observations_per_update=args.observations_per_update)
            losses = [_intervention_loss_terms(model, observation, motion, pixel_uv=_pixel_uv(observation, pixel_mode, args.pixel_samples), objective="data_only", normalization_weight=0.) for observation in selected]
            loss = torch.stack([item["data_mse"] for item in losses]).mean(); loss.backward()
            if step == 1 and (not any(p.grad is not None and float(p.grad.norm()) > 0 for p in model.respiratory_mbc.levels[0].parameters()) or any(p.grad is not None for n,p in model.named_parameters() if not n.startswith("respiratory_mbc.levels.0."))): raise RuntimeError("bridge Phase-A gradient ownership audit failed")
            optimizer.step(); curve.append({"step": step, "data_mse": float(loss.detach())})
    finally:
        hook.remove()
    if encoder_calls:
        raise RuntimeError("bridge Phase-A forward unexpectedly invoked the FiLM encoder")
    final_rows, final_arrays, _ = _evaluate(model, groups, motion, grid_shape=tuple(args.evaluation_grid_shape)); changed, unchanged = _assert_only_level0_changed(model, before)
    initial, final = {"per_location": initial_rows, "aggregate": aggregate_location_metrics(initial_rows)}, {"per_location": final_rows, "aggregate": aggregate_location_metrics(final_rows)}
    with (output / "training_curve.csv").open("w", newline="") as handle: writer=csv.DictWriter(handle, fieldnames=["step","data_mse"]); writer.writeheader(); writer.writerows(curve)
    (output / "metrics_initial.json").write_text(json.dumps(initial, indent=2)+"\n"); (output / "metrics_final.json").write_text(json.dumps(final, indent=2)+"\n"); np.savez_compressed(output / "recovery_arrays.npz", **initial_arrays, **{f"final_{key}": value for key,value in final_arrays.items()})
    result = {"arm": "R0" if pixel_mode == "random" else "R1", "pixel_mode": pixel_mode, "protocol": protocol, "exposure_accounting": _exposure_accounting(steps=args.steps, observations_per_update=args.observations_per_update, pixels_per_observation=args.pixel_samples), "initial_metrics": initial, "final_metrics": final, "final_dvf": _dvf(final), "parameters_changed": changed, "parameters_unchanged": unchanged, "encoder_calls": encoder_calls}
    (output / "summary.json").write_text(json.dumps(result, indent=2)+"\n"); return result


def run(args) -> dict[str, Any]:
    if not args.smoke and (args.steps, args.observations_per_update, args.pixel_samples, args.locations_per_view, args.frames_per_location) != (750, 32, 256, 3, 50): raise ValueError("formal bridge requires C4a 750/32/256/9-location/450-frame protocol")
    if not args.smoke and (args.manifest is None or args.qc_table is None or args.canonical_domain is None): raise ValueError("formal bridge requires manifest, QC table, and canonical domain")
    modes = ("random", "fixed_grid") if args.arm == "all" else (args.arm,); results = {"R0" if mode == "random" else "R1": _run_arm(args, mode) for mode in modes}
    if args.existing_b1_summary is not None:
        b1 = _existing_b1_at_step(args.existing_b1_summary, args.existing_b1_curve); results["B1_A_750"] = {"pixel_mode": "fixed_grid", "observations_per_update": 450, "final_dvf": b1}
    if {"R0", "R1", "B1_A_750"} <= set(results):
        comparison = {"results": results, "classification": _classify_bridge(results["R0"]["final_dvf"], results["R1"]["final_dvf"], results["B1_A_750"]["final_dvf"])}; (args.output_dir / "bridge_comparison.json").write_text(json.dumps(comparison, indent=2)+"\n")
        with (args.output_dir / "bridge_comparison.csv").open("w", newline="") as handle: writer=csv.DictWriter(handle, fieldnames=["arm","pixel_mode","observations_per_update","full_fov_dynamic_cosine","full_fov_dynamic_rms_ratio","full_fov_dynamic_normalized_epe"]); writer.writeheader(); writer.writerows({"arm":name, "pixel_mode":value["pixel_mode"], "observations_per_update":value.get("observations_per_update", value.get("exposure_accounting",{}).get("observations_per_update")), **value["final_dvf"]} for name,value in results.items())
    return results


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--source-config",type=Path,default=PROJECT_ROOT/"configs/source_first_change4_paperaligned.yaml"); parser.add_argument("--experiment-config",type=Path,default=PROJECT_ROOT/"experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml"); parser.add_argument("--manifest",type=Path); parser.add_argument("--qc-table",type=Path); parser.add_argument("--canonical-domain",type=Path); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--existing-b1-summary",type=Path); parser.add_argument("--existing-b1-curve",type=Path); parser.add_argument("--arm",choices=("all","random","fixed_grid"),default="all"); parser.add_argument("--device",default="cpu"); parser.add_argument("--seed",type=int,default=0); parser.add_argument("--locations-per-view",type=int,default=3); parser.add_argument("--frames-per-location",type=int,default=50); parser.add_argument("--steps",type=int,default=750); parser.add_argument("--observations-per-update",type=int,default=32); parser.add_argument("--pixel-samples",type=int,default=256); parser.add_argument("--evaluation-grid-shape",type=lambda x:tuple(map(int,x.split(","))),default=(8,8,8)); parser.add_argument("--smoke",action="store_true"); run(parser.parse_args())
if __name__ == "__main__": main()
