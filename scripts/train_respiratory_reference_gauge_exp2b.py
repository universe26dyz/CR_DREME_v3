#!/usr/bin/env python3
"""Train the isolated 200-step Exp2-B fixed respiratory score-gain experiment."""
from __future__ import annotations

import copy
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from train_respiratory_reference_gauge_exp2a import _build_exact_model, _sha256, _state_digest
from train_respiratory_s2vdreme_reference_v1 import _load_config
from train_source_first import observations_from_manifest
from cardioresp4d.adapters.nesvor_inr import detect_checkpoint_encoding_backend
from cardioresp4d.experiments.respiratory_reference_gauge_exp2b import (
    EXP2B_ALLOWED_SCORE_GAINS, EXP2B_PROTOCOL_BASE, Exp2BReferenceRunner,
    exp2b_gauge_diagnostics, exp2b_protocol, exp2b_provenance,
    validate_exp2b_protocol, validate_exp2b_score_gain,
)
from cardioresp4d.experiments.respiratory_reference_gauge_exp2a import exp2a_input_bindings
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import (
    ReferenceSegment, apply_corrected_static_poses, derive_effective_external_domain,
    load_corrected_pose_map,
)
from cardioresp4d.training.runtime_state import capture_rng_state, restore_rng_state, set_reproducibility
from cardioresp4d.training.sampler import ViewLocationBalancedSampler
from cardioresp4d.visualization.reference_gauge import ReferenceImageProvider


CHECKPOINT_SCHEMA = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, default=ROOT / "configs" / "respiratory_reference_gauge_exp2b.yaml")
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True); parser.add_argument("--static-manifest", type=Path, required=True); parser.add_argument("--mean-manifest", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--resume", type=Path); parser.add_argument("--score-gain", type=float, required=True, choices=EXP2B_ALLOWED_SCORE_GAINS)
    args = parser.parse_args(); score_gain = validate_exp2b_score_gain(args.score_gain); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    config = _load_config(args.experiment_config); validate_exp2b_protocol(config)
    effective_config = copy.deepcopy(config); effective_config.setdefault("exp2b", {})["score_gain"] = score_gain
    set_reproducibility(EXP2B_PROTOCOL_BASE["seed"])
    static_manifest = json.loads(args.static_manifest.read_text()); static_model = Path(static_manifest["outputs"]["model"]["path"]); pose_manifest = Path(static_manifest["corrected_pose_manifest"]["path"])
    if static_manifest.get("source_manifest_sha256") != _sha256(args.manifest) or static_manifest.get("qc_table_sha256") != _sha256(args.qc_table):
        raise ValueError("static manifest does not bind this dynamic manifest/QC table")
    if static_manifest.get("outputs", {}).get("model", {}).get("sha256") and static_manifest["outputs"]["model"]["sha256"] != _sha256(static_model):
        raise ValueError("static model hash does not match static manifest")
    if static_manifest.get("corrected_pose_manifest", {}).get("sha256") and static_manifest["corrected_pose_manifest"]["sha256"] != _sha256(pose_manifest):
        raise ValueError("corrected pose hash does not match static manifest")
    if static_manifest.get("mean_slice_manifest", {}).get("path") and Path(static_manifest["mean_slice_manifest"]["path"]).resolve() != args.mean_manifest.resolve():
        raise ValueError("Exp2-B mean manifest differs from the authoritative static manifest")
    domain = json.loads(args.canonical_domain.read_text()); poses = load_corrected_pose_map(pose_manifest); effective = derive_effective_external_domain(poses, domain["world_min_mm"], domain["world_max_mm"]); model_domain = {**domain, "world_min_mm": effective["model_float32_effective_domain"]["lower_lps_mm"], "world_max_mm": effective["model_float32_effective_domain"]["upper_lps_mm"]}
    observations, normalization, groups = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=config["training"]["normalization"]["mode"], assign_slice_positions=True); observations = apply_corrected_static_poses(observations, poses)
    static_payload = torch.load(static_model, map_location="cpu", weights_only=False); model, backend, internal_bounds = _build_exact_model(effective_config, model_domain, static_payload, observations, device)
    provider = ReferenceImageProvider.from_mean_artifacts(args.mean_manifest, observations=observations); model.set_reference_score_provider(provider); model.set_reference_score_gain(score_gain)
    sampler = ViewLocationBalancedSampler(observations, seed=EXP2B_PROTOCOL_BASE["seed"]); runner = Exp2BReferenceRunner(model, sampler, score_gain=score_gain, pixel_samples=EXP2B_PROTOCOL_BASE["pixel_samples"], observations_per_update=EXP2B_PROTOCOL_BASE["observations_per_update"], optimizer_config=config["training"]["optimizer"])
    provenance = exp2b_provenance(manifest=args.manifest, qc_table=args.qc_table, canonical_domain=args.canonical_domain, static_manifest=args.static_manifest, mean_manifest=args.mean_manifest, config=effective_config, score_gain=score_gain, static_model=static_model)
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_schema") != CHECKPOINT_SCHEMA or checkpoint.get("provenance") != provenance:
            raise ValueError("Exp2-B resume checkpoint schema/provenance mismatch, including score_gain")
        score_policy = checkpoint.get("score_gauge", {})
        if score_policy.get("type") != "gain_times_raw_minus_reference" or float(score_policy.get("score_gain", -1.0)) != score_gain:
            raise ValueError("Exp2-B resume checkpoint score_gain does not match this run")
        model.load_state_dict(checkpoint["model"], strict=True); runner.load_boundary_state(checkpoint["runner_state"]); restore_rng_state(checkpoint["rng_state"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_bindings = exp2a_input_bindings(manifest=args.manifest, qc_table=args.qc_table, canonical_domain=args.canonical_domain, static_manifest=args.static_manifest, mean_manifest=args.mean_manifest, static_model=static_model, corrected_pose_manifest=pose_manifest)
    (args.output_dir / "experiment_inputs.json").write_text(json.dumps(input_bindings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "effective_config.json").write_text(json.dumps(effective_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    initialization_audit = {"seed": EXP2B_PROTOCOL_BASE["seed"], "encoder_initial_state_sha256": _state_digest(model.film_encoder), "l1_mbc_initial_state_sha256": _state_digest(model.respiratory_mbc.levels[0]), "model_config_sha256": provenance["config_sha256"], "initialization_policy": "fresh deterministic Exp2-B construction after set_reproducibility(seed)", "same_seeded_initialization_expected_by_protocol": True, "exact_historical_pretraining_weight_equality_not_directly_observable": True}
    if checkpoint is not None and checkpoint.get("initialization_audit"):
        initialization_audit = checkpoint["initialization_audit"]
    before = exp2b_gauge_diagnostics(model, sampler, score_gain=score_gain)
    if runner.global_step < EXP2B_PROTOCOL_BASE["steps"]:
        segment = ReferenceSegment("exp2b_step_200", "stage2a", EXP2B_PROTOCOL_BASE["steps"], 1, frozenset({"respiratory_encoder", "respiratory_mbc"})); report = runner.run_segment(segment)
    else:
        report = runner.completed[-1]
    after = exp2b_gauge_diagnostics(model, sampler, score_gain=score_gain)
    static_initialization = {"mode": "direct_exact_state_with_coordinate_bridge", "static_model_sha256": _sha256(static_model), "internal_inr_bounds_mm": internal_bounds.tolist(), "encoding_backend": backend, "canonical_frozen": True}
    score_gauge = {"type": "gain_times_raw_minus_reference", "policy": EXP2B_PROTOCOL_BASE["score_policy"], "score_gain": score_gain, "reference_manifest_sha256": provider.provenance["mean_manifest_sha256"], "reference_artifact_hashes": provider.provenance["mean_stacks"]}
    payload = {"checkpoint_schema": CHECKPOINT_SCHEMA, "experiment": EXP2B_PROTOCOL_BASE["experiment"], "model": model.state_dict(), "runner_state": runner.state_dict(), "rng_state": capture_rng_state(), "effective_config": effective_config, "provenance": provenance, "static_initialization": static_initialization, "score_gauge": score_gauge, "diagnostics": {"step_0": before, "step_200": after}, "initialization_audit": initialization_audit}
    torch.save(payload, args.output_dir / "checkpoint_step_200.pt"); torch.save(payload, args.output_dir / "checkpoint_last.pt")
    training_report = {"experiment": EXP2B_PROTOCOL_BASE["experiment"], "protocol": exp2b_protocol(score_gain), "score_gain": score_gain, "runner_report": report, "diagnostics": payload["diagnostics"], "initialization_audit": initialization_audit, "provenance": provenance, "input_bindings": input_bindings, "normalization_parameters": normalization, "normalization_groups": groups}
    (args.output_dir / "training_report.json").write_text(json.dumps(training_report, indent=2) + "\n")
    print(json.dumps({"status": "trained", "experiment": EXP2B_PROTOCOL_BASE["experiment"], "score_gain": score_gain, "output": str(args.output_dir), "global_step": runner.global_step}, indent=2))


if __name__ == "__main__":
    main()
