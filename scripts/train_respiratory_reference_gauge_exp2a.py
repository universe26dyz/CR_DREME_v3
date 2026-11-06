#!/usr/bin/env python3
"""Train the isolated 200-step Exp2-A hard reference-score gauge experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

from train_respiratory_s2vdreme_reference_v1 import _load_config
from train_source_first import observations_from_manifest
from cardioresp4d.adapters.nesvor_inr import detect_checkpoint_encoding_backend, lps_to_ras_world_matrix
from cardioresp4d.experiments.respiratory_reference_gauge_exp2a import EXP2A_PROTOCOL, Exp2AReferenceRunner, exp2a_gauge_diagnostics, exp2a_provenance
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import ReferenceSegment, apply_corrected_static_poses, derive_effective_external_domain, load_corrected_pose_map, strict_canonical_state_from_static_checkpoint
from cardioresp4d.training.build_model import build_source_first_model
from cardioresp4d.training.runtime_state import capture_rng_state, restore_rng_state, set_reproducibility
from cardioresp4d.training.sampler import ViewLocationBalancedSampler
from cardioresp4d.visualization.reference_gauge import ReferenceImageProvider


CHECKPOINT_SCHEMA = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.state_dict().items()):
        digest.update(name.encode()); digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validate_config(config: dict) -> None:
    values = config.get("exp2a", {})
    expected = {"steps": 200, "pixel_samples": 256, "active_respiratory_levels": 1, "canonical_frozen": True, "score_policy": "raw_minus_same_encoder_reference_mean"}
    if any(values.get(key) != value for key, value in expected.items()):
        raise ValueError("Exp2-A config does not match the fixed hard reference-score gauge protocol")
    training = config.get("training", {})
    if training.get("observations_per_update") != 32 or training.get("seed") != 20260823 or training.get("normalization", {}).get("mode") != "per_series":
        raise ValueError("Exp2-A requires seed 20260823, 32 observations/update, and per_series normalization")
    if float(training.get("optimizer", {}).get("respiratory_encoder_lr")) != 5e-4 or float(training.get("optimizer", {}).get("respiratory_mbc_lr")) != 5e-4:
        raise ValueError("Exp2-A requires 5e-4 encoder and L1 MBC learning rates")


def _build_exact_model(config: dict, domain: dict, static_payload: dict, observations: list, device: torch.device):
    candidate = static_payload.get("model", static_payload.get("inr_state_dict")); internal_bounds = candidate["bounding_box"].detach().cpu().float(); backend = detect_checkpoint_encoding_backend(dict(candidate)); bridge = lps_to_ras_world_matrix(device=device)
    model = build_source_first_model(config, domain, n_dynamic_frames=max(item.dynamic_frame_id for item in observations) + 1, device=device, canonical_encoding_backend=backend, canonical_internal_bounds_world_mm=internal_bounds.to(device), canonical_world_to_internal_transform=bridge).to(device)
    state = strict_canonical_state_from_static_checkpoint(static_payload, model.canonical.inr.state_dict(), internal_bounds_world_mm=internal_bounds, expected_architecture=model.canonical.args, expected_encoding_backend=backend, world_to_internal_transform=model.canonical.input_world_to_internal.cpu())
    model.canonical.inr.load_state_dict(state, strict=True)
    return model, backend, internal_bounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, default=ROOT / "configs" / "respiratory_reference_gauge_exp2a.yaml")
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True); parser.add_argument("--static-manifest", type=Path, required=True); parser.add_argument("--mean-manifest", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--resume", type=Path)
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    config = _load_config(args.experiment_config); _validate_config(config); set_reproducibility(EXP2A_PROTOCOL["seed"])
    static_manifest = json.loads(args.static_manifest.read_text()); static_model = Path(static_manifest["outputs"]["model"]["path"]); pose_manifest = Path(static_manifest["corrected_pose_manifest"]["path"])
    if static_manifest.get("source_manifest_sha256") != _sha256(args.manifest) or static_manifest.get("qc_table_sha256") != _sha256(args.qc_table):
        raise ValueError("static manifest does not bind this dynamic manifest/QC table")
    if static_manifest.get("outputs", {}).get("model", {}).get("sha256") and static_manifest["outputs"]["model"]["sha256"] != _sha256(static_model):
        raise ValueError("static model hash does not match static manifest")
    if static_manifest.get("corrected_pose_manifest", {}).get("sha256") and static_manifest["corrected_pose_manifest"]["sha256"] != _sha256(pose_manifest):
        raise ValueError("corrected pose hash does not match static manifest")
    if static_manifest.get("mean_slice_manifest", {}).get("path") and Path(static_manifest["mean_slice_manifest"]["path"]).resolve() != args.mean_manifest.resolve():
        raise ValueError("Exp2-A mean manifest differs from the authoritative static manifest")
    domain = json.loads(args.canonical_domain.read_text()); poses = load_corrected_pose_map(pose_manifest); effective = derive_effective_external_domain(poses, domain["world_min_mm"], domain["world_max_mm"]); model_domain = {**domain, "world_min_mm": effective["model_float32_effective_domain"]["lower_lps_mm"], "world_max_mm": effective["model_float32_effective_domain"]["upper_lps_mm"]}
    observations, normalization, groups = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=config["training"]["normalization"]["mode"], assign_slice_positions=True); observations = apply_corrected_static_poses(observations, poses)
    static_payload = torch.load(static_model, map_location="cpu", weights_only=False); model, backend, internal_bounds = _build_exact_model(config, model_domain, static_payload, observations, device)
    provider = ReferenceImageProvider.from_mean_artifacts(args.mean_manifest, observations=observations); model.set_reference_score_provider(provider)
    sampler = ViewLocationBalancedSampler(observations, seed=EXP2A_PROTOCOL["seed"]); runner = Exp2AReferenceRunner(model, sampler, pixel_samples=EXP2A_PROTOCOL["pixel_samples"], observations_per_update=EXP2A_PROTOCOL["observations_per_update"], optimizer_config=config["training"]["optimizer"])
    provenance = exp2a_provenance(manifest=args.manifest, qc_table=args.qc_table, canonical_domain=args.canonical_domain, static_manifest=args.static_manifest, mean_manifest=args.mean_manifest, config=config, static_model=static_model)
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_schema") != CHECKPOINT_SCHEMA or checkpoint.get("provenance") != provenance:
            raise ValueError("Exp2-A resume checkpoint schema/provenance mismatch")
        if checkpoint.get("score_gauge", {}).get("policy") != EXP2A_PROTOCOL["score_policy"]:
            raise ValueError("Exp2-A resume checkpoint is not a hard reference-gauge checkpoint")
        model.load_state_dict(checkpoint["model"], strict=True); runner.load_boundary_state(checkpoint["runner_state"]); restore_rng_state(checkpoint["rng_state"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initialization_audit = {"seed": EXP2A_PROTOCOL["seed"], "encoder_initial_state_sha256": _state_digest(model.film_encoder), "l1_mbc_initial_state_sha256": _state_digest(model.respiratory_mbc.levels[0]), "model_config_sha256": provenance["config_sha256"], "initialization_policy": "fresh deterministic Exp2-A construction after set_reproducibility(seed)", "same_seeded_initialization_expected_by_protocol": True, "exact_historical_pretraining_weight_equality_not_directly_observable": True}
    if checkpoint is not None and checkpoint.get("initialization_audit"):
        initialization_audit = checkpoint["initialization_audit"]
    before = exp2a_gauge_diagnostics(model, sampler)
    if runner.global_step < EXP2A_PROTOCOL["steps"]:
        segment = ReferenceSegment("exp2a_step_200", "stage2a", EXP2A_PROTOCOL["steps"], 1, frozenset({"respiratory_encoder", "respiratory_mbc"})); report = runner.run_segment(segment)
    else:
        report = runner.completed[-1]
    after = exp2a_gauge_diagnostics(model, sampler)
    static_initialization = {"mode": "direct_exact_state_with_coordinate_bridge", "static_model_sha256": _sha256(static_model), "internal_inr_bounds_mm": internal_bounds.tolist(), "encoding_backend": backend, "canonical_frozen": True}
    payload = {"checkpoint_schema": CHECKPOINT_SCHEMA, "model": model.state_dict(), "runner_state": runner.state_dict(), "rng_state": capture_rng_state(), "effective_config": config, "provenance": provenance, "static_initialization": static_initialization, "score_gauge": {"policy": EXP2A_PROTOCOL["score_policy"], "reference_manifest_sha256": provider.provenance["mean_manifest_sha256"], "reference_artifact_hashes": provider.provenance["mean_stacks"]}, "diagnostics": {"step_0": before, "step_200": after}, "initialization_audit": initialization_audit}
    torch.save(payload, args.output_dir / "checkpoint_step_200.pt"); torch.save(payload, args.output_dir / "checkpoint_last.pt")
    (args.output_dir / "training_report.json").write_text(json.dumps({"experiment": EXP2A_PROTOCOL["experiment"], "protocol": EXP2A_PROTOCOL, "runner_report": report, "diagnostics": payload["diagnostics"], "initialization_audit": initialization_audit, "provenance": provenance, "normalization_parameters": normalization, "normalization_groups": groups}, indent=2) + "\n")
    print(json.dumps({"status": "trained", "output": str(args.output_dir), "global_step": runner.global_step}, indent=2))


if __name__ == "__main__":
    main()
