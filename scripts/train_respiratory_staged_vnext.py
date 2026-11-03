#!/usr/bin/env python3
"""Formal launcher for the fixed respiratory-only vNext A--E protocol."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_source_first import observations_from_manifest
from cardioresp4d.adapters.nesvor_inr import detect_checkpoint_encoding_backend
from cardioresp4d.adapters.source_lock import verify_vendored_source_lock
from cardioresp4d.frequency.pca_waveform_prior import PCAWaveformPrior
from cardioresp4d.frequency.training_prior import load_training_frequency_prior
from cardioresp4d.training.build_model import build_source_first_model, effective_model_config, uses_s2v_style_film
from cardioresp4d.training.respiratory_staged_vnext import RespiratoryStagedVNextRunner, respiratory_staged_vnext_segments, validate_respiratory_staged_vnext_config
from cardioresp4d.training.runtime_state import capture_rng_state, checkpoint_compatible_dynamic_frame_count, restore_rng_state, set_reproducibility
from cardioresp4d.training.sampler import ViewLocationBalancedSampler
from cardioresp4d.training.source_first_config import validate_source_dependencies, validate_source_first_config
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer

CHECKPOINT_SCHEMA = 1


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()


def checkpoint_name_for_segment(segment) -> str:
    return {"resp_static_bootstrap_a1": "checkpoint_A1_end.pt", "resp_static_bootstrap_a2": "checkpoint_A_end.pt", "resp_pca_warmup": "checkpoint_B_end.pt", "resp_mbc_bootstrap_c1": "checkpoint_C1_end.pt", "resp_mbc_bootstrap_c2": "checkpoint_C2_end.pt", "resp_mbc_bootstrap_c3": "checkpoint_C_end.pt", "resp_score_refine": "checkpoint_D_end.pt", "resp_canonical_refine": "checkpoint_E_end.pt"}[segment.name]


def resume_next_segment_index(checkpoint: Mapping[str, Any], segments, provenance: Mapping[str, Any]) -> int:
    if checkpoint.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("--resume requires respiratory vNext checkpoint_schema=1")
    if checkpoint.get("provenance") != dict(provenance):
        raise ValueError("--resume provenance does not match the effective vNext protocol inputs")
    state = checkpoint.get("runner_state")
    if not isinstance(state, Mapping):
        raise ValueError("--resume lacks respiratory vNext runner_state")
    completed = state.get("completed")
    if not isinstance(completed, list) or not completed:
        raise ValueError("--resume must be a completed respiratory vNext segment boundary")
    index = len(completed) - 1
    if index >= len(segments) or completed[index].get("segment") != segments[index].name:
        raise ValueError("--resume completed schedule is not a respiratory vNext prefix")
    if state.get("current_segment") != segments[index].name or int(state.get("segment_step", -1)) != segments[index].steps:
        raise ValueError("--resume is not at a completed segment boundary; replay is forbidden")
    if int(state.get("global_step", -1)) != sum(item.steps for item in segments[: index + 1]):
        raise ValueError("--resume global_step is inconsistent with its completed segment boundary")
    return index + 1


def _write_metrics(output_dir: Path, reports: list[dict[str, object]]) -> None:
    (output_dir / "segment_metrics.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "segment_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in reports for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(reports)


def _checkpoint_payload(*, model, runner, config: Mapping[str, Any], provenance: Mapping[str, Any], normalization_parameters, normalization_groups, frequency_prior, pca_prior, source_lock) -> dict[str, object]:
    return {"checkpoint_schema": CHECKPOINT_SCHEMA, "git_sha": _git_head(), "model": model.state_dict(), "runner_state": runner.state_dict(), "optimizer": runner.last_optimizer_state, "sampler": runner.trainer.sampler.state_dict(), "rng_state": capture_rng_state(), "effective_config": dict(config), "normalization_parameters": normalization_parameters, "normalization_groups": normalization_groups, "frequency_prior": asdict(frequency_prior), "respiratory_pca_waveform_prior": pca_prior.metadata, "source_lock": source_lock, "provenance": dict(provenance)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the formal fixed respiratory-vNext A--E protocol")
    parser.add_argument("--source-config", required=True, type=Path); parser.add_argument("--vnext-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--qc-table", required=True, type=Path); parser.add_argument("--canonical-domain", required=True, type=Path)
    parser.add_argument("--frequency-bands", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--pixel-samples", type=int, default=256); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--resume", type=Path)
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    source = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); overlay = yaml.safe_load(args.vnext_config.read_text(encoding="utf-8")); config = _deep_merge(source, overlay)
    validate_source_first_config(config, PROJECT_ROOT); validate_respiratory_staged_vnext_config(config); validate_source_dependencies()
    source_lock = verify_vendored_source_lock(PROJECT_ROOT)
    domain = json.loads(args.canonical_domain.read_text(encoding="utf-8")); reproducibility = set_reproducibility(args.seed)
    bands_path = args.frequency_bands.resolve()
    provenance = {"effective_config": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(), "seed": args.seed, "manifest": _sha256(args.manifest), "qc_table": _sha256(args.qc_table), "canonical_domain": _sha256(args.canonical_domain), "frequency_bands": _sha256(bands_path), "respiratory_pca": _sha256(bands_path), "source_lock": source_lock}
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    observations, normalization_parameters, normalization_groups = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=config["training"]["normalization"]["mode"], assign_slice_positions=uses_s2v_style_film(config))
    n_dynamic_frames = checkpoint_compatible_dynamic_frame_count(checkpoint, observations)
    backend = None if checkpoint is None else detect_checkpoint_encoding_backend(checkpoint["model"])
    model = build_source_first_model(config, domain, n_dynamic_frames=n_dynamic_frames, device=device, canonical_encoding_backend=backend).to(device)
    if checkpoint is not None: model.load_state_dict(checkpoint["model"])
    frequency_prior = load_training_frequency_prior(bands_path, allow_template_fallback=False)
    respiratory_pca_prior = PCAWaveformPrior.load(bands_path, component="respiratory", strict=True)
    temporal = config["training"]["temporal_auxiliary"]
    trainer = UnifiedProgressiveTrainer(model, ViewLocationBalancedSampler(observations, seed=args.seed), pixel_samples=args.pixel_samples, observations_per_update=int(config["training"]["observations_per_update"]), optimizer_config=config["training"]["optimizer"], loss_weights=config["training"]["loss_weights"], frequency_prior=frequency_prior, pca_waveform_prior=respiratory_pca_prior, pca_waveform_ridge=float(temporal.get("pca_waveform_ridge", 1e-4)), pca_waveform_min_frames=int(temporal.get("pca_waveform_min_frames", 8)), temporal_every=int(temporal["every_steps"]), temporal_batch_size=int(temporal["max_frames"]), cardiac_sampling_fraction=float(config["training"]["cardiac_sampling_fraction"]))
    runner = RespiratoryStagedVNextRunner(trainer, respiratory_pca_waveform_prior=respiratory_pca_prior, loss_weights=config["training"]["loss_weights"])
    segments = respiratory_staged_vnext_segments(); start = 0
    if checkpoint is not None:
        start = resume_next_segment_index(checkpoint, segments, provenance); runner.load_boundary_state(checkpoint["runner_state"]); restore_rng_state(checkpoint["rng_state"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for segment in segments[start:]:
        runner.run_segment(segment)
        payload = _checkpoint_payload(model=model, runner=runner, config=config, provenance=provenance, normalization_parameters=normalization_parameters, normalization_groups=normalization_groups, frequency_prior=frequency_prior, pca_prior=respiratory_pca_prior, source_lock=source_lock)
        torch.save(payload, args.output_dir / checkpoint_name_for_segment(segment)); torch.save(payload, args.output_dir / "checkpoint_last.pt")
        _write_metrics(args.output_dir, runner.completed)
    report = {"protocol": "respiratory_staged_vnext_fixed_A_to_E", "segments": runner.completed, "effective_config": {**effective_model_config(model), "optimizer_config": config["training"]["optimizer"], "loss_weights": config["training"]["loss_weights"], "observations_per_update": trainer.observations_per_update}, "normalization_parameters": normalization_parameters, "normalization_groups": normalization_groups, "frequency_prior": asdict(frequency_prior), "respiratory_pca_waveform_prior": respiratory_pca_prior.metadata, "source_lock": source_lock, "provenance": provenance, "reproducibility": reproducibility, "device": str(device), "torch_version": torch.__version__, "resumed_from": None if args.resume is None else str(args.resume)}
    (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "effective_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
