#!/usr/bin/env python3
"""Read-only C4 paper-aligned one-update loss-scale sanity gate."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src")); sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cardioresp4d.adapters.source_lock import verify_vendored_source_lock
from cardioresp4d.frequency.training_prior import load_training_frequency_prior
from cardioresp4d.training.build_model import build_source_first_model, effective_model_config
from cardioresp4d.training.runtime_state import set_reproducibility, training_dynamic_frame_count
from cardioresp4d.training.sampler import ViewLocationBalancedSampler
from cardioresp4d.training.source_first_config import validate_source_dependencies, validate_source_first_config
from cardioresp4d.training.stage_contract import StageSegment, paperaligned_segment, stage_contract
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer, effective_loss_weights
from train_source_first import observations_from_manifest


def _mean_components(rows: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    return {name: sum(float(row[name].detach()) for row in rows) / len(rows) for name in rows[0]}


def nonfinite_trainable_gradient_names(model: torch.nn.Module) -> list[str]:
    """Return trainable parameters lacking a finite backward result."""
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all())]


def backward_finite_check(trainer: UnifiedProgressiveTrainer, segment: StageSegment) -> dict[str, object]:
    """Run one read-only mean-loss backward pass for a formal C4 segment."""
    trainer._configure_stage(segment.stage, segment=segment)
    assert trainer.optimizer is not None
    trainer.model.train(); trainer.optimizer.zero_grad(set_to_none=True)
    for observation in trainer.sampler.sample_step(observations_per_update=trainer.observations_per_update):
        (trainer._base_loss(trainer._observation_components(observation, segment.stage), segment.stage) / trainer.observations_per_update).backward()
    if stage_contract(segment.stage).enable_motion:
        temporal_loss = trainer._temporal_loss(trainer._temporal_components(segment.stage), segment.stage)
        if temporal_loss.requires_grad:
            temporal_loss.backward()
    bad_parameter_names = nonfinite_trainable_gradient_names(trainer.model)
    return {"finite": not bad_parameter_names, "bad_parameter_names": bad_parameter_names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs" / "source_first_change4_paperaligned.yaml")
    parser.add_argument("--frequency-bands", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True)
    parser.add_argument("--segment", default="s3b_full"); parser.add_argument("--device", default="cpu"); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); validate_source_first_config(config, PROJECT_ROOT); validate_source_dependencies()
    weights = effective_loss_weights(config["training"].get("loss_weights"))
    if weights["cardiac_target_concentration"] != 0. or weights["cardiac_pca_waveform"] != 0.:
        parser.error("paper-aligned C4 preflight requires cardiac_target_concentration = cardiac_pca_waveform = 0")
    segment = paperaligned_segment(args.segment); set_reproducibility(args.seed)
    domain = json.loads(args.canonical_domain.read_text(encoding="utf-8"))
    observations, _, _ = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=config["training"]["normalization"]["mode"])
    model = build_source_first_model(config, domain, n_dynamic_frames=training_dynamic_frame_count(observations), device=device).to(device)
    temporal = config["training"]["temporal_auxiliary"]
    trainer = UnifiedProgressiveTrainer(model, ViewLocationBalancedSampler(observations, seed=args.seed), pixel_samples=256, observations_per_update=int(config["training"].get("observations_per_update", 32)), optimizer_config=config["training"]["optimizer"], loss_weights=config["training"]["loss_weights"], frequency_prior=load_training_frequency_prior(args.frequency_bands, allow_template_fallback=False), pca_waveform_prior=None, temporal_every=int(temporal["every_steps"]), temporal_batch_size=int(temporal["max_frames"]), cardiac_sampling_fraction=float(config["training"]["cardiac_sampling_fraction"]))
    trainer._configure_stage(segment.stage, segment=segment); model.train()
    with torch.no_grad():
        observation_rows = [trainer._observation_components(item, segment.stage) for item in trainer.sampler.sample_step(observations_per_update=trainer.observations_per_update)]
        raw = _mean_components(observation_rows)
        raw.update({name: float(value.detach()) for name, value in trainer._temporal_components(segment.stage).items()})
    requested = {"image": weights["image"], "mbc_normalization": weights["mbc_normalization"], "zero_mean_score": weights["zero_mean_score"], "cardiac_leakage_in_resp": weights["cardiac_leakage_in_resp"], "respiratory_leakage_in_card": weights["respiratory_leakage_in_card"]}
    data = raw["data"]
    if not math.isfinite(data) or data <= 0:
        raise ValueError("loss-scale preflight requires finite positive data loss")
    components = {}
    for name, weight in requested.items():
        value, weighted = raw[name], raw[name] * weight
        if not math.isfinite(value) or not math.isfinite(weighted):
            raise ValueError(f"non-finite loss-scale component: {name}")
        components[name] = {"raw_component": value, "weight": weight, "weighted_component": weighted, "weighted_over_data_loss": weighted / data}
    warnings = [f"{name}: weighted regularizer exceeds 100x data loss" for name, values in components.items() if abs(values["weighted_over_data_loss"]) > 100.]
    backward_checks = {name: backward_finite_check(trainer, paperaligned_segment(name)) for name in ("s1a", "s2a_init", "s3a", "s3b_full")}
    checks_finite = all(bool(check["finite"]) for check in backward_checks.values())
    payload = {"status": "PASS_READ_ONLY_NO_OPTIMIZER_STEP" if checks_finite else "FAIL_NONFINITE_BACKWARD_GRADIENT", "segment": segment.name, "stage": segment.stage, "observations_per_update": trainer.observations_per_update, "raw_data_loss": data, "components": components, "warnings": warnings, "backward_checks": backward_checks, "effective_config": {**effective_model_config(model), "loss_weights": weights}, "frequency_prior": asdict(trainer.frequency_prior), "source_lock": verify_vendored_source_lock(PROJECT_ROOT), "seed": args.seed}
    args.output_json.parent.mkdir(parents=True, exist_ok=True); args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "warnings": warnings, "output": str(args.output_json)}, indent=2))
    if not checks_finite:
        raise RuntimeError("non-finite trainable gradient in C4 backward preflight")


if __name__ == "__main__":
    main()
