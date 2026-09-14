#!/usr/bin/env python3
"""Run v3 unified training directly from acquired valid dynamic frames."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cardioresp4d.data.dataset import CardioRespDataset
from cardioresp4d.adapters.nesvor_inr import detect_checkpoint_encoding_backend
from cardioresp4d.adapters.source_lock import verify_vendored_source_lock
from cardioresp4d.frequency.training_prior import load_training_frequency_prior
from cardioresp4d.geometry.world_geometry import DicomPlane
from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler
from cardioresp4d.training.build_model import build_source_first_model, effective_model_config
from cardioresp4d.training.runtime_state import capture_rng_state, is_hard_invalid_reason, restore_rng_state, set_reproducibility, validate_dynamic_frame_rows
from cardioresp4d.training.source_first_config import validate_source_dependencies, validate_source_first_config
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer


def observations_from_manifest(manifest: Path, qc_table: Path, device: torch.device, *, normalization_mode: str) -> tuple[list[DynamicObservation], dict[str, dict[str, float]], list[dict]]:
    dataset = CardioRespDataset(manifest, valid_only=False, qc_table_path=qc_table, normalization_mode=normalization_mode)
    validate_dynamic_frame_rows(dataset._rows)
    valid_tokens = sorted(row["source_file_token"] for row in dataset._rows if str(row.get("qc_valid", "1")).lower() not in {"0", "false"} and not is_hard_invalid_reason(row.get("qc_reason")))
    ids = {token: index for index, token in enumerate(valid_tokens)}
    observations: list[DynamicObservation] = []
    for index, row in enumerate(dataset._rows):
        sample = dataset[index]; plane = DicomPlane.from_geometry(sample["geometry"])
        token = row["source_file_token"]
        observations.append(DynamicObservation(image=torch.from_numpy(sample["image"])[None].to(device), view=sample["view"].upper(), slice_id=sample["slice_id"], dynamic_frame_id=ids.get(token, -1), center_mm=torch.from_numpy(plane.pixel_to_world((plane.columns - 1) / 2., (plane.rows - 1) / 2.)).float().to(device), row_direction=torch.from_numpy(plane.row_direction).float().to(device), column_direction=torch.from_numpy(plane.column_direction).float().to(device), normal=torch.from_numpy(plane.normal).float().to(device), pixel_spacing_mm=torch.from_numpy(plane.pixel_spacing).float().to(device), slice_thickness_mm=float(plane.slice_thickness or row["slice_thickness"]), qc_valid=bool(sample["qc_valid"]), qc_reason=str(sample["qc_reason"]), timestamp_s=float(sample["timestamp_s"])))
    return observations, dataset.normalization_parameters, dataset.normalization_group_report


def main() -> None:
    parser = argparse.ArgumentParser(description="v3 source-first unified Stage1 -> Stage2a real-data validation")
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs" / "source_first.yaml")
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--qc-table", required=True, type=Path)
    parser.add_argument("--canonical-domain", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frequency-bands", type=Path, help="Phase-1 aggregate frequency_bands.json; overrides config-relative path")
    parser.add_argument("--device", default="cuda"); parser.add_argument("--pixel-samples", type=int, default=256)
    parser.add_argument("--stage1-steps", type=int, default=100); parser.add_argument("--stage2a-steps", type=int, default=100)
    parser.add_argument("--stage2b-steps", type=int, default=0); parser.add_argument("--stage2c-steps", type=int, default=0)
    parser.add_argument("--stage3a-steps", type=int, default=0); parser.add_argument("--stage3b-steps", type=int, default=0); parser.add_argument("--stage3c-steps", type=int, default=0)
    parser.add_argument("--stage3-steps", type=int, default=0, help="Deprecated; use --stage3a-steps/--stage3b-steps/--stage3c-steps")
    parser.add_argument("--seed", type=int, help="Overrides training.seed and is recorded in the checkpoint/report")
    parser.add_argument("--resume", type=Path, help="Complete source-first checkpoint created by this entry point")
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    if args.stage3_steps:
        parser.error("--stage3-steps is deprecated after v3_change4; use --stage3a-steps / --stage3b-steps / --stage3c-steps")
    with args.source_config.open(encoding="utf-8") as handle: source_config = yaml.safe_load(handle)
    validate_source_first_config(source_config, PROJECT_ROOT)
    validate_source_dependencies()
    source_lock = verify_vendored_source_lock(PROJECT_ROOT)
    with args.canonical_domain.open(encoding="utf-8") as handle: domain = json.load(handle)
    seed = int(source_config["training"].get("seed", 0) if args.seed is None else args.seed)
    reproducibility = set_reproducibility(seed)
    checkpoint = None; checkpoint_backend = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_schema") != 1:
            raise ValueError("--resume requires source-first checkpoint_schema=1")
        checkpoint_backend = detect_checkpoint_encoding_backend(checkpoint["model"])
    observations, normalization_parameters, normalization_groups = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=source_config["training"]["normalization"]["mode"])
    n_dynamic_frames = sum(1 for observation in observations if observation.qc_valid)
    model = build_source_first_model(source_config, domain, n_dynamic_frames=n_dynamic_frames, device=device, canonical_encoding_backend=checkpoint_backend).to(device)
    bands_path = (args.frequency_bands or (args.source_config.parent / source_config["training"]["temporal_auxiliary"]["frequency_bands_json"])).resolve()
    prior = load_training_frequency_prior(bands_path, allow_template_fallback=bool(source_config["training"].get("frequency_prior",{}).get("allow_template_fallback",False)))
    trainer = UnifiedProgressiveTrainer(model, ViewLocationBalancedSampler(observations, seed=seed), pixel_samples=args.pixel_samples, optimizer_config=source_config["training"]["optimizer"], loss_weights=source_config["training"]["loss_weights"], frequency_prior=prior, temporal_every=int(source_config["training"]["temporal_auxiliary"]["every_steps"]), temporal_batch_size=int(source_config["training"]["temporal_auxiliary"]["max_frames"]), cardiac_sampling_fraction=float(source_config["training"]["cardiac_sampling_fraction"]))
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        trainer.load_training_state_dict(checkpoint["training_state"])
        restore_rng_state(checkpoint["rng_state"])
    stages = (("stage1", args.stage1_steps), ("stage2a", args.stage2a_steps), ("stage2b", args.stage2b_steps), ("stage2c", args.stage2c_steps), ("stage3a", args.stage3a_steps), ("stage3b", args.stage3b_steps), ("stage3c", args.stage3c_steps))
    prior_payload = asdict(prior)
    report = {"stages": {stage: trainer.run_stage(stage, steps=steps) for stage, steps in stages if steps > 0}, "effective_config": {**effective_model_config(model), "optimizer_config": source_config["training"]["optimizer"]}, "canonical_encoding_backend": model.canonical.encoding_backend, "normalization_parameters": normalization_parameters, "normalization_groups": normalization_groups, "frequency_prior": prior_payload, "source_lock": source_lock, "source_lock_verified": bool(source_lock.get("verified")), "reproducibility": reproducibility, "device": str(device), "torch_version": torch.__version__, "resumed_from": str(args.resume) if args.resume else None}
    args.output_dir.mkdir(parents=True, exist_ok=True); torch.save({"checkpoint_schema": 1, "model": model.state_dict(), "training_state": trainer.training_state_dict(), "rng_state": capture_rng_state(), "report": report, "frequency_prior": prior_payload, "normalization_parameters": normalization_parameters}, args.output_dir / "source_first_last.pt")
    (args.output_dir / "source_first_training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "effective_config.json").write_text(json.dumps(report["effective_config"], indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "training_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = trainer.metrics; fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
