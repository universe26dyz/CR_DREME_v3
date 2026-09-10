#!/usr/bin/env python3
"""Run v3 unified training directly from acquired valid dynamic frames."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cardioresp4d.data.dataset import CardioRespDataset
from cardioresp4d.adapters.source_lock import verify_vendored_source_lock
from cardioresp4d.frequency.training_prior import load_training_frequency_prior
from cardioresp4d.geometry.world_geometry import DicomPlane
from cardioresp4d.training.model import SourceFirstDynamicModel
from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler
from cardioresp4d.training.source_first_config import validate_source_dependencies, validate_source_first_config
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer


def observations_from_manifest(manifest: Path, qc_table: Path, device: torch.device, *, normalization_mode: str) -> tuple[list[DynamicObservation], dict[str, dict[str, float]]]:
    dataset = CardioRespDataset(manifest, valid_only=False, qc_table_path=qc_table, normalization_mode=normalization_mode)
    ids = {token: index for index, token in enumerate(sorted(row.get("source_file_token", f"{row['view']}:{row['slice_id']}:{row['frame_index']}") for row in dataset._rows))}
    observations: list[DynamicObservation] = []
    for index, row in enumerate(dataset._rows):
        sample = dataset[index]; plane = DicomPlane.from_geometry(sample["geometry"])
        token = row.get("source_file_token", f"{row['view']}:{row['slice_id']}:{row['frame_index']}")
        observations.append(DynamicObservation(image=torch.from_numpy(sample["image"])[None].to(device), view=sample["view"].upper(), slice_id=sample["slice_id"], dynamic_frame_id=ids[token], center_mm=torch.from_numpy(plane.pixel_to_world((plane.columns - 1) / 2., (plane.rows - 1) / 2.)).float().to(device), row_direction=torch.from_numpy(plane.row_direction).float().to(device), column_direction=torch.from_numpy(plane.column_direction).float().to(device), normal=torch.from_numpy(plane.normal).float().to(device), pixel_spacing_mm=torch.from_numpy(plane.pixel_spacing).float().to(device), slice_thickness_mm=float(plane.slice_thickness or row["slice_thickness"]), qc_valid=bool(sample["qc_valid"]), qc_reason=str(sample["qc_reason"]), timestamp_s=float(sample["timestamp_s"])))
    return observations, dataset.normalization_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="v3 source-first unified Stage1 -> Stage2a real-data validation")
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs" / "source_first.yaml")
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--qc-table", required=True, type=Path)
    parser.add_argument("--canonical-domain", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frequency-bands", type=Path, help="Phase-1 aggregate frequency_bands.json; overrides config-relative path")
    parser.add_argument("--device", default="cuda"); parser.add_argument("--pixel-samples", type=int, default=256)
    parser.add_argument("--stage1-steps", type=int, default=100); parser.add_argument("--stage2a-steps", type=int, default=100)
    parser.add_argument("--stage2b-steps", type=int, default=0); parser.add_argument("--stage2c-steps", type=int, default=0); parser.add_argument("--stage3-steps", type=int, default=0)
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    with args.source_config.open(encoding="utf-8") as handle: source_config = yaml.safe_load(handle)
    validate_source_first_config(source_config, PROJECT_ROOT)
    validate_source_dependencies()
    source_lock = verify_vendored_source_lock(PROJECT_ROOT)
    with args.canonical_domain.open(encoding="utf-8") as handle: domain = json.load(handle)
    observations, normalization_parameters = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=source_config["training"]["normalization"]["mode"])
    cardiac = domain["cardiac_box"]
    model = SourceFirstDynamicModel(torch.tensor(domain["world_min_mm"], device=device), torch.tensor(domain["world_max_mm"], device=device), cardiac_lower_world_mm=torch.tensor(cardiac["min_mm"], device=device), cardiac_upper_world_mm=torch.tensor(cardiac["max_mm"], device=device), n_dynamic_frames=len(observations)).to(device)
    bands_path = (args.frequency_bands or (args.source_config.parent / source_config["training"]["temporal_auxiliary"]["frequency_bands_json"])).resolve()
    prior = load_training_frequency_prior(bands_path, allow_template_fallback=bool(source_config["training"].get("frequency_prior",{}).get("allow_template_fallback",False)))
    trainer = UnifiedProgressiveTrainer(model, ViewLocationBalancedSampler(observations), pixel_samples=args.pixel_samples, loss_weights=source_config["training"]["loss_weights"], frequency_prior=prior, temporal_every=int(source_config["training"]["temporal_auxiliary"]["every_steps"]), temporal_batch_size=int(source_config["training"]["temporal_auxiliary"]["max_frames"]), cardiac_sampling_fraction=float(source_config["training"]["cardiac_sampling_fraction"]))
    stages = (("stage1", args.stage1_steps), ("stage2a", args.stage2a_steps), ("stage2b", args.stage2b_steps), ("stage2c", args.stage2c_steps), ("stage3", args.stage3_steps))
    report = {"stages": {stage: trainer.run_stage(stage, steps=steps) for stage, steps in stages if steps > 0}, "effective_config": source_config, "normalization_parameters": normalization_parameters, "frequency_prior": prior.__dict__, "source_lock": source_lock}
    args.output_dir.mkdir(parents=True, exist_ok=True); torch.save({"model": model.state_dict(), "report": report}, args.output_dir / "source_first_last.pt")
    (args.output_dir / "source_first_training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "effective_config.json").write_text(json.dumps(report["effective_config"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
