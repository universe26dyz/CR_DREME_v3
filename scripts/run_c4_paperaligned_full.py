#!/usr/bin/env python3
"""Server-oriented, resumable launcher for the formal 6250-update C4 schedule."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cardioresp4d.adapters.source_lock import verify_vendored_source_lock
from cardioresp4d.training.source_first_config import validate_source_first_config
from cardioresp4d.training.stage_contract import paperaligned_segments


def segment_schedule() -> tuple[tuple[str, str], ...]:
    cumulative = 0
    result = []
    for segment in paperaligned_segments():
        cumulative += segment.steps
        result.append((f"{segment.name}_{cumulative}", segment.name))
    return tuple(result)


def next_unfinished_segment(output_dir: Path, schedule: tuple[tuple[str, str], ...]) -> tuple[int, Path | None]:
    """Verify contiguous completed lineage and return the next segment index."""
    parent: Path | None = None
    for index, (directory_name, segment_name) in enumerate(schedule):
        checkpoint = output_dir / directory_name / "source_first_last.pt"
        if not checkpoint.is_file():
            later = [output_dir / name / "source_first_last.pt" for name, _ in schedule[index + 1:]]
            if any(path.exists() for path in later):
                raise ValueError("paper-aligned segment outputs are not a contiguous resume lineage")
            return index, parent
        state = torch.load(checkpoint, map_location="cpu", weights_only=False).get("training_state", {})
        if state.get("current_segment") != segment_name:
            raise ValueError(f"{checkpoint} does not record completed segment {segment_name}")
        parent = checkpoint
    return len(schedule), parent


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prepare_training_output_dir(output_dir: Path, *, resume: bool) -> None:
    """Create an empty run root without weakening normal overwrite protection."""
    if output_dir.exists() and not resume and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite existing output; use --resume after checking its lineage")
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=PROJECT_ROOT / "configs" / "source_first_change4_paperaligned.yaml")
    parser.add_argument("--frequency-bands", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/data/dengyz/dataset/CR_DREME_v3/v1_change4_paperaligned_full")); parser.add_argument("--device", default="cuda"); parser.add_argument("--pixel-samples", type=int, default=256); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for path in (args.source_config, args.frequency_bands, args.manifest, args.qc_table, args.canonical_domain):
        if not path.is_file(): parser.error(f"required input is missing: {path}")
    config = yaml.safe_load(args.source_config.read_text(encoding="utf-8")); validate_source_first_config(config, PROJECT_ROOT)
    weights = config["training"].get("loss_weights", {})
    if float(weights.get("cardiac_target_concentration", 0.)) != 0. or float(weights.get("cardiac_pca_waveform", 0.)) != 0.:
        parser.error("formal C4 requires cardiac_target_concentration = cardiac_pca_waveform = 0")
    schedule = segment_schedule()
    prepare_training_output_dir(args.output_dir, resume=args.resume)
    logs = args.output_dir / "logs"; logs.mkdir(exist_ok=True)
    index, parent = next_unfinished_segment(args.output_dir, schedule)
    if index and not args.resume:
        raise FileExistsError("completed segments exist; use --resume to continue their verified lineage")
    manifest_path = args.output_dir / "c4_paperaligned_run_manifest.json"
    run_manifest = {"status": "running", "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(), "source_config": str(args.source_config), "effective_requested_config": config, "source_lock": verify_vendored_source_lock(PROJECT_ROOT), "schedule": [{"directory": directory, "segment": segment} for directory, segment in schedule], "completed": []}
    if manifest_path.is_file() and args.resume:
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8")); run_manifest["status"] = "running"
    for directory_name, segment_name in schedule[:index]:
        checkpoint = args.output_dir / directory_name / "source_first_last.pt"
        run_manifest["completed"] = [entry for entry in run_manifest.get("completed", []) if entry["segment"] != segment_name]
        run_manifest["completed"].append({"segment": segment_name, "checkpoint": str(checkpoint), "status": "verified_existing"})
    _write_manifest(manifest_path, run_manifest)
    for directory_name, segment_name in schedule[index:]:
        destination = args.output_dir / directory_name
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite incomplete segment directory: {destination}")
        command = [sys.executable, "scripts/train_source_first.py", "--source-config", str(args.source_config), "--frequency-bands", str(args.frequency_bands), "--manifest", str(args.manifest), "--qc-table", str(args.qc_table), "--canonical-domain", str(args.canonical_domain), "--output-dir", str(destination), "--device", args.device, "--pixel-samples", str(args.pixel_samples), "--seed", str(args.seed), "--segment", segment_name]
        if parent is not None: command.extend(("--resume", str(parent)))
        started = time.monotonic()
        with (logs / f"{directory_name}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        parent = destination / "source_first_last.pt"
        if not parent.is_file(): raise RuntimeError(f"segment did not produce a checkpoint: {parent}")
        run_manifest["completed"].append({"segment": segment_name, "checkpoint": str(parent), "runtime_seconds": time.monotonic() - started, "status": "completed"})
        _write_manifest(manifest_path, run_manifest)
        print(json.dumps({"segment": segment_name, "checkpoint": str(parent)}, separators=(",", ":")))
    run_manifest["status"] = "completed"; _write_manifest(manifest_path, run_manifest)


if __name__ == "__main__":
    main()
