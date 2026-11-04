#!/usr/bin/env python3
"""Build deterministic per-location temporal means for Experiment 1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_source_first import observations_from_manifest
from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import build_temporal_mean_locations, export_temporal_mean_stacks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--qc-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path); parser.add_argument("--device", default="cpu")
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type != "cpu": parser.error("this preprocessing launcher is intentionally local CPU only")
    source = yaml.safe_load(args.source_config.read_text())
    observations, normalization, groups = observations_from_manifest(args.manifest, args.qc_table, device, normalization_mode=source["training"]["normalization"]["mode"], assign_slice_positions=True)
    locations = build_temporal_mean_locations(observations)
    manifest = export_temporal_mean_stacks(locations, args.output_dir)
    (args.output_dir / "mean_preprocessing_provenance.json").write_text(json.dumps({"input_manifest": str(args.manifest), "qc_table": str(args.qc_table), "normalization_parameters": normalization, "normalization_groups": groups, "mean_manifest": str(manifest)}, indent=2) + "\n")
    print(manifest)


if __name__ == "__main__":
    main()
