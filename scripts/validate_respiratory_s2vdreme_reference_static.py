#!/usr/bin/env python3
"""Fail-closed validation of a NeSVoR static INR for Experiment-1 initialization."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.experiments.respiratory_s2vdreme_reference_v1 import canonical_init_provenance, strict_canonical_state_from_static_checkpoint
from cardioresp4d.training.build_model import build_source_first_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True, type=Path); parser.add_argument("--canonical-domain", required=True, type=Path)
    parser.add_argument("--static-model", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.experiment_config.read_text()); domain = json.loads(args.canonical_domain.read_text())
    model = build_source_first_model(config, domain, n_dynamic_frames=1, device=torch.device("cpu"))
    payload = torch.load(args.static_model, map_location="cpu", weights_only=False)
    strict_canonical_state_from_static_checkpoint(payload, model.canonical.inr.state_dict(), dynamic_bounds_world_mm=torch.stack((model.canonical_lower_world_mm, model.canonical_upper_world_mm)), expected_architecture=model.canonical.args)
    args.output.write_text(json.dumps(canonical_init_provenance(args.static_model, mode="direct_exact_state"), indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
