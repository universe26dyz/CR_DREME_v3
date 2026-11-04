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
from cardioresp4d.adapters.nesvor_inr import detect_checkpoint_encoding_backend, lps_to_ras_world_matrix
from cardioresp4d.training.build_model import build_source_first_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True, type=Path); parser.add_argument("--canonical-domain", required=True, type=Path)
    parser.add_argument("--static-model", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.experiment_config.read_text()); domain = json.loads(args.canonical_domain.read_text())
    payload = torch.load(args.static_model, map_location="cpu", weights_only=False)
    candidate = payload.get("model", payload.get("inr_state_dict"))
    if not isinstance(candidate, dict) or not isinstance(candidate.get("bounding_box"), torch.Tensor):
        raise ValueError("static checkpoint has no auditable internal INR bounding_box")
    internal_bounds = candidate["bounding_box"].detach().cpu()
    backend = detect_checkpoint_encoding_backend(candidate)
    bridge = lps_to_ras_world_matrix()
    model = build_source_first_model(config, domain, n_dynamic_frames=1, device=torch.device("cpu"), canonical_encoding_backend=backend, canonical_internal_bounds_world_mm=internal_bounds, canonical_world_to_internal_transform=bridge)
    strict_canonical_state_from_static_checkpoint(payload, model.canonical.inr.state_dict(), internal_bounds_world_mm=internal_bounds, expected_architecture=model.canonical.args, expected_encoding_backend=backend, world_to_internal_transform=model.canonical.input_world_to_internal)
    args.output.write_text(json.dumps(canonical_init_provenance(args.static_model, mode="direct_exact_state_with_coordinate_bridge"), indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
