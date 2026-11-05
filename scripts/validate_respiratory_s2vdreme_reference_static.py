#!/usr/bin/env python3
"""Fail-closed validation of a NeSVoR static INR for Experiment-1 initialization."""
from __future__ import annotations

import argparse
import hashlib
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


def resolve_validation_device(device_spec: str, backend: str, *, cuda_available: bool | None = None) -> torch.device:
    """Select a device compatible with the checkpoint's preserved INR backend."""
    device = torch.device(device_spec)
    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    if device.type == "cuda" and not available:
        raise ValueError("CUDA requested but unavailable for static INR validation")
    if backend == "tinycudann" and device.type != "cuda":
        raise ValueError("tinycudann static INR validation cannot run on CPU; request a CUDA device")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True, type=Path); parser.add_argument("--canonical-domain", required=True, type=Path)
    parser.add_argument("--static-model", required=True, type=Path); parser.add_argument("--output", required=True, type=Path); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = yaml.safe_load(args.experiment_config.read_text()); domain = json.loads(args.canonical_domain.read_text())
    payload = torch.load(args.static_model, map_location="cpu", weights_only=False)
    candidate = payload.get("model", payload.get("inr_state_dict"))
    if not isinstance(candidate, dict) or not isinstance(candidate.get("bounding_box"), torch.Tensor):
        raise ValueError("static checkpoint has no auditable internal INR bounding_box")
    internal_bounds = candidate["bounding_box"].detach().cpu()
    backend = detect_checkpoint_encoding_backend(candidate)
    device = resolve_validation_device(args.device, backend)
    bridge = lps_to_ras_world_matrix(device=device)
    model = build_source_first_model(config, domain, n_dynamic_frames=1, device=device, canonical_encoding_backend=backend, canonical_internal_bounds_world_mm=internal_bounds.to(device), canonical_world_to_internal_transform=bridge).to(device)
    strict_canonical_state_from_static_checkpoint(payload, model.canonical.inr.state_dict(), internal_bounds_world_mm=internal_bounds, expected_architecture=model.canonical.args, expected_encoding_backend=backend, world_to_internal_transform=model.canonical.input_world_to_internal)
    model.canonical.inr.load_state_dict(candidate, strict=True)
    external_bounds = torch.tensor([domain["world_min_mm"], domain["world_max_mm"]], dtype=torch.float32)
    if not torch.allclose(torch.stack((model.canonical_lower_world_mm, model.canonical_upper_world_mm)).cpu(), external_bounds, atol=1e-5, rtol=0.0):
        raise RuntimeError("static validator changed external LPS canonical bounds")
    if not torch.allclose(model.canonical.inr.bounding_box.cpu(), internal_bounds, atol=1e-5, rtol=0.0):
        raise RuntimeError("static validator direct load changed internal RAS INR bbox")
    report = {**canonical_init_provenance(args.static_model, mode="direct_exact_state_with_coordinate_bridge"), "accepted": True, "encoding_backend": backend, "device": str(device), "external_lps_bbox": external_bounds.tolist(), "internal_ras_bbox": internal_bounds.tolist(), "world_to_internal_matrix": [[-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], "strict_load": True, "static_model_sha256": hashlib.sha256(args.static_model.read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
