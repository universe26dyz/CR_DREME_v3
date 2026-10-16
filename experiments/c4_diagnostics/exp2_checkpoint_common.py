"""Shared frozen-checkpoint setup for Exp2-B/C diagnostics only."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from cardioresp4d.training.build_model import build_source_first_model
from cardioresp4d.training.model import SourceFirstDynamicModel
from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, _load_domain, geometry_from_manifest
from experiments.c4_diagnostics.exp2a_encoder_temporal_identifiability import validate_source_timestamps
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from experiments.c4_diagnostics.synthetic.render import PhantomCanonicalAdapter


@dataclass
class FrozenMotionAudit:
    model: SourceFirstDynamicModel
    payload: dict[str, Any]
    config: dict[str, Any]
    domain: dict[str, Any]
    motion: GroundTruthMotion
    observations: list
    groups: dict[str, list]
    recovery_arrays: Path
    timestamps_validated: bool


def group_by_location(observations: list) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for observation in observations:
        grouped[f"{observation.view}/{observation.slice_id}"].append(observation)
    return {location: sorted(sequence, key=lambda item: item.timestamp_s) for location, sequence in grouped.items()}


def reconstruct_motion_only_model(config: dict[str, Any], domain: dict[str, Any], checkpoint_state: dict[str, torch.Tensor], *, n_dynamic_frames: int, device: torch.device | None = None, allow_empty_state: bool = False) -> SourceFirstDynamicModel:
    """Rebuild Exp1 motion-only state with its phantom canonical before loading."""
    device = torch.device("cpu") if device is None else device
    model = build_source_first_model(config, domain, n_dynamic_frames=n_dynamic_frames, device=device).to(device)
    model.canonical = PhantomCanonicalAdapter(AnalyticCanonicalPhantom(domain)).to(device)
    if not checkpoint_state:
        if allow_empty_state:
            return model
        raise ValueError("motion-only checkpoint contains no model state")
    try:
        model.load_state_dict(checkpoint_state, strict=True)
    except RuntimeError as error:
        raise ValueError("motion-only checkpoint does not strictly match the reconstructed phantom-canonical model") from error
    return model


def prepare_frozen_motion_audit(args, *, require_static_collapse: bool = True) -> FrozenMotionAudit:
    """Recreate exact formal Exp1 observations and a frozen motion-only model."""
    if args.device != "cpu":
        raise ValueError("Exp2 checkpoint diagnostics are offline CPU diagnostics; use --device cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload or "summary" not in payload:
        raise ValueError("checkpoint must be an Exp1 synthetic_recovery_last.pt payload")
    summary = payload["summary"]
    if summary.get("mode") != "motion_only":
        raise ValueError("checkpoint must be an Exp1 motion_only checkpoint")
    if require_static_collapse and summary.get("failure_signature") != "static_collapse":
        raise ValueError("Exp2-B/C require the Exp1 motion_only static_collapse checkpoint")
    if int(summary.get("n_locations", -1)) != 9 or int(summary.get("n_frames", -1)) != 450:
        raise ValueError("Exp2-B/C require the formal 9-location / 450-frame source checkpoint")
    with args.source_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with args.experiment_config.open(encoding="utf-8") as handle:
        synthetic = yaml.safe_load(handle)["synthetic"]
    domain = _load_domain(args.canonical_domain)
    motion = GroundTruthMotion.from_domain(
        domain,
        respiratory_frequency_hz=float(synthetic["respiratory_frequency_hz"]),
        cardiac_frequency_hz=float(synthetic["cardiac_frequency_hz"]),
        respiratory_amplitude_mm=float(synthetic["respiratory_amplitude_mm"]),
        cardiac_amplitude_mm=float(synthetic["cardiac_amplitude_mm"]),
    )
    frames = geometry_from_manifest(args.manifest, args.qc_table, locations_per_view=args.locations_per_view, frames_per_location=args.frames_per_location, device=torch.device("cpu"))
    observations = _build_observations(
        frames,
        image_shape=tuple(int(value) for value in synthetic["image_shape"]),
        phantom=AnalyticCanonicalPhantom(domain),
        motion=motion,
        psf_samples=int(synthetic["psf_samples"]),
        seed=int(summary["provenance"]["seed"]),
    )
    groups = group_by_location(observations)
    if len(groups) != 9 or len(observations) != 450 or any(len(sequence) != 50 for sequence in groups.values()):
        raise ValueError("recreated observations are not the required formal 9 locations / 450 frames")
    recovery_arrays = args.checkpoint.parent / "recovery_arrays.npz"
    if not recovery_arrays.is_file():
        raise FileNotFoundError("Exp2-B/C require checkpoint-adjacent recovery_arrays.npz for timestamp validation")
    if not validate_source_timestamps(recovery_arrays, groups):
        raise ValueError("source recovery timestamps were not validated")
    model = reconstruct_motion_only_model(config, domain, payload["model"], n_dynamic_frames=len(observations)).eval()
    return FrozenMotionAudit(model, payload, config, domain, motion, observations, groups, recovery_arrays, True)


def deterministic_indices(size: int, count: int) -> list[int]:
    if size <= 0 or count <= 0:
        raise ValueError("deterministic selection requires positive size and count")
    return np.linspace(0, size - 1, min(size, count)).round().astype(int).tolist()


def parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in module.named_parameters()}


def assert_parameters_unchanged(module: torch.nn.Module, before: dict[str, torch.Tensor]) -> None:
    after = dict(module.named_parameters())
    if set(before) != set(after) or any(not torch.equal(before[name], parameter.detach().cpu()) for name, parameter in after.items()):
        raise RuntimeError("frozen diagnostic mutated a model parameter")


def full_image_uv(observation) -> torch.Tensor:
    height, width = observation.image.shape[-2:]
    return torch.stack(torch.meshgrid(torch.arange(width, device=observation.image.device), torch.arange(height, device=observation.image.device), indexing="xy"), dim=-1).reshape(-1, 2).to(dtype=observation.image.dtype)


def matched_psf_sample_world_points(model: SourceFirstDynamicModel, observation, pixel_uv: torch.Tensor, standard_normal_offsets: torch.Tensor) -> torch.Tensor:
    """Use the model's production PSF sigma and a cached target offset stream."""
    points = SourceFirstDynamicModel._pixel_world(observation, pixel_uv.to(observation.image))
    if standard_normal_offsets.shape != (points.shape[0], model.psf.n_samples, 3):
        raise ValueError("cached PSF offsets must be [pixels, psf_samples, xyz]")
    spacing = observation.pixel_spacing_mm.to(points)
    resolution = torch.stack((spacing[1], spacing[0], points.new_tensor(observation.slice_thickness_mm))).expand(points.shape[0], 3)
    basis = torch.stack((observation.row_direction.to(points).expand(points.shape[0], -1), observation.column_direction.to(points).expand(points.shape[0], -1), observation.normal.to(points).expand(points.shape[0], -1)), dim=-1)
    return points[:, None] + torch.einsum("nij,nsj->nsi", basis, standard_normal_offsets.to(points) * model.psf.sigma_mm(resolution).to(points)[:, None])
