"""Reusable, deterministic dynamic checkpoint audit primitives."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from cardioresp4d.adapters.nesvor_inr import detect_checkpoint_encoding_backend, lps_to_ras_world_matrix
from cardioresp4d.training.build_model import build_source_first_model
from cardioresp4d.diagnostics.change5c import diagnostic_rng, stable_diagnostic_seed
from cardioresp4d.training.stage_contract import stage_contract
from cardioresp4d.training.respiratory_staged_vnext import respiratory_dvf_rms_metrics, respiratory_score_rms_metrics
from .dynamics import world_grid


@dataclass(frozen=True)
class CheckpointBoundary:
    label: str
    stage: str
    active_levels: int
    path: Path


_BOUNDARIES = (
    ("L1_warm", "stage2a"), ("L1_joint", "stage2a"),
    ("L2_warm", "stage2b"), ("L2_joint", "stage2b"),
    ("L3_warm", "stage2c"), ("L3_joint", "stage2c"),
)


def select_boundary_subset(boundaries: Iterable[CheckpointBoundary], requested: str | None) -> list[CheckpointBoundary]:
    """Return a validated, protocol-ordered boundary subset.

    ``requested`` is the comma-separated CLI spelling used by retrospective
    audits.  Validation happens before filtering so a typo cannot silently
    produce an empty or scientifically different audit.
    """
    available = list(boundaries)
    if requested is None or not requested.strip():
        return available
    names = [item.strip() for item in requested.split(",") if item.strip()]
    known = {label for label, _ in _BOUNDARIES} | {"last"}
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"unknown boundary name(s): {', '.join(unknown)}")
    if len(names) != len(set(names)):
        raise ValueError("boundary names must be unique")
    by_name = {item.label: item for item in available}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise FileNotFoundError(f"requested boundary checkpoint(s) are unavailable: {', '.join(missing)}")
    return [by_name[name] for name in names]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_dynamic_checkpoint_root(root: str | Path) -> Path | None:
    """Resolve either an experiment root or its direct ``04_training`` root."""
    root = Path(root)
    if (root / "checkpoint_L1_warm_end.pt").is_file():
        return root
    training = root / "04_training"
    if (training / "checkpoint_L1_warm_end.pt").is_file():
        return training
    return None


def discover_boundary_checkpoints(root: str | Path) -> list[CheckpointBoundary]:
    root = resolve_dynamic_checkpoint_root(root)
    if root is None:
        return []
    result = []
    for label, stage in _BOUNDARIES:
        path = root / f"checkpoint_{label}_end.pt"
        if path.is_file():
            result.append(CheckpointBoundary(label, stage, stage_contract(stage).active_respiratory_levels, path))
    last = root / "checkpoint_last.pt"
    if last.is_file() and (not result or _sha256(last) != _sha256(result[-1].path)):
        result.append(CheckpointBoundary("last", result[-1].stage if result else "stage2c", result[-1].active_levels if result else 3, last))
    return result


def location_local_rms_metrics(values_by_location: Iterable[torch.Tensor], *, kind: str = "score") -> dict[str, float]:
    values = tuple(values_by_location)
    if kind == "dvf":
        raw = respiratory_dvf_rms_metrics(values)
        total, static, dynamic = raw["respiratory_dvf_total_rms_mm"], raw["respiratory_dvf_static_rms_mm"], raw["respiratory_dvf_dynamic_rms_mm"]
        return {"static_rms_mm": static, "dynamic_rms_mm": dynamic, "total_rms_mm": total, "static_dynamic_ratio": static / max(dynamic, torch.finfo(torch.float32).eps)}
    raw = respiratory_score_rms_metrics(values)
    total, static, dynamic = raw["respiratory_score_total_rms"], raw["respiratory_score_static_rms"], raw["respiratory_score_dynamic_rms"]
    return {"static_rms": static, "dynamic_rms": dynamic, "total_rms": total, "static_dynamic_ratio": static / max(dynamic, torch.finfo(torch.float32).eps)}


def pool_location_metrics_by_view(values_by_view: Mapping[str, Iterable[torch.Tensor]], *, kind: str = "score") -> dict[str, dict[str, float]]:
    return {view: location_local_rms_metrics(values, kind=kind) for view, values in values_by_view.items()}


def select_fixed_location_frames(observations: Iterable[Any], *, frames_per_location: int = 3) -> dict[tuple[str, str], list[Any]]:
    if frames_per_location <= 0:
        raise ValueError("frames_per_location must be positive")
    grouped: dict[str, dict[str, list[Any]]] = {view: {} for view in ("SAX", "2CH", "4CH")}
    for observation in observations:
        if observation.qc_valid:
            grouped.setdefault(observation.view.upper(), {}).setdefault(observation.slice_id, []).append(observation)
    result: dict[tuple[str, str], list[Any]] = {}
    for view in ("SAX", "2CH", "4CH"):
        locations = []
        for slice_id, frames in grouped.get(view, {}).items():
            centers = torch.stack([frame.center_mm.detach().float().cpu() for frame in frames]); normals = torch.stack([frame.normal.detach().float().cpu() for frame in frames]); normal = normals.mean(0); normal = normal / torch.linalg.vector_norm(normal)
            locations.append((float(torch.dot(centers.mean(0), normal)), slice_id, sorted(frames, key=lambda item: item.timestamp_s)))
        if not locations:
            continue
        _, slice_id, frames = sorted(locations, key=lambda item: (item[0], item[1]))[len(locations) // 2]
        indices = torch.linspace(0, len(frames) - 1, min(frames_per_location, len(frames))).round().long().tolist()
        result[(view, slice_id)] = [frames[index] for index in indices]
    return result


def active_respiratory_levels(stage: str) -> int:
    return stage_contract(stage).active_respiratory_levels


def posthoc_center_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim < 2 or scores.shape[0] == 0:
        raise ValueError("score sequence must have a non-empty time dimension")
    return scores - scores.mean(dim=0, keepdim=True)


def fixed_physical_grid(lower: torch.Tensor, upper: torch.Tensor, shape: tuple[int, int, int] = (16, 16, 16)) -> tuple[torch.Tensor, torch.Tensor]:
    return world_grid(lower, upper, shape)


def mbc_points_from_world_grid(grid: torch.Tensor) -> torch.Tensor:
    """Flatten a structured ``[..., 3]`` world grid for the SINR-MBC API."""
    if not isinstance(grid, torch.Tensor) or grid.ndim == 0 or grid.shape[-1] != 3:
        raise ValueError("world grid must have shape [..., 3]")
    if grid.numel() == 0:
        raise ValueError("world grid must contain at least one point")
    if not torch.isfinite(grid).all():
        raise ValueError("world grid must be finite")
    return grid.reshape(1, -1, 3)


def static_dynamic_rms(values: torch.Tensor) -> dict[str, float]:
    if values.ndim < 2 or not torch.isfinite(values).all():
        raise ValueError("values must be finite with a time dimension")
    mean = values.mean(dim=0); dynamic = values - mean
    static_rms = float(mean.square().mean().sqrt()); dynamic_rms = float(dynamic.square().mean().sqrt())
    return {"static_rms": static_rms, "dynamic_rms": dynamic_rms, "static_dynamic_ratio": static_rms / max(dynamic_rms, torch.finfo(values.dtype).eps)}


def dynamic_field_labels() -> dict[str, str]:
    return {"raw_mbc": "raw MBC basis field", "weighted_dvf": "score-weighted DVF"}


def canonical_internal_bounds_from_checkpoint(checkpoint: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    """Return the exact canonical INR bounds saved in an Exp1 checkpoint.

    The model state is authoritative because it is the tensor used by the
    training-time canonical INR.  Metadata is retained as a consistency check
    for current and legacy checkpoint layouts, never as a substitute for the
    saved state tensor.
    """
    model_state = checkpoint.get("model")
    if not isinstance(model_state, Mapping) or "canonical.inr.bounding_box" not in model_state:
        raise ValueError("Exp1 checkpoint model lacks canonical.inr.bounding_box")
    state_value = model_state["canonical.inr.bounding_box"]
    if not isinstance(state_value, torch.Tensor):
        raise ValueError("canonical.inr.bounding_box must be a tensor")
    state_bounds = state_value.detach().to(device=device, dtype=torch.float32)
    if tuple(state_bounds.shape) != (2, 3):
        raise ValueError(f"canonical.inr.bounding_box must have shape (2, 3), got {tuple(state_bounds.shape)}")
    if not torch.isfinite(state_bounds).all():
        raise ValueError("canonical.inr.bounding_box must be finite")
    if torch.any(state_bounds[1] <= state_bounds[0]):
        raise ValueError("canonical.inr.bounding_box must have upper values greater than lower values")

    initialization = checkpoint.get("static_initialization")
    metadata_bounds: list[tuple[str, Any]] = []
    if initialization is not None:
        if not isinstance(initialization, Mapping):
            raise ValueError("static_initialization must be a mapping when present")
        direct_admission = initialization.get("direct_state_admission")
        if direct_admission is not None:
            if not isinstance(direct_admission, Mapping):
                raise ValueError("static_initialization.direct_state_admission must be a mapping when present")
            if "internal_inr_bbox" in direct_admission:
                metadata_bounds.append(("static_initialization.direct_state_admission.internal_inr_bbox", direct_admission["internal_inr_bbox"]))
            elif "static_bbox" in direct_admission:
                metadata_bounds.append(("static_initialization.direct_state_admission.static_bbox", direct_admission["static_bbox"]))
        if "internal_inr_bbox" in initialization:
            metadata_bounds.append(("static_initialization.internal_inr_bbox", initialization["internal_inr_bbox"]))
        elif "static_bbox" in initialization:
            metadata_bounds.append(("static_initialization.static_bbox", initialization["static_bbox"]))

    for label, value in metadata_bounds:
        try:
            metadata_tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        except Exception as error:
            raise ValueError(f"{label} is not a valid bbox") from error
        if tuple(metadata_tensor.shape) != (2, 3) or not torch.isfinite(metadata_tensor).all() or torch.any(metadata_tensor[1] <= metadata_tensor[0]):
            raise ValueError(f"{label} must be a finite valid bbox with shape (2, 3)")
        if not torch.allclose(state_bounds, metadata_tensor, atol=1e-5, rtol=0.0):
            raise ValueError(f"canonical.inr.bounding_box conflicts with {label}")
    return state_bounds


def weighted_dvf_from_precomputed_basis(raw_basis: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Apply score weighting to a precomputed ``[B,L,N,3]`` MBC basis.

    This is diagnostic-only algebra.  It is intentionally equivalent to
    ``ScoreWeightedMBCField(mbc, scores)(grid)`` while avoiding repeated basis
    evaluation for every temporal frame on a fixed grid.
    """
    if raw_basis.ndim == 3:
        raw_basis = raw_basis.unsqueeze(0)
    if scores.ndim == 2:
        scores = scores.unsqueeze(0)
    if raw_basis.ndim != 4 or scores.ndim != 3:
        raise ValueError(f"basis/scores shape mismatch: {tuple(raw_basis.shape)} vs {tuple(scores.shape)}")
    batch_compatible = raw_basis.shape[0] in (1, scores.shape[0]) or scores.shape[0] in (1, raw_basis.shape[0])
    if raw_basis.shape[1] != scores.shape[1] or raw_basis.shape[-1] != scores.shape[-1] or not batch_compatible:
        raise ValueError(f"basis/scores shape mismatch: {tuple(raw_basis.shape)} vs {tuple(scores.shape)}")
    batch = max(raw_basis.shape[0], scores.shape[0])
    raw_basis = raw_basis.expand(batch, -1, -1, -1)
    scores = scores.expand(batch, -1, -1)
    return (raw_basis * scores.unsqueeze(-2)).sum(dim=1)


def reconstruct_exp1_checkpoint(checkpoint: dict[str, Any], domain: dict[str, Any], observations: list[Any], device: torch.device) -> torch.nn.Module:
    """Reconstruct the exact source-first model encoded by an Exp1 checkpoint."""
    config = checkpoint.get("effective_config")
    if not isinstance(config, dict):
        raise ValueError("Exp1 checkpoint lacks effective_config")
    provenance = checkpoint.get("provenance", {})
    external = provenance.get("effective_external_domain", {}).get("model_float32_effective_domain")
    if external is not None:
        domain = {**domain, "world_min_mm": external["lower_lps_mm"], "world_max_mm": external["upper_lps_mm"]}
    internal_bounds = canonical_internal_bounds_from_checkpoint(checkpoint, device)
    bridge = lps_to_ras_world_matrix(device=device)
    backend = detect_checkpoint_encoding_backend(checkpoint["model"])
    model = build_source_first_model(config, domain, n_dynamic_frames=max(item.dynamic_frame_id for item in observations) + 1, device=device, canonical_encoding_backend=backend, canonical_internal_bounds_world_mm=internal_bounds, canonical_world_to_internal_transform=bridge).to(device)
    model.load_state_dict(checkpoint["model"], strict=True); model.eval(); model.canonical.inr.train()
    return model


def predict_observation_slice(model: torch.nn.Module, observation: Any, stage: str, *, seed: int, chunk_size: int = 1024) -> torch.Tensor:
    height, width = observation.image.shape[-2:]; linear = torch.arange(height * width, device=observation.image.device); pixels = torch.stack((linear.remainder(width), torch.div(linear, width, rounding_mode="floor")), dim=-1).to(observation.image.dtype); outputs = []
    diagnostic_seed = stable_diagnostic_seed(seed, observation.view, observation.slice_id, observation.dynamic_frame_id, purpose="dynamic_audit_psf")
    with diagnostic_rng(diagnostic_seed, observation.image.device), torch.no_grad():
        for start in range(0, len(pixels), chunk_size):
            outputs.append(model.predict(observation, pixels[start:start + chunk_size], stage)["predicted_intensity"].detach())
    return torch.cat(outputs).reshape(height, width)


def dvf_from_scores(model: Any, grid: torch.Tensor, scores: torch.Tensor, active_levels: int) -> torch.Tensor:
    basis = model.respiratory_mbc.raw_active(mbc_points_from_world_grid(grid), active_levels=active_levels)
    return weighted_dvf_from_precomputed_basis(basis, scores[:, :active_levels])
