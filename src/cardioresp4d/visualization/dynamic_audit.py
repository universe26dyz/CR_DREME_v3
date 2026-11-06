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


def static_dynamic_rms(values: torch.Tensor) -> dict[str, float]:
    if values.ndim < 2 or not torch.isfinite(values).all():
        raise ValueError("values must be finite with a time dimension")
    mean = values.mean(dim=0); dynamic = values - mean
    static_rms = float(mean.square().mean().sqrt()); dynamic_rms = float(dynamic.square().mean().sqrt())
    return {"static_rms": static_rms, "dynamic_rms": dynamic_rms, "static_dynamic_ratio": static_rms / max(dynamic_rms, torch.finfo(values.dtype).eps)}


def dynamic_field_labels() -> dict[str, str]:
    return {"raw_mbc": "raw MBC basis field", "weighted_dvf": "score-weighted DVF"}


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
    initialization = checkpoint.get("static_initialization", {})
    bounds = initialization.get("internal_inr_bbox", initialization.get("static_bbox"))
    internal_bounds = None if bounds is None else torch.as_tensor(bounds, dtype=torch.float32, device=device)
    bridge = lps_to_ras_world_matrix(device=device)
    backend = detect_checkpoint_encoding_backend(checkpoint["model"])
    model = build_source_first_model(config, domain, n_dynamic_frames=max(item.dynamic_frame_id for item in observations) + 1, device=device, canonical_encoding_backend=backend, canonical_internal_bounds_world_mm=internal_bounds, canonical_world_to_internal_transform=bridge if internal_bounds is not None else None).to(device)
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
    basis = model.respiratory_mbc.raw_active(grid, active_levels=active_levels)
    return weighted_dvf_from_precomputed_basis(basis, scores[:, :active_levels])
