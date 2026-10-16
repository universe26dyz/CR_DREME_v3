"""View/location-balanced sampler for valid acquired dynamic observations."""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from collections import defaultdict

import torch
from .runtime_state import HARD_INVALID_REASONS, is_hard_invalid_reason

_VIEWS = ("SAX", "2CH", "4CH")
_HARD_INVALID = HARD_INVALID_REASONS
_VIEW_IDS = {"SAX": 0, "2CH": 1, "4CH": 2}


@dataclass(frozen=True)
class DynamicObservation:
    image: torch.Tensor  # [1,H,W], normalized acquired image
    view: str
    slice_id: str
    dynamic_frame_id: int
    center_mm: torch.Tensor
    row_direction: torch.Tensor
    column_direction: torch.Tensor
    normal: torch.Tensor
    pixel_spacing_mm: torch.Tensor  # [row, column]
    slice_thickness_mm: float
    qc_valid: bool
    qc_reason: str
    timestamp_s: float = 0.0
    normalized_slice_position: float | None = None


def view_id_for(view: str) -> int:
    """Fixed S2V view-family mapping, independent of acquisition order."""
    try:
        return _VIEW_IDS[view.upper()]
    except KeyError as error:
        raise ValueError(f"unsupported S2V view {view!r}; expected SAX, 2CH, or 4CH") from error


def assign_normalized_slice_positions(observations: list[DynamicObservation], *, normal_tolerance: float = 1e-4, center_tolerance_mm: float = 1e-4) -> list[DynamicObservation]:
    """Assign physical stack-relative positions after all observations are built.

    Positions are computed once per QC-valid location within each view, so every
    temporal frame of a location receives the identical scalar.
    """
    if normal_tolerance <= 0 or center_tolerance_mm <= 0:
        raise ValueError("slice-position tolerances must be positive")
    by_view: dict[str, dict[str, list[DynamicObservation]]] = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        if observation.qc_valid:
            view_id_for(observation.view)
            by_view[observation.view.upper()][observation.slice_id].append(observation)
    positions: dict[tuple[str, str], float] = {}
    for view, locations in by_view.items():
        representatives: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        reference_normal: torch.Tensor | None = None
        for slice_id, frames in locations.items():
            centers = torch.stack([frame.center_mm.detach().to(dtype=torch.float64, device="cpu") for frame in frames])
            if not torch.allclose(centers, centers[0].expand_as(centers), atol=center_tolerance_mm, rtol=0.0):
                raise ValueError(f"same slice ID must have one physical center: {view}/{slice_id}")
            normals = torch.stack([frame.normal.detach().to(dtype=torch.float64, device="cpu") for frame in frames])
            norms = torch.linalg.vector_norm(normals, dim=-1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=normal_tolerance, rtol=0.0):
                raise ValueError(f"within-view plane normals must be unit vectors: {view}/{slice_id}")
            if reference_normal is None:
                reference_normal = normals[0]
            signs = torch.where((normals * reference_normal).sum(dim=-1, keepdim=True) >= 0, torch.ones_like(normals[:, :1]), -torch.ones_like(normals[:, :1]))
            aligned = normals * signs
            if torch.any(torch.abs((aligned * reference_normal).sum(dim=-1)) < 1.0 - normal_tolerance):
                raise ValueError(f"within-view plane normals are not sufficiently consistent: {view}")
            representatives[slice_id] = (centers[0], aligned.mean(dim=0))
        assert reference_normal is not None
        aligned_normals = []
        for _, normal in representatives.values():
            if torch.dot(normal, reference_normal) < 0:
                normal = -normal
            aligned_normals.append(normal)
        stack_normal = torch.stack(aligned_normals).mean(dim=0)
        stack_normal = stack_normal / torch.linalg.vector_norm(stack_normal)
        projections = {slice_id: float(torch.dot(center, stack_normal)) for slice_id, (center, _) in representatives.items()}
        q_min, q_max = min(projections.values()), max(projections.values())
        half = (q_max - q_min) / 2.0
        for slice_id, projection in projections.items():
            positions[(view, slice_id)] = 0.0 if half <= center_tolerance_mm else (projection - (q_min + q_max) / 2.0) / half
    return [replace(observation, normalized_slice_position=positions[(observation.view.upper(), observation.slice_id)]) if observation.qc_valid else observation for observation in observations]


class ViewLocationBalancedSampler:
    """Choose one valid frame per view after uniformly choosing a location."""
    def __init__(self, observations: list[DynamicObservation], *, seed: int = 0) -> None:
        grouped: dict[str, dict[str, list[DynamicObservation]]] = defaultdict(lambda: defaultdict(list))
        for observation in observations:
            hard = is_hard_invalid_reason(observation.qc_reason)
            if not observation.qc_valid and not hard:
                raise ValueError("qc_valid=false observation has no hard-invalid token")
            if observation.qc_valid and hard:
                raise ValueError("hard-invalid observation must have qc_valid=false")
            if not observation.qc_valid:
                continue
            grouped[observation.view.upper()][observation.slice_id].append(observation)
        missing = [view for view in _VIEWS if not grouped[view]]
        if missing: raise ValueError("view-balanced sampler lacks valid observations for " + ", ".join(missing))
        self._grouped = grouped; self._rng = random.Random(seed); self._update_index = 0

    def sample_step(self, *, observations_per_update: int = 3) -> list[DynamicObservation]:
        if observations_per_update < len(_VIEWS):
            raise ValueError("observations_per_update must include every required view")
        base, remainder = divmod(observations_per_update, len(_VIEWS))
        extras = {_VIEWS[(self._update_index + index) % len(_VIEWS)] for index in range(remainder)}
        selected: list[DynamicObservation] = []
        for view in _VIEWS:
            locations = sorted(self._grouped[view])
            for _ in range(base + int(view in extras)):
                location = self._rng.choice(locations)
                selected.append(self._rng.choice(self._grouped[view][location]))
        self._update_index += 1
        return selected

    def temporal_batch(self, *, max_items: int) -> list[DynamicObservation]:
        """One differentiable same-location sequence sorted by true acquisition time."""
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        candidates = [(view, location) for view, locations in self._grouped.items() for location, frames in locations.items() if len(frames) >= 3]
        if not candidates:
            return []
        view, location = self._rng.choice(sorted(candidates))
        ordered=sorted(self._grouped[view][location], key=lambda item: item.timestamp_s)
        if len(ordered)<=max_items: return ordered
        # Deterministic full-span stratification; never use only early frames.
        indices=torch.linspace(0,len(ordered)-1,max_items).round().long().tolist()
        return [ordered[index] for index in indices]

    def state_dict(self) -> dict:
        return {"python_random_state": self._rng.getstate(), "update_index": self._update_index}

    def load_state_dict(self, state: dict) -> None:
        self._rng.setstate(state["python_random_state"])
        self._update_index = int(state.get("update_index", 0))
