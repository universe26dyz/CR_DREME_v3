"""View/location-balanced sampler for valid acquired dynamic observations."""
from __future__ import annotations

import random
from dataclasses import dataclass
from collections import defaultdict

import torch

_VIEWS = ("SAX", "2CH", "4CH")
_HARD_INVALID = {"slice_local_scale_absolute", "manual_exclusion"}


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


class ViewLocationBalancedSampler:
    """Choose one valid frame per view after uniformly choosing a location."""
    def __init__(self, observations: list[DynamicObservation], *, seed: int = 0) -> None:
        grouped: dict[str, dict[str, list[DynamicObservation]]] = defaultdict(lambda: defaultdict(list))
        for observation in observations:
            if not observation.qc_valid or observation.qc_reason in _HARD_INVALID:
                continue
            grouped[observation.view.upper()][observation.slice_id].append(observation)
        missing = [view for view in _VIEWS if not grouped[view]]
        if missing: raise ValueError("view-balanced sampler lacks valid observations for " + ", ".join(missing))
        self._grouped = grouped; self._rng = random.Random(seed)

    def sample_step(self) -> list[DynamicObservation]:
        selected: list[DynamicObservation] = []
        for view in _VIEWS:
            locations = sorted(self._grouped[view])
            location = self._rng.choice(locations)
            selected.append(self._rng.choice(self._grouped[view][location]))
        return selected

    def temporal_batch(self, *, max_items: int) -> list[DynamicObservation]:
        """One differentiable same-location sequence sorted by true acquisition time."""
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        candidates = [(view, location) for view, locations in self._grouped.items() for location, frames in locations.items() if len(frames) >= 3]
        if not candidates:
            return []
        view, location = self._rng.choice(sorted(candidates))
        return sorted(self._grouped[view][location], key=lambda item: item.timestamp_s)[:max_items]
