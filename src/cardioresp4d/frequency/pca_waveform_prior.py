"""Cached, per-fixed-location Phase-1 PCA cardiac waveform evidence."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class PCAWaveformMatch:
    timestamps_s: torch.Tensor
    waveform: torch.Tensor
    selected_pc: int


@dataclass(frozen=True)
class _LocationWaveform:
    timestamps_s: np.ndarray
    waveform: np.ndarray
    selected_pc: int
    source_path: str


class PCAWaveformPrior:
    """One cached selected-PC waveform per reliable ``view/slice_id`` only."""
    def __init__(self, source_path: Path, locations: dict[str, _LocationWaveform], reliable_candidate_count: int) -> None:
        self.source_path = str(source_path)
        self.locations = locations
        self.metadata = {
            "source_frequency_bands": str(source_path),
            "selected_pc_convention": "1-based in frequency_bands.json",
            "reliable_candidate_count": reliable_candidate_count,
            "available_location_count": len(locations),
        }

    @classmethod
    def load(cls, frequency_bands_json: str | Path, *, strict: bool = True) -> "PCAWaveformPrior":
        source = Path(frequency_bands_json).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        candidates = payload.get("cardiac", {}).get("per_slice_candidates", [])
        if not isinstance(candidates, list):
            raise ValueError("cardiac.per_slice_candidates must be a list for PCA waveform supervision")
        locations: dict[str, _LocationWaveform] = {}
        reliable_count = 0
        for candidate in candidates:
            if not isinstance(candidate, dict) or not candidate.get("reliable"):
                continue
            key = candidate.get("slice_key")
            if not isinstance(key, str) or key.count("/") != 1 or key in locations:
                raise ValueError("reliable PCA waveform candidate requires a unique view/slice_id slice_key")
            selected_pc = candidate.get("selected_pc")
            if isinstance(selected_pc, bool) or not isinstance(selected_pc, int) or selected_pc < 1:
                if strict:
                    raise ValueError(f"reliable PCA waveform candidate {key} has invalid 1-based selected_pc")
                continue
            reliable_count += 1
            view, slice_id = key.split("/", 1)
            artifact = source.parent / view / slice_id / "pca_psd.npz"
            if not artifact.is_file():
                if strict:
                    raise ValueError(f"reliable PCA waveform candidate {key} lacks {artifact}")
                continue
            with np.load(artifact, allow_pickle=False) as archive:
                if "timestamps_s" not in archive or "temporal_pcs" not in archive:
                    raise ValueError(f"PCA artifact {artifact} lacks timestamps_s/temporal_pcs")
                timestamps = np.asarray(archive["timestamps_s"], dtype=np.float64)
                temporal_pcs = np.asarray(archive["temporal_pcs"], dtype=np.float64)
            if timestamps.ndim != 1 or temporal_pcs.ndim != 2 or temporal_pcs.shape[0] != timestamps.size or timestamps.size < 1:
                raise ValueError(f"PCA artifact {artifact} has incompatible timestamp/temporal_pcs shapes")
            if not np.isfinite(timestamps).all() or not np.isfinite(temporal_pcs).all() or np.any(np.diff(timestamps) <= 0):
                raise ValueError(f"PCA artifact {artifact} has non-finite or non-increasing data")
            index = selected_pc - 1
            if index >= temporal_pcs.shape[1]:
                raise ValueError(f"PCA artifact {artifact} cannot select PC {selected_pc}")
            locations[key] = _LocationWaveform(timestamps, temporal_pcs[:, index].copy(), selected_pc, str(artifact))
        return cls(source, locations, reliable_count)

    def match(self, view: str, slice_id: str, timestamps_s: torch.Tensor, *, tolerance_s: float = 1e-6, device: torch.device | None = None, dtype: torch.dtype | None = None) -> PCAWaveformMatch | None:
        """Strictly pair current valid timestamps with the same location's PC.

        Phase-1 and training both derive timestamp_s from the manifest.  A
        one-microsecond tolerance absorbs CSV float parsing only; no waveform
        interpolation or cross-location frame pairing is permitted.
        """
        if not tolerance_s > 0:
            raise ValueError("PCA waveform timestamp tolerance must be positive")
        location = self.locations.get(f"{view}/{slice_id}")
        if location is None:
            return None
        requested = torch.as_tensor(timestamps_s).reshape(-1)
        if not requested.numel() or not torch.isfinite(requested).all():
            raise ValueError("PCA waveform matching requires finite current timestamps")
        indices: list[int] = []
        for value in requested.detach().cpu().to(torch.float64).tolist():
            insertion = int(np.searchsorted(location.timestamps_s, value))
            candidates = [index for index in (insertion - 1, insertion) if 0 <= index < location.timestamps_s.size]
            if not candidates:
                raise ValueError(f"PCA waveform {view}/{slice_id} has no timestamp matching {value:.12g}")
            index = min(candidates, key=lambda item: abs(location.timestamps_s[item] - value))
            if abs(location.timestamps_s[index] - value) > tolerance_s:
                raise ValueError(f"PCA waveform {view}/{slice_id} timestamp mismatch at {value:.12g}; no interpolation is allowed")
            indices.append(index)
        target_device = requested.device if device is None else device
        target_dtype = requested.dtype if dtype is None else dtype
        return PCAWaveformMatch(
            requested.to(device=target_device, dtype=target_dtype),
            torch.as_tensor(location.waveform[indices], device=target_device, dtype=target_dtype),
            location.selected_pc,
        )
