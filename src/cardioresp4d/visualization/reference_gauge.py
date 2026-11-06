"""Authoritative temporal-mean reference images for score-gauge diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import torch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReferenceImageProvider:
    """Immutable ``(view, slice_id)`` -> normalized mean-image mapping."""

    def __init__(self, references: dict[tuple[str, str], torch.Tensor], provenance: dict[str, Any]) -> None:
        self._references = {key: value.detach().clone() for key, value in references.items()}
        self.provenance = dict(provenance)

    @classmethod
    def from_mean_artifacts(cls, mean_manifest: str | Path, observations: Iterable[Any] | None = None) -> "ReferenceImageProvider":
        manifest_path = Path(mean_manifest).resolve(); payload = json.loads(manifest_path.read_text())
        records = payload.get("locations", [])
        if not isinstance(records, list):
            raise ValueError("mean manifest locations must be a list")
        references: dict[tuple[str, str], torch.Tensor] = {}; stack_hashes: dict[str, str] = {}; stack_cache: dict[Path, np.ndarray] = {}
        for record in records:
            if record.get("real", True) is False or str(record.get("slice_id", "")).upper().startswith("MISSING"):
                continue
            key = (str(record["view"]).upper(), str(record["slice_id"]))
            if key in references:
                raise ValueError(f"duplicate authoritative reference location {key[0]}/{key[1]}")
            stack_path = (manifest_path.parent / str(record["stack_file"])).resolve()
            if not stack_path.is_file():
                raise FileNotFoundError(f"missing authoritative mean stack: {stack_path}")
            stack_hashes[stack_path.name] = _sha256(stack_path)
            if stack_path not in stack_cache:
                stack_cache[stack_path] = np.asanyarray(nib.load(str(stack_path)).dataobj, dtype=np.float32)
            volume = stack_cache[stack_path]; index = int(record["stack_index"])
            if volume.ndim != 3 or index < 0 or index >= volume.shape[-1]:
                raise ValueError(f"invalid mean-stack index for {key[0]}/{key[1]}")
            image = np.asarray(volume[..., index].T, dtype=np.float32)
            if not np.isfinite(image).all():
                raise ValueError(f"non-finite authoritative reference image for {key[0]}/{key[1]}")
            expected_shape = tuple(int(value) for value in record.get("image_shape", (1, *image.shape))[-2:])
            if image.shape != expected_shape:
                raise ValueError(f"authoritative reference shape mismatch for {key[0]}/{key[1]}: {image.shape} != {expected_shape}")
            references[key] = torch.from_numpy(image.copy())[None]
        provider = cls(references, {"policy": "mean_stack_volume_index_transpose", "mean_manifest": str(manifest_path), "mean_manifest_sha256": _sha256(manifest_path), "mean_stacks": stack_hashes, "location_count": len(references)})
        if observations is not None:
            for observation in observations:
                if not observation.qc_valid:
                    continue
                provider.validate_observation(observation)
        return provider

    def validate_observation(self, observation: Any) -> None:
        key = (str(observation.view).upper(), str(observation.slice_id))
        if key not in self._references:
            raise KeyError(f"missing authoritative reference for {key[0]}/{key[1]}")
        reference = self._references[key]
        if tuple(reference.shape) != tuple(observation.image.shape):
            raise ValueError(f"reference image shape mismatch for {key[0]}/{key[1]}: {tuple(reference.shape)} != {tuple(observation.image.shape)}")

    def __call__(self, observation: Any) -> torch.Tensor:
        self.validate_observation(observation)
        key = (str(observation.view).upper(), str(observation.slice_id))
        return self._references[key].to(device=observation.image.device, dtype=observation.image.dtype).clone()

    @property
    def keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._references))
