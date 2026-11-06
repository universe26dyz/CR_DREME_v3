"""Generic contracts and safety checks for read-only experiment audits."""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping


STAGES = tuple(f"V{index}_{name}" for index, name in enumerate((
    "raw_acquisition", "normalization", "temporal_mean", "stack_geometry",
    "registration", "static_support", "static_inr", "static_reprojection",
    "dynamic_zero_motion", "dynamic_boundaries",
)))


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_safe_output_dir(output_dir: str | Path, authoritative_artifacts: list[str | Path]) -> Path:
    """Create an output directory only when it cannot contain an input artifact."""
    output = Path(output_dir).resolve()
    for artifact in authoritative_artifacts:
        path = Path(artifact).resolve()
        try:
            path.relative_to(output)
        except ValueError:
            continue
        raise ValueError(f"output directory would overwrite authoritative artifact: {path}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def stage_status(render_status: str = "partial", reason: str | None = None, **details: Any) -> dict[str, Any]:
    if render_status not in {"rendered", "partial", "failed"}:
        raise ValueError("render_status must be rendered, partial, or failed")
    payload: dict[str, Any] = {"render_status": render_status, "manual_admission": "pending"}
    if reason is not None:
        payload["reason"] = reason
    payload.update(details)
    return payload


class ExperimentVisualizationAdapter(ABC):
    """Experiment-specific source layout plugged into the generic audit core."""

    name: str

    @property
    @abstractmethod
    def scientific_status(self) -> Mapping[str, str]: ...

    @abstractmethod
    def source_artifacts(self, experiment_root: Path, manifest: Path, qc_table: Path, canonical_domain: Path) -> dict[str, Path]: ...

    @abstractmethod
    def render(self, *, experiment_root: Path, manifest: Path, qc_table: Path, canonical_domain: Path, output_dir: Path, device: str) -> dict[str, dict[str, Any]]: ...

    # Stable extension points: future adapters can implement a dynamic renderer
    # without coupling generic report/safety code to a checkpoint schema.
    def render_v8_dynamic_zero_motion(self, **_: Any) -> dict[str, Any]:
        return stage_status("partial", "dynamic zero-motion hook not implemented by this adapter")

    def available_dynamic_checkpoints(self, **_: Any) -> list[Path]:
        return []

    def render_v9_dynamic_boundaries(self, **_: Any) -> dict[str, Any]:
        return stage_status("partial", "dynamic boundary hook not implemented by this adapter")
