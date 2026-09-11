"""Single formal owner of progressive-stage runtime semantics."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageContract:
    """Which source paths are permitted to execute in one training stage."""

    name: str
    active_respiratory_levels: int
    enable_film: bool
    enable_cardiac: bool
    enable_uncertainty: bool

    @property
    def enable_motion(self) -> bool:
        return self.active_respiratory_levels > 0 or self.enable_cardiac

    @property
    def trainable_modules(self) -> frozenset[str]:
        modules = {"canonical"}
        if self.enable_film:
            modules.update(("film", "respiratory_mbc"))
        if self.enable_cardiac:
            modules.add("cardiac_mbc")
        if self.enable_uncertainty:
            modules.add("uncertainty")
        return frozenset(modules)


_STAGES = {
    "stage1": StageContract("stage1", 0, False, False, False),
    "stage2a": StageContract("stage2a", 1, True, False, False),
    "stage2b": StageContract("stage2b", 2, True, False, False),
    "stage2c": StageContract("stage2c", 3, True, False, False),
    "stage3": StageContract("stage3", 3, True, True, True),
}


def stage_contract(stage: str) -> StageContract:
    """Return the immutable formal contract for a supported progressive stage."""
    try:
        return _STAGES[stage]
    except KeyError as exc:
        raise ValueError(f"unsupported progressive stage: {stage}") from exc
