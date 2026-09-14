"""Single formal owner of v3_change4 progressive-stage runtime semantics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FilmTrainMode = Literal["none", "all", "card_head_only"]
DataTerm = Literal["mse", "gaussian_nll"]


@dataclass(frozen=True)
class StageContract:
    """Execution, loss and optimizer ownership for exactly one formal stage.

    Stage3 is deliberately represented as 3a/3b/3c: this makes cardiac
    identifiability and uncertainty ownership explicit instead of inferring
    them from a historical stage name.
    """

    name: str
    active_respiratory_levels: int
    enable_film: bool
    enable_cardiac: bool
    enable_uncertainty: bool
    data_term: DataTerm
    film_train_mode: FilmTrainMode
    train_canonical: bool
    train_respiratory_mbc: bool
    train_cardiac_mbc: bool

    @property
    def enable_motion(self) -> bool:
        return self.active_respiratory_levels > 0 or self.enable_cardiac

    @property
    def trainable_modules(self) -> frozenset[str]:
        modules: set[str] = set()
        if self.train_canonical:
            modules.add("canonical")
        if self.film_train_mode != "none":
            modules.add("film")
        if self.train_respiratory_mbc:
            modules.add("respiratory_mbc")
        if self.train_cardiac_mbc:
            modules.add("cardiac_mbc")
        if self.enable_uncertainty:
            modules.add("uncertainty")
        return frozenset(modules)


_STAGES = {
    "stage1": StageContract("stage1", 0, False, False, False, "mse", "none", True, False, False),
    "stage2a": StageContract("stage2a", 1, True, False, False, "mse", "all", True, True, False),
    "stage2b": StageContract("stage2b", 2, True, False, False, "mse", "all", True, True, False),
    "stage2c": StageContract("stage2c", 3, True, False, False, "mse", "all", True, True, False),
    # Cardiac warm-up preserves the settled respiratory solution: only the
    # local cardiac basis and its output head may absorb cardiac residual.
    "stage3a": StageContract("stage3a", 3, True, True, False, "mse", "card_head_only", False, False, True),
    "stage3b": StageContract("stage3b", 3, True, True, False, "mse", "all", True, True, True),
    "stage3c": StageContract("stage3c", 3, True, True, True, "gaussian_nll", "all", True, True, True),
}


def stage_contract(stage: str) -> StageContract:
    """Return the immutable formal contract for a supported progressive stage."""
    try:
        return _STAGES[stage]
    except KeyError as exc:
        raise ValueError(f"unsupported progressive stage: {stage}") from exc


def progressive_stage_order() -> tuple[str, ...]:
    """Stable checkpoint order; Stage1–Stage2c is retained for v3 resume."""
    return tuple(_STAGES)
