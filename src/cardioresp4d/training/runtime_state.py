"""Runtime identity, hard-QC and reproducibility state shared by train/preflight."""
from __future__ import annotations

import math
import random
from typing import Any, Mapping

import numpy as np
import torch

HARD_INVALID_REASONS = frozenset({"slice_local_scale_absolute", "manual_exclusion"})


def qc_reason_tokens(reason: object) -> frozenset[str]:
    return frozenset(token.strip() for token in str(reason or "").split(";") if token.strip() and token.strip() != "valid")


def is_hard_invalid_reason(reason: object) -> bool:
    return bool(qc_reason_tokens(reason) & HARD_INVALID_REASONS)


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no"}


def validate_dynamic_frame_rows(rows: list[Mapping[str, Any]]) -> None:
    """Validate manifest/QC handoff before IDs, embeddings or losses exist."""
    tokens: set[str] = set()
    keys: set[tuple[str, str, str]] = set()
    by_location: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in rows:
        token = str(row.get("source_file_token", ""))
        key = (str(row.get("view", "")), str(row.get("slice_id", "")), str(row.get("frame_index", "")))
        if not token:
            raise ValueError("dynamic-frame manifest row lacks source_file_token")
        if token in tokens:
            raise ValueError(f"duplicate dynamic-frame source_file_token: {token}")
        if key in keys:
            raise ValueError(f"duplicate dynamic-frame manifest key: {key}")
        tokens.add(token); keys.add(key)
        try:
            timestamp = float(row.get("timestamp_s"))
        except (TypeError, ValueError) as exc:
            raise ValueError("dynamic-frame timestamp must be finite seconds") from exc
        if not math.isfinite(timestamp):
            raise ValueError("dynamic-frame timestamp must be finite seconds")
        try:
            frame_index = int(row.get("frame_index"))
        except (TypeError, ValueError) as exc:
            raise ValueError("dynamic-frame frame_index must be an integer") from exc
        by_location.setdefault((key[0], key[1]), []).append((frame_index, timestamp))
        valid = _as_bool(row.get("qc_valid", True))
        hard = is_hard_invalid_reason(row.get("qc_reason", "valid"))
        if not valid and not hard:
            raise ValueError("qc_valid=false requires a hard-invalid reason token")
        if valid and hard:
            raise ValueError("hard-invalid reason token requires qc_valid=false")
    for location, values in by_location.items():
        ordered = [timestamp for _, timestamp in sorted(values)]
        if any(later < earlier for earlier, later in zip(ordered, ordered[1:])):
            raise ValueError(f"dynamic-frame timestamps are non-monotonic for {location}; midnight unwrap is not implicit")


def set_reproducibility(seed: int) -> dict[str, Any]:
    """Seed every runtime RNG that can affect source-first CPU/GPU execution."""
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {"seed": seed, "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled())}


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
