"""Pure aggregation helpers shared by checkpoint diagnostic CLIs."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import torch


def location_key(record: Mapping[str, object]) -> str:
    return f"{record['view']}/{record['slice_id']}"


def _continuous(values: Iterable[object]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return {"n": 0, "mean": None, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    value = torch.tensor(finite, dtype=torch.float64)
    return {"n": len(finite), "mean": float(value.mean()), "median": float(value.median()), "q25": float(torch.quantile(value, .25)), "q75": float(torch.quantile(value, .75)), "min": float(value.min()), "max": float(value.max())}


def _boolean(values: Iterable[object]) -> dict[str, float | int]:
    observed = [bool(value) for value in values if value is not None]
    count = sum(observed)
    return {"n": len(observed), "count": count, "fraction": 0. if not observed else count / len(observed)}


def _group(records: list[Mapping[str, object]], continuous: tuple[str, ...], boolean: tuple[str, ...]) -> dict[str, object]:
    return {"n_locations": len(records), "n_eligible": sum(bool(record.get("eligible")) for record in records), "skip_reasons": {str(reason): sum(record.get("skip_reason") == reason for record in records) for reason in sorted({record.get("skip_reason") for record in records if record.get("skip_reason")})}, "continuous": {name: _continuous(record.get(name) for record in records) for name in continuous}, "boolean": {name: _boolean(record.get(name) for record in records) for name in boolean}}


def summarize_records(records: list[Mapping[str, object]], *, continuous: tuple[str, ...], boolean: tuple[str, ...]) -> dict[str, object]:
    by_view: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    by_pc: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        by_view[str(record["view"])].append(record)
        selected_pc = record.get("selected_cardiac_pc")
        if selected_pc is not None:
            by_pc[str(selected_pc)].append(record)
    return {"overall": _group(records, continuous, boolean), "by_view": {name: _group(group, continuous, boolean) for name, group in sorted(by_view.items())}, "by_selected_pc": {name: _group(group, continuous, boolean) for name, group in sorted(by_pc.items())}}
