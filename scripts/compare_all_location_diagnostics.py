#!/usr/bin/env python3
"""Read-only exact-key comparison of full-location diagnostic JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


METRICS = ("cardiac_pca_waveform_r2", "cardiac_target_fraction", "card_target_over_wrong", "card_score_std")
BOOLEANS = ("same_peak_exact", "same_peak_within_0.02_hz", "target_gt_wrong")


def _key(record: dict) -> str:
    return f"{record['view']}/{record['slice_id']}"


def _stats(values: list[float], tolerance: float) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean_delta": None, "median_delta": None, "q25": None, "q75": None, "n_improved": 0, "n_worsened": 0, "n_unchanged": 0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {"n": len(values), "mean_delta": float(tensor.mean()), "median_delta": float(tensor.median()), "q25": float(torch.quantile(tensor, .25)), "q75": float(torch.quantile(tensor, .75)), "n_improved": int((tensor > tolerance).sum()), "n_worsened": int((tensor < -tolerance).sum()), "n_unchanged": int((tensor.abs() <= tolerance).sum())}


def _transition(left: list[dict], right: list[dict], field: str) -> dict[str, int]:
    counts = {"false_to_true": 0, "true_to_false": 0, "true_to_true": 0, "false_to_false": 0, "unavailable": 0}
    for first, second in zip(left, right):
        a, b = first.get(field), second.get(field)
        if a is None or b is None:
            counts["unavailable"] += 1
        else:
            counts[f"{str(bool(a)).lower()}_to_{str(bool(b)).lower()}"] += 1
    return counts


def compare_records(left_records: list[dict], right_records: list[dict], *, tolerance: float = 1e-6) -> dict:
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    left, right = {_key(record): record for record in left_records}, {_key(record): record for record in right_records}
    keys = sorted(left.keys() & right.keys())
    paired_left, paired_right = [left[key] for key in keys], [right[key] for key in keys]
    metrics = {name: _stats([float(second[name]) - float(first[name]) for first, second in zip(paired_left, paired_right) if first.get(name) is not None and second.get(name) is not None], tolerance) for name in METRICS}
    for first, second in zip(paired_left, paired_right):
        first["target_gt_wrong"] = None if first.get("card_target_amp") is None or first.get("card_wrong_amp") is None else first["card_target_amp"] > first["card_wrong_amp"]
        second["target_gt_wrong"] = None if second.get("card_target_amp") is None or second.get("card_wrong_amp") is None else second["card_target_amp"] > second["card_wrong_amp"]
    return {"paired_location_keys": keys, "missing_from_right": sorted(left.keys() - right.keys()), "missing_from_left": sorted(right.keys() - left.keys()), "tolerance": tolerance, "metrics": metrics, "transitions": {name: _transition(paired_left, paired_right, name) for name in BOOLEANS}}


def _load(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["records"] if isinstance(payload, dict) and "records" in payload else payload["all_locations"]["records"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=JSON", help="two or more labeled all-location diagnostic JSON files")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if len(args.input) < 2: parser.error("at least two --input LABEL=JSON values are required")
    labeled = {}
    for value in args.input:
        if "=" not in value: parser.error("--input must be LABEL=JSON")
        label, path = value.split("=", 1); labeled[label] = _load(Path(path))
    labels = list(labeled)
    baseline = labels[0]
    result = {"baseline": baseline, "comparisons": {label: compare_records(labeled[baseline], labeled[label], tolerance=args.tolerance) for label in labels[1:]}, "note": "Paired descriptive deltas only; no significance claim is made from one subject."}
    args.output_json.parent.mkdir(parents=True, exist_ok=True); args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": baseline, "comparisons": {label: len(value["paired_location_keys"]) for label, value in result["comparisons"].items()}, "output": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
