#!/usr/bin/env python3
"""Exp3-B0: offline factor/stage audit for existing Exp3 recovery outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _metric(summary: dict[str, Any], name: str) -> float | None:
    values = summary.get("final_metrics", {}).get("aggregate", {}).get(name, {})
    return values.get("mean") if isinstance(values, dict) else None


def audit(summary_path: Path, curve_path: Path | None, output_dir: Path) -> dict[str, Any]:
    """Audit only supplied outputs; this never fabricates formal experiment data."""
    summary = json.loads(summary_path.read_text(encoding="utf-8")); rows: list[dict[str, str]] = []
    if curve_path is not None:
        with curve_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    def best(column: str, reverse: bool) -> dict[str, Any] | None:
        valid = [row for row in rows if row.get(column) not in (None, "")]
        return (max if reverse else min)(valid, key=lambda row: float(row[column])) if valid else None
    dvf = {name: _metric(summary, name) for name in ("full_fov_dynamic_cosine", "full_fov_dynamic_rms_mm", "full_fov_gt_dynamic_rms_mm", "full_fov_dynamic_rms_ratio", "full_fov_dynamic_normalized_epe", "full_fov_static_rms_mm", "pixel_center_dynamic_cosine", "pixel_center_dynamic_rms_ratio", "pixel_center_dynamic_normalized_epe")}
    best_reconstruction, best_cosine = best("full_batch_reconstruction_mse", False), best("full_fov_dynamic_cosine", True)
    result = {"experiment": "Exp3-B0 factor diagnostic audit", "input_summary": str(summary_path), "input_training_curve": None if curve_path is None else str(curve_path), "dvf_primary_metrics": dvf, "score_contract": "Raw score amplitude/sign are gauge-dependent diagnostics; DVF metrics determine recovery.", "factor_diagnostics": {key: summary.get(key) for key in ("encoder_parameter_norm", "respiratory_mbc_level0_parameter_norm", "learned_respiratory_mbc_calls") if key in summary}, "basis_field_metric_supported": False, "basis_field_metric_reason": "Offline summary/curve mode has no unambiguous physical unit-basis field evaluation.", "curve_audit": {"best_reconstruction_step": None if best_reconstruction is None else int(best_reconstruction.get("global_step", best_reconstruction.get("step", 0))), "best_full_fov_cosine_step": None if best_cosine is None else int(best_cosine.get("global_step", best_cosine.get("step", 0))), "joint_stage_regression": bool(best_cosine is not None and rows and best_cosine is not rows[-1])}}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "factor_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "factor_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"]); writer.writeheader(); writer.writerows({"metric": key, "value": value} for key, value in dvf.items())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3-B0 offline factor diagnostic audit")
    parser.add_argument("--summary", type=Path, required=True); parser.add_argument("--training-curve", type=Path); parser.add_argument("--output-dir", type=Path, required=True)
    audit(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
