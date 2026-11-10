#!/usr/bin/env python3
"""Build a machine-readable Exp2-A/Exp2-B fixed score-gain comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


def _mean_summary(summary: Mapping[str, Any] | None, key: str) -> float | None:
    value = (summary or {}).get(key, {})
    return _number(value.get("mean")) if isinstance(value, Mapping) else None


def extract_experiment_metrics(report: Mapping[str, Any], *, default_gain: float | None = None) -> dict[str, Any]:
    diagnostics = report.get("diagnostics", {}).get("step_200", {})
    runner = report.get("runner_report", {})
    gain = report.get("score_gain", report.get("protocol", {}).get("score_gain", default_gain))
    if gain is None:
        raise ValueError("training report lacks score_gain")
    unscaled_score = diagnostics.get("unscaled_effective_score_dynamic_rms", diagnostics.get("respiratory_score_dynamic_rms"))
    used_score = diagnostics.get("used_score_dynamic_rms", diagnostics.get("respiratory_score_dynamic_rms"))
    unscaled_dvf = diagnostics.get("unscaled_effective_dvf_dynamic_rms_mm", diagnostics.get("respiratory_dvf_dynamic_rms_mm"))
    used_dvf = diagnostics.get("used_dvf_dynamic_rms_mm", diagnostics.get("respiratory_dvf_dynamic_rms_mm"))
    gradient = runner.get("gradient_summary", {})
    losses = runner.get("loss_components", {})
    result = {
        "gain": float(gain),
        "reconstruction_mse_full": _number(diagnostics.get("reconstruction_mse_full")),
        "reconstruction_mse_SAX": _number(diagnostics.get("reconstruction_mse_SAX")),
        "reconstruction_mse_2CH": _number(diagnostics.get("reconstruction_mse_2CH")),
        "reconstruction_mse_4CH": _number(diagnostics.get("reconstruction_mse_4CH")),
        "unscaled_effective_score_dynamic_rms": _number(unscaled_score),
        "used_score_dynamic_rms": _number(used_score),
        "unscaled_effective_dvf_dynamic_rms_mm": _number(unscaled_dvf),
        "used_dvf_dynamic_rms_mm": _number(used_dvf),
        "encoder_gradient_norm_mean": _mean_summary(gradient, "encoder_norm"),
        "encoder_gradient_rms_mean": _mean_summary(gradient, "encoder_rms"),
        "l1_mbc_gradient_norm_mean": _mean_summary(gradient, "l1_mbc_norm"),
        "loss_data": _mean_summary(losses, "data"),
        "loss_smooth_resp": _mean_summary(losses, "smooth_resp"),
        "loss_mbc_normalization": _mean_summary(losses, "mbc_normalization"),
        "loss_total": _mean_summary(losses, "total"),
    }
    if result["loss_total"] is None:
        result["loss_total"] = _number(runner.get("loss_mean", runner.get("loss_last")))
    return result


def _fold(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline == 0:
        return None
    return value / baseline


def build_score_gain_comparison(exp2a: Mapping[str, Any], gain10: Mapping[str, Any], gain100: Mapping[str, Any]) -> dict[str, Any]:
    metrics = {"gain1": extract_experiment_metrics(exp2a, default_gain=1.0), "gain10": extract_experiment_metrics(gain10), "gain100": extract_experiment_metrics(gain100)}
    baseline = metrics["gain1"]
    comparisons = {}
    for label in ("gain10", "gain100"):
        current = metrics[label]
        comparisons[f"{label}_vs_gain1"] = {f"{key}_fold_vs_gain1": _fold(current[key], baseline[key]) for key in current if key != "gain"}
    return {"scientific_interpretation": "pending", "experiments": metrics, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp2a-report", type=Path, required=True)
    parser.add_argument("--gain10-report", type=Path, required=True)
    parser.add_argument("--gain100-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_score_gain_comparison(*(json.loads(path.read_text()) for path in (args.exp2a_report, args.gain10_report, args.gain100_report)))
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "compared", "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
