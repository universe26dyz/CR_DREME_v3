#!/usr/bin/env python3
"""Build a machine-readable Exp1 L1-warm versus Exp2-A gauge comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_comparison_payload(exp1_gauge: dict, exp2a: dict, exp1_boundary: dict | list | None = None) -> dict:
    exp1_row = exp1_gauge.get("pooled", {}).get("L1_warm")
    if exp1_row is None:
        raise ValueError("Exp1 gauge audit does not contain L1_warm")
    all_locations = exp1_row.get("all_locations", exp1_row)
    authoritative = {}
    if exp1_boundary is not None:
        rows = exp1_boundary if isinstance(exp1_boundary, list) else exp1_boundary.get("boundaries", exp1_boundary.get("rows", []))
        if isinstance(rows, dict):
            rows = [rows]
        authoritative = next((row for row in rows if str(row.get("segment", row.get("checkpoint", ""))) in {"L1_warm", "checkpoint_L1_warm_end.pt"}), {})
    exp2_diag = exp2a.get("diagnostics", {}).get("step_200", {})
    exp2_report = exp2a.get("runner_report", {})
    row = {
        "primary_comparison": "Exp1 checkpoint_L1_warm_end.pt vs Exp2-A checkpoint_step_200.pt",
        "exp1_checkpoint": "checkpoint_L1_warm_end.pt", "exp2a_checkpoint": "checkpoint_step_200.pt",
        "exp1_reconstruction_mse_full": authoritative.get("reconstruction_mse_full"),
        "exp1_reconstruction_mse_SAX": authoritative.get("reconstruction_mse_SAX"),
        "exp1_reconstruction_mse_2CH": authoritative.get("reconstruction_mse_2CH"),
        "exp1_reconstruction_mse_4CH": authoritative.get("reconstruction_mse_4CH"),
        "exp2a_reconstruction_mse_full": exp2_report.get("reconstruction_mse_full"),
        "exp2a_reconstruction_mse_SAX": exp2_report.get("reconstruction_mse_SAX"),
        "exp2a_reconstruction_mse_2CH": exp2_report.get("reconstruction_mse_2CH"),
        "exp2a_reconstruction_mse_4CH": exp2_report.get("reconstruction_mse_4CH"),
        "exp1_raw_score_static_rms": all_locations["raw_score"]["static_rms"], "exp1_raw_score_dynamic_rms": all_locations["raw_score"]["dynamic_rms"], "exp1_raw_score_static_dynamic_ratio": all_locations["raw_score"]["static_dynamic_ratio"],
        "exp1_effective_score_static_rms": all_locations["effective_score"]["static_rms"], "exp1_effective_score_dynamic_rms": all_locations["effective_score"]["dynamic_rms"], "exp1_effective_score_static_dynamic_ratio": all_locations["effective_score"]["static_dynamic_ratio"],
        "exp1_raw_dvf_static_rms_mm": all_locations["raw_dvf"]["static_rms_mm"], "exp1_raw_dvf_dynamic_rms_mm": all_locations["raw_dvf"]["dynamic_rms_mm"], "exp1_effective_dvf_static_rms_mm": all_locations["gauge_dvf"]["static_rms_mm"], "exp1_effective_dvf_dynamic_rms_mm": all_locations["gauge_dvf"]["dynamic_rms_mm"],
        "exp1_metric_source": "authoritative boundary diagnostics plus full-cohort retrospective gauge audit",
        "exp2a_metric_source": "step-200 full-cohort diagnostics",
    }
    for name in ("raw", "effective"):
        row[f"exp2a_{name}_score_static_rms"] = exp2_diag.get(f"{name}_score_static_rms"); row[f"exp2a_{name}_score_dynamic_rms"] = exp2_diag.get(f"{name}_score_dynamic_rms"); row[f"exp2a_{name}_score_static_dynamic_ratio"] = exp2_diag.get(f"{name}_score_static_dynamic_ratio"); row[f"exp2a_{name}_dvf_static_rms_mm"] = exp2_diag.get(f"{name}_dvf_static_rms_mm"); row[f"exp2a_{name}_dvf_dynamic_rms_mm"] = exp2_diag.get(f"{name}_dvf_dynamic_rms_mm")
    row["exp2a_mbc_parameter_norm"] = exp2_report.get("respiratory_mbc_parameter_norm"); row["exp2a_encoder_parameter_norm"] = exp2_report.get("respiratory_encoder_parameter_norm"); row["exp2a_gradient_summary"] = exp2_report.get("gradient_summary")
    return {"scientific_interpretation": "pending", "comparison": row, "exp2a_protocol": exp2a.get("protocol")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp1-gauge-json", type=Path, required=True)
    parser.add_argument("--exp2a-report", type=Path, required=True)
    parser.add_argument("--exp1-boundary-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exp1 = json.loads(args.exp1_gauge_json.read_text()); exp2a = json.loads(args.exp2a_report.read_text()); exp1_boundary = json.loads(args.exp1_boundary_json.read_text()) if args.exp1_boundary_json else None
    payload = build_comparison_payload(exp1, exp2a, exp1_boundary)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "written", "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
