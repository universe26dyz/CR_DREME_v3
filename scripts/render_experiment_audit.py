#!/usr/bin/env python3
"""Render a read-only, stage-wise experiment audit without retraining or reconstruction."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.visualization import adapter_for
from cardioresp4d.visualization.contract import STAGES, ensure_safe_output_dir
from cardioresp4d.visualization.report import write_audit_manifest, write_html_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True); parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--qc-table", required=True, type=Path)
    parser.add_argument("--canonical-domain", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stages", help="comma-separated stage names, e.g. V8_dynamic_zero_motion,V9_dynamic_boundaries")
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}: parser.error("--device must be cpu or cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    requested_stages = None
    if args.stages is not None:
        requested_stages = tuple(item.strip() for item in args.stages.split(",") if item.strip())
        unknown = sorted(set(requested_stages) - set(STAGES))
        if unknown:
            parser.error(f"unknown stage(s): {', '.join(unknown)}")
        if not requested_stages:
            parser.error("--stages must contain at least one stage")
    adapter = adapter_for(args.experiment); sources = adapter.source_artifacts(args.experiment_root, args.manifest, args.qc_table, args.canonical_domain)
    output = ensure_safe_output_dir(args.output_dir, list(sources.values()))
    stages = adapter.render(experiment_root=args.experiment_root, manifest=args.manifest, qc_table=args.qc_table, canonical_domain=args.canonical_domain, output_dir=output, device=str(device), stages=requested_stages)
    manifest_path = write_audit_manifest(output / "audit_manifest.json", adapter=adapter.name, sources=sources, stages=stages, display_policies={"stage_renderers": "recorded with each stage figure"}, render_configuration={"device": str(device), "read_only_scientific_artifacts": True})
    audit = json.loads(manifest_path.read_text()); audit["scientific_status"] = dict(adapter.scientific_status); manifest_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    summary = {**dict(adapter.scientific_status), "stage_status": stages}; (output / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_html_report(output / "report.html", audit, stages)
    mandatory = {f"V{index}_{name}" for index, name in enumerate(("raw_acquisition", "normalization", "temporal_mean", "stack_geometry", "registration", "static_support", "static_inr", "static_reprojection"))}
    mandatory_requested = mandatory if requested_stages is None else mandatory.intersection(requested_stages)
    failed = {name: value.get("reason", "unknown failure") for name, value in stages.items() if name in mandatory_requested and value.get("render_status") == "failed"}
    if requested_stages is not None:
        failed = {name: value.get("reason", "unknown failure") for name, value in stages.items() if name in requested_stages and value.get("render_status") == "failed"}
        if failed:
            print(json.dumps({"requested_stage_failures": failed, "report": str(output / "report.html")}, indent=2), file=sys.stderr)
            return 2
    if failed:
        print(json.dumps({"mandatory_stage_failures": failed, "report": str(output / "report.html")}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(output), **dict(adapter.scientific_status)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
