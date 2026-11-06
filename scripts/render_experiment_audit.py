#!/usr/bin/env python3
"""Render a read-only, stage-wise experiment audit without retraining or reconstruction."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.visualization import adapter_for
from cardioresp4d.visualization.contract import ensure_safe_output_dir
from cardioresp4d.visualization.report import write_audit_manifest, write_html_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True); parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--qc-table", required=True, type=Path)
    parser.add_argument("--canonical-domain", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}: parser.error("--device must be cpu or cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available(): parser.error("CUDA requested but unavailable")
    adapter = adapter_for(args.experiment); sources = adapter.source_artifacts(args.experiment_root, args.manifest, args.qc_table, args.canonical_domain)
    output = ensure_safe_output_dir(args.output_dir, list(sources.values()))
    stages = adapter.render(experiment_root=args.experiment_root, manifest=args.manifest, qc_table=args.qc_table, canonical_domain=args.canonical_domain, output_dir=output, device=str(device))
    manifest_path = write_audit_manifest(output / "audit_manifest.json", adapter=adapter.name, sources=sources, stages=stages, display_policies={"stage_renderers": "recorded with each stage figure"}, render_configuration={"device": str(device), "read_only_scientific_artifacts": True})
    audit = json.loads(manifest_path.read_text()); audit["scientific_status"] = dict(adapter.scientific_status); manifest_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    summary = {**dict(adapter.scientific_status), "stage_status": stages}; (output / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_html_report(output / "report.html", audit, stages)
    print(json.dumps({"output_dir": str(output), **dict(adapter.scientific_status)}, indent=2))


if __name__ == "__main__":
    main()
