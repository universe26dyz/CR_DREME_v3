#!/usr/bin/env python3
"""One-command, restart-safe, read-only Change5C closure orchestration."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from cardioresp4d.diagnostics.change5c import paired_ablation_comparison


def _experiment(value: str) -> tuple[str, Path, Path]:
    if "=" not in value or "," not in value.split("=", 1)[1]: raise argparse.ArgumentTypeError("--experiment must be LABEL=CONFIG,CHECKPOINT")
    label, rest = value.split("=", 1); config, checkpoint = rest.split(",", 1); return label, Path(config), Path(checkpoint)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle: subprocess.run(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _output(path: Path, reuse: bool) -> bool:
    if not path.exists(): return False
    if reuse: return True
    raise FileExistsError(f"refusing to overwrite existing output without --reuse-valid: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="append", type=_experiment, required=True, metavar="LABEL=CONFIG,CHECKPOINT"); parser.add_argument("--frequency-bands", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--qc-table", type=Path, required=True); parser.add_argument("--canonical-domain", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", default="cuda"); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--reuse-valid", action="store_true")
    args = parser.parse_args(); experiments = args.experiment
    if len(experiments) < 3: parser.error("Change5C expects CTRL, LATE, and C5B experiments")
    for _, config, checkpoint in experiments:
        for path in (config, checkpoint):
            if not path.is_file(): parser.error(f"required input is missing: {path}")
    for path in (args.frequency_bands, args.manifest, args.qc_table, args.canonical_domain):
        if not path.is_file(): parser.error(f"required common input is missing: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True); logs = args.output_dir / "logs"; python = sys.executable
    contract = args.output_dir / "checkpoint_contract_audit.json"
    if not _output(contract, args.reuse_valid): _run([python, "scripts/audit_checkpoint_contract.py", *[item for label, config, checkpoint in experiments for item in ("--experiment", f"{label}={config},{checkpoint}")], "--output-json", str(contract)], logs / "checkpoint_contract_audit.log")
    contract_records = json.loads(contract.read_text(encoding="utf-8"))["records"]
    for record in contract_records:
        if record.get("current_stage") != "stage3a": raise ValueError(f"{record['label']} is not a Stage3a checkpoint: {record.get('current_stage')}")
        weights = record["effective_loss_weights"]; label = record["label"].upper()
        expected = {"CTRL": {"cardiac_pca_waveform": 0.}, "LATE": {"cardiac_pca_waveform": 1e-4}, "C5B": {"cardiac_target_concentration": .003, "cardiac_pca_waveform": .001}}
        for token, expected_weights in expected.items():
            if token in label:
                for name, value in expected_weights.items():
                    if float(weights.get(name, float("nan"))) != value: raise ValueError(f"{record['label']} effective {name}={weights.get(name)}; expected {value}")
    semantic_paths, ablation_paths = {}, {}
    for label, config, checkpoint in experiments:
        directory = args.output_dir / label; semantic = directory / "all_location_diagnostic.json"; ablation = directory / "all_location_cardiac_ablation.json"; semantic_paths[label] = semantic; ablation_paths[label] = ablation
        if not _output(semantic, args.reuse_valid): _run([python, "scripts/diagnose_change4_checkpoint.py", "--source-config", str(config), "--frequency-bands", str(args.frequency_bands), "--manifest", str(args.manifest), "--qc-table", str(args.qc_table), "--canonical-domain", str(args.canonical_domain), "--checkpoint", str(checkpoint), "--device", args.device, "--all-locations", "--all-locations-dir", str(directory), "--output-json", str(directory / "legacy_diagnostic.json")], logs / f"{label}_semantic.log")
        if not _output(ablation, args.reuse_valid): _run([python, "scripts/audit_all_location_cardiac_ablation.py", "--source-config", str(config), "--manifest", str(args.manifest), "--qc-table", str(args.qc_table), "--canonical-domain", str(args.canonical_domain), "--checkpoint", str(checkpoint), "--device", args.device, "--output-json", str(ablation), "--output-csv", str(directory / "all_location_cardiac_ablation.csv")], logs / f"{label}_ablation.log")
        for mode in ("location-matched", "training-step-matched"):
            output = directory / f"stage3a_gradient_{mode}.json"
            if not _output(output, args.reuse_valid): _run([python, "scripts/audit_stage3a_loss_gradients.py", "--source-config", str(config), "--frequency-bands", str(args.frequency_bands), "--manifest", str(args.manifest), "--qc-table", str(args.qc_table), "--canonical-domain", str(args.canonical_domain), "--checkpoint", str(checkpoint), "--mode", mode, "--view", "SAX", "--slice-id", "SAX_s026", "--device", args.device, "--seed", str(args.seed), "--output-json", str(output)], logs / f"{label}_gradient_{mode}.log")
    baseline = experiments[0][0]; comparisons = {}
    for label, _, _ in experiments[1:]:
        left, right = json.loads(ablation_paths[baseline].read_text(encoding="utf-8"))["records"], json.loads(ablation_paths[label].read_text(encoding="utf-8"))["records"]
        comparisons[label] = paired_ablation_comparison([row for row in left if row.get("status") == "evaluated"], [row for row in right if row.get("status") == "evaluated"])
    comparison_path = args.output_dir / "compare_cardiac_ablation.json"; comparison_path.write_text(json.dumps({"baseline": baseline, "comparisons": comparisons}, indent=2) + "\n", encoding="utf-8")
    candidate = next((label for label, _, _ in experiments if "C5B" in label.upper()), experiments[-1][0]); locations = args.output_dir / "selected_visualization_locations.json"
    if not _output(locations, args.reuse_valid): _run([python, "scripts/select_change5c_visualization_locations.py", "--baseline-semantic", str(semantic_paths[baseline]), "--candidate-semantic", str(semantic_paths[candidate]), "--candidate-ablation", str(ablation_paths[candidate]), "--output-json", str(locations)], logs / "select_locations.log")
    selected = json.loads(locations.read_text(encoding="utf-8"))["locations"]
    for label, config, checkpoint in experiments:
        for location in selected:
            view, slice_id = location["view"], location["slice_id"]; output = args.output_dir / label / f"visualize_{view}_{slice_id}"
            if not _output(output, args.reuse_valid): _run([python, "scripts/visualize_checkpoint_dynamics.py", "--source-config", str(config), "--manifest", str(args.manifest), "--qc-table", str(args.qc_table), "--canonical-domain", str(args.canonical_domain), "--checkpoint", str(checkpoint), "--view", view, "--slice-id", slice_id, "--device", args.device, "--seed", str(args.seed), "--output-dir", str(output)], logs / f"{label}_{view}_{slice_id}_visual.log")
    comparison_visual = args.output_dir / "visual_comparison"
    if not _output(comparison_visual, args.reuse_valid): _run([python, "scripts/visualize_change5c_comparison.py", *[item for label, _, _ in experiments for item in ("--input", f"{label}={args.output_dir / label}")], "--locations-json", str(locations), "--output-dir", str(comparison_visual)], logs / "visual_comparison.log")
    provenance = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(), "common_input_sha256": {name: _sha256(path) for name, path in {"frequency_bands": args.frequency_bands, "manifest": args.manifest, "qc_table": args.qc_table, "canonical_domain": args.canonical_domain}.items()}, "experiments": [{"label": label, "config": str(config), "checkpoint": str(checkpoint), "config_sha256": _sha256(config), "checkpoint_sha256": _sha256(checkpoint)} for label, config, checkpoint in experiments]}
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    per_checkpoint = {label: {"semantic_summary": json.loads(semantic_paths[label].read_text(encoding="utf-8")).get("summary", json.loads(semantic_paths[label].read_text(encoding="utf-8")).get("all_locations", {}).get("summary")), "cardiac_ablation": json.loads(ablation_paths[label].read_text(encoding="utf-8"))["overall"], "gradient_location_matched": str(args.output_dir / label / "stage3a_gradient_location-matched.json"), "gradient_training_step_matched": str(args.output_dir / label / "stage3a_gradient_training-step-matched.json")} for label, _, _ in experiments}
    summary = {"status": "completed_read_only", "baseline": baseline, "experiments": per_checkpoint, "outputs": {"contract": str(contract), "ablation_comparison": str(comparison_path), "visualization_locations": str(locations), "visual_evidence": str(comparison_visual)}}
    (args.output_dir / "change5c_closure_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_lines = ["# Change5C closure report", "", "Read-only descriptive analysis; positive MSE gain is not a statistical-significance claim.", "", "## Per checkpoint"]
    for label, payload in per_checkpoint.items(): report_lines.extend([f"- **{label}**: cardiac-ROI ablation summary `{payload['cardiac_ablation']}`; gradient audits: `{Path(payload['gradient_location_matched']).name}`, `{Path(payload['gradient_training_step_matched']).name}`."])
    report_lines.extend(["", "## Paired comparisons", f"- Baseline: `{baseline}`. Exact-key descriptive reconstruction deltas: `compare_cardiac_ablation.json`.", "", "## Visual evidence", "- `visual_comparison/*_comparison.png` and optional GIF: fixed-scale acquired/resp-only/joint/residual/cardiac-effect cine/contact-sheet panels.", "- `visual_comparison/*_dvf_jacobian.png`: cardiac/respiratory/total DVF maps plus Jacobian slice and numeric RMS/p95/max/fraction<=0.", "- `visual_comparison/*_cardiac_effect_3d.png`: resp-only, joint, and isolated cardiac-effect implied 3D planes.", "", "Observation-conditioned implied 3D dynamics are not globally synchronized physiological cine."])
    (args.output_dir / "change5c_closure_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
