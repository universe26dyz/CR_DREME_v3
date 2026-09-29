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
from cardioresp4d.diagnostics.change5c import paired_ablation_comparison, provenance_matches

SLICE_CHUNK_SIZE = 1024
VOLUME_CHUNK_SIZE = 65536


def _experiment(value: str) -> tuple[str, Path, Path]:
    if "=" not in value or "," not in value.split("=", 1)[1]: raise argparse.ArgumentTypeError("--experiment must be LABEL=CONFIG,CHECKPOINT")
    label, rest = value.split("=", 1); config, checkpoint = rest.split(",", 1); return label, Path(config), Path(checkpoint)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def closure_sampling_metadata() -> dict:
    return {"gradient_pixel_samples": 256, "ablation_frames_per_location": 5, "ablation_max_cardiac_pixels": 512, "visualization": {"slice_chunk_size": SLICE_CHUNK_SIZE, "volume_chunk_size": VOLUME_CHUNK_SIZE}}


def visualization_command(python: str, config: Path, manifest: Path, qc_table: Path, canonical_domain: Path, checkpoint: Path, view: str, slice_id: str, device: str, seed: int, output: Path) -> list[str]:
    return [python, "scripts/visualize_checkpoint_dynamics.py", "--source-config", str(config), "--manifest", str(manifest), "--qc-table", str(qc_table), "--canonical-domain", str(canonical_domain), "--checkpoint", str(checkpoint), "--view", view, "--slice-id", slice_id, "--device", device, "--seed", str(seed), "--slice-chunk-size", str(SLICE_CHUNK_SIZE), "--volume-chunk-size", str(VOLUME_CHUNK_SIZE), "--output-dir", str(output)]


def _run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle: subprocess.run(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _completion_path(path: Path) -> Path:
    return path / ".change5c_completion.json" if path.is_dir() else path.with_name(path.name + ".completion.json")


def _output(path: Path, reuse: bool, expected: dict) -> bool:
    if not path.exists(): return False
    if reuse:
        completion = _completion_path(path)
        actual = json.loads(completion.read_text(encoding="utf-8")) if completion.is_file() else None
        if provenance_matches(expected, actual): return True
        raise ValueError(f"refusing to reuse incomplete or provenance-mismatched output: {path}")
    raise FileExistsError(f"refusing to overwrite existing output without --reuse-valid: {path}")


def _mark_completed(path: Path, provenance: dict) -> None:
    completion = _completion_path(path); completion.parent.mkdir(parents=True, exist_ok=True)
    completion.write_text(json.dumps({**provenance, "completion": True}, indent=2) + "\n", encoding="utf-8")


def render_closure_report(per_checkpoint: dict, comparisons: dict, contract_by_label: dict, baseline: str) -> str:
    """Render the gradient section from the loss-first audit JSON contract."""
    del comparisons, contract_by_label, baseline
    lines = []
    for label, payload in per_checkpoint.items():
        location_audit = payload["gradient_location_matched"]["audits"][0]
        head_data = location_audit["data"]["cardiac_film_head"]["raw_l2_norm"]
        head_semantic = location_audit["semantic_aux"]["cardiac_film_head"]["combined_l2_norm"]
        mbc_data = location_audit["data"]["cardiac_mbc"]["raw_l2_norm"]
        mbc_total = location_audit["total"]["cardiac_mbc"]["combined_l2_norm"]
        concentration = location_audit.get("cardiac_target_concentration", {}).get("cardiac_mbc", {}).get("raw_l2_norm")
        pca = location_audit.get("cardiac_pca_waveform", {}).get("cardiac_mbc", {}).get("raw_l2_norm")
        ratio = location_audit["combined_ratios"]["cardiac_film_head"]["semantic_aux_to_data_norm_ratio"]
        cosine = location_audit["cosine_similarity"]["cardiac_film_head"].get("data_vs_semantic_aux")
        lines.append(f"- **{label} location-matched:** head data={head_data:.4g}, semantic={head_semantic:.4g}, ratio={ratio}, cos(data,semantic)={cosine}; MBC data={mbc_data:.4g}, total={mbc_total:.4g}, direct concentration/PCA raw={concentration}/{pca}.")
        lines.append(f"- **{label} training-step aggregate:** `{payload['gradient_training_step_matched']['training_step_aggregate']}`")
    return "\n".join(lines)


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
    provenance = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(), "seed": args.seed, "sampling": closure_sampling_metadata(), "inputs": {"frequency_bands_sha256": _sha256(args.frequency_bands), "manifest_sha256": _sha256(args.manifest), "qc_table_sha256": _sha256(args.qc_table), "canonical_domain_sha256": _sha256(args.canonical_domain), "experiments": [{"label": label, "config_sha256": _sha256(config), "checkpoint_sha256": _sha256(checkpoint)} for label, config, checkpoint in experiments]}}
    contract = args.output_dir / "checkpoint_contract_audit.json"
    if not _output(contract, args.reuse_valid, provenance):
        _run([python, "scripts/audit_checkpoint_contract.py", *[item for label, config, checkpoint in experiments for item in ("--experiment", f"{label}={config},{checkpoint}")], "--output-json", str(contract)], logs / "checkpoint_contract_audit.log"); _mark_completed(contract, provenance)
    contract_records = json.loads(contract.read_text(encoding="utf-8"))["records"]
    for record in contract_records:
        if record.get("current_stage") != "stage3a": raise ValueError(f"{record['label']} is not a Stage3a checkpoint: {record.get('current_stage')}")
        if not record["relevant_loss_weight_agreement"]["all_relevant_keys_agree"]: raise ValueError(f"{record['label']} supplied YAML disagrees with checkpoint-effective loss weights")
        weights = record["checkpoint_effective_loss_weights"]; label = record["label"].upper()
        expected = {"CTRL": {"cardiac_pca_waveform": 0.}, "LATE": {"cardiac_pca_waveform": 1e-4}, "C5B": {"cardiac_target_concentration": .003, "cardiac_pca_waveform": .001}}
        for token, expected_weights in expected.items():
            if token in label:
                for name, value in expected_weights.items():
                    if float(weights.get(name, float("nan"))) != value: raise ValueError(f"{record['label']} effective {name}={weights.get(name)}; expected {value}")
    semantic_paths, ablation_paths = {}, {}
    for label, config, checkpoint in experiments:
        directory = args.output_dir / label; semantic = directory / "all_location_diagnostic.json"; ablation = directory / "all_location_cardiac_ablation.json"; semantic_paths[label] = semantic; ablation_paths[label] = ablation
        if not _output(semantic, args.reuse_valid, provenance):
            _run([python, "scripts/diagnose_change4_checkpoint.py", "--source-config", str(config), "--frequency-bands", str(args.frequency_bands), "--manifest", str(args.manifest), "--qc-table", str(args.qc_table), "--canonical-domain", str(args.canonical_domain), "--checkpoint", str(checkpoint), "--device", args.device, "--all-locations", "--all-locations-dir", str(directory), "--output-json", str(directory / "legacy_diagnostic.json")], logs / f"{label}_semantic.log"); _mark_completed(semantic, provenance)
        if not _output(ablation, args.reuse_valid, provenance):
            _run([python, "scripts/audit_all_location_cardiac_ablation.py", "--source-config", str(config), "--manifest", str(args.manifest), "--qc-table", str(args.qc_table), "--canonical-domain", str(args.canonical_domain), "--checkpoint", str(checkpoint), "--device", args.device, "--seed", str(args.seed), "--output-json", str(ablation), "--output-csv", str(directory / "all_location_cardiac_ablation.csv")], logs / f"{label}_ablation.log"); _mark_completed(ablation, provenance)
        for mode in ("location-matched", "training-step-matched"):
            output = directory / f"stage3a_gradient_{mode}.json"
            if not _output(output, args.reuse_valid, provenance):
                _run([python, "scripts/audit_stage3a_loss_gradients.py", "--source-config", str(config), "--frequency-bands", str(args.frequency_bands), "--manifest", str(args.manifest), "--qc-table", str(args.qc_table), "--canonical-domain", str(args.canonical_domain), "--checkpoint", str(checkpoint), "--mode", mode, "--view", "SAX", "--slice-id", "SAX_s026", "--device", args.device, "--seed", str(args.seed), "--output-json", str(output)], logs / f"{label}_gradient_{mode}.log"); _mark_completed(output, provenance)
    baseline = experiments[0][0]; comparisons = {}
    for label, _, _ in experiments[1:]:
        left, right = json.loads(ablation_paths[baseline].read_text(encoding="utf-8"))["records"], json.loads(ablation_paths[label].read_text(encoding="utf-8"))["records"]
        comparisons[label] = paired_ablation_comparison([row for row in left if row.get("status") == "evaluated"], [row for row in right if row.get("status") == "evaluated"])
    comparison_path = args.output_dir / "compare_cardiac_ablation.json"; comparison_path.write_text(json.dumps({"baseline": baseline, "comparisons": comparisons}, indent=2) + "\n", encoding="utf-8")
    candidate = next((label for label, _, _ in experiments if "C5B" in label.upper()), experiments[-1][0]); locations = args.output_dir / "selected_visualization_locations.json"
    if not _output(locations, args.reuse_valid, provenance):
        _run([python, "scripts/select_change5c_visualization_locations.py", "--baseline-semantic", str(semantic_paths[baseline]), "--candidate-semantic", str(semantic_paths[candidate]), "--baseline-ablation", str(ablation_paths[baseline]), "--candidate-ablation", str(ablation_paths[candidate]), "--output-json", str(locations)], logs / "select_locations.log"); _mark_completed(locations, provenance)
    selected = json.loads(locations.read_text(encoding="utf-8"))["locations"]
    for label, config, checkpoint in experiments:
        for location in selected:
            view, slice_id = location["view"], location["slice_id"]; output = args.output_dir / label / f"visualize_{view}_{slice_id}"
            if not _output(output, args.reuse_valid, provenance):
                _run(visualization_command(python, config, args.manifest, args.qc_table, args.canonical_domain, checkpoint, view, slice_id, args.device, args.seed, output), logs / f"{label}_{view}_{slice_id}_visual.log"); _mark_completed(output, provenance)
    comparison_visual = args.output_dir / "visual_comparison"
    if not _output(comparison_visual, args.reuse_valid, provenance):
        _run([python, "scripts/visualize_change5c_comparison.py", *[item for label, _, _ in experiments for item in ("--input", f"{label}={args.output_dir / label}")], "--locations-json", str(locations), "--output-dir", str(comparison_visual)], logs / "visual_comparison.log"); _mark_completed(comparison_visual, provenance)
    for location in selected:
        view, slice_id = location["view"], location["slice_id"]; score_png = comparison_visual / f"{view}_{slice_id}_scores_pca_spectrum.png"
        if not _output(score_png, args.reuse_valid, provenance):
            _run([python, "scripts/visualize_change5c_score_comparison.py", *[item for label, config, checkpoint in experiments for item in ("--experiment", f"{label}={config},{checkpoint}")], "--frequency-bands", str(args.frequency_bands), "--manifest", str(args.manifest), "--qc-table", str(args.qc_table), "--canonical-domain", str(args.canonical_domain), "--view", view, "--slice-id", slice_id, "--device", args.device, "--output-png", str(score_png), "--output-json", str(comparison_visual / f"{view}_{slice_id}_scores_pca_spectrum.json")], logs / f"{view}_{slice_id}_score_visual.log"); _mark_completed(score_png, provenance)
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    semantic_payloads = {label: json.loads(semantic_paths[label].read_text(encoding="utf-8")) for label, _, _ in experiments}
    per_checkpoint = {label: {"semantic_summary": semantic_payloads[label]["summary"], "motion": semantic_payloads[label].get("aggregate_motion_statistics"), "cardiac_ablation": json.loads(ablation_paths[label].read_text(encoding="utf-8")), "gradient_location_matched": json.loads((args.output_dir / label / "stage3a_gradient_location-matched.json").read_text(encoding="utf-8")), "gradient_training_step_matched": json.loads((args.output_dir / label / "stage3a_gradient_training-step-matched.json").read_text(encoding="utf-8"))} for label, _, _ in experiments}
    summary = {"status": "completed_read_only", "baseline": baseline, "experiments": per_checkpoint, "outputs": {"contract": str(contract), "ablation_comparison": str(comparison_path), "visualization_locations": str(locations), "visual_evidence": str(comparison_visual)}}
    (args.output_dir / "change5c_closure_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    def fmt(value) -> str: return "n/a" if value is None else f"{float(value):.4g}"
    report_lines = ["# Change5C closure report", "", "Read-only descriptive analysis; positive MSE gain is not a statistical-significance claim.", "", "## Checkpoint contract", "", "|checkpoint|stage|global/stage step|checkpoint concentration/PCA|frozen groups equal|", "|---|---|---|---|---|"]
    contract_by_label = {row["label"]: row for row in contract_records}
    for label, payload in per_checkpoint.items():
        row = contract_by_label[label]; weights = row["checkpoint_effective_loss_weights"]
        frozen = ", ".join(name for name, value in json.loads(contract.read_text(encoding="utf-8"))["frozen_group_comparison"].items() if value["all_equal"])
        report_lines.append(f"|{label}|{row['current_stage']}|{row['global_step']}/{row['stage_step']}|{weights.get('cardiac_target_concentration')}/{weights.get('cardiac_pca_waveform')}|{frozen or 'differences reported'}|")
    report_lines.extend(["", "## Full-location semantic and reconstruction results", "", "|checkpoint|PCA R² mean/median|target fraction mean/median|target/wrong mean/median|card score std mean/median|exact/within peak|ROI gain mean/median|positive/negative/zero|effect mean/median|", "|---|---|---|---|---|---|---|---|---|"])
    for label, payload in per_checkpoint.items():
        continuous, boolean = payload["semantic_summary"]["overall"]["continuous"], payload["semantic_summary"]["overall"]["boolean"]; ablation = payload["cardiac_ablation"]["overall"]
        pair = lambda name: f"{fmt(continuous[name]['mean'])}/{fmt(continuous[name]['median'])}"
        report_lines.append(f"|{label}|{pair('cardiac_pca_waveform_r2')}|{pair('cardiac_target_fraction')}|{pair('card_target_over_wrong')}|{pair('card_score_std')}|{boolean['same_peak_exact']['count']}/{boolean['same_peak_within_0.02_hz']['count']}|{fmt(ablation['relative_mse_improvement_percent']['mean'])}/{fmt(ablation['relative_mse_improvement_percent']['median'])}|{ablation['n_positive_gain']}/{ablation['n_negative_gain']}/{ablation['n_zero_gain']}|{fmt(ablation['mean_abs_joint_minus_resp']['mean'])}/{fmt(ablation['mean_abs_joint_minus_resp']['median'])}|")
        report_lines.extend([f"- {label} per view: `{payload['cardiac_ablation']['per_view']}`"])
    report_lines.extend(["", "## Paired reconstruction changes", "", f"Baseline: `{baseline}`. `{json.dumps(comparisons)}`", "", "## Gradient competition", ""])
    report_lines.extend(render_closure_report(per_checkpoint, comparisons, contract_by_label, baseline).splitlines())
    report_lines.extend(["", "## Motion, Jacobian, and visual evidence", "", "Selected-frame cardiac DVF RMS/p95/max and Jacobian min/p01/median/p99/max/fraction<=0 are annotated in each `*_dvf_jacobian.png`.", "- `visual_comparison/*_comparison.png` and optional GIF: fixed-scale reprojection/cardiac-effect cine/contact sheets.", "- `visual_comparison/*_scores_pca_spectrum.png`: true-timestamp score/PCA/NUDFT/band evidence.", "- `visual_comparison/*_dvf_jacobian.png`: cardiac DVF magnitude + quiver and cardiac-centered Jacobian plane.", "- `visual_comparison/*_cardiac_effect_3d.png`: cardiac-centered orthogonal resp-only/joint/effect planes.", "", "Observation-conditioned implied 3D dynamics are not globally synchronized physiological cine."])
    (args.output_dir / "change5c_closure_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
