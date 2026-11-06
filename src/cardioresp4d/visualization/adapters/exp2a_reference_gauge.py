"""Read-only visualization adapter for the isolated Exp2-A gauge output."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..contract import ExperimentVisualizationAdapter, STAGES, stage_status
from ..dynamic_audit import dynamic_field_labels


class Exp2AReferenceGaugeAdapter(ExperimentVisualizationAdapter):
    name = "exp2a_reference_gauge"

    @property
    def scientific_status(self) -> Mapping[str, str]:
        return {"experiment_scientific_status": "EXP2A_REFERENCE_GAUGE_PENDING_RETROSPECTIVE_AUDIT", "reason": "Exp2-A is isolated and must be interpreted only after the full-cohort Exp1 retrospective gauge audit."}

    def source_artifacts(self, experiment_root: Path, manifest: Path, qc_table: Path, canonical_domain: Path) -> dict[str, Path]:
        paths = {
            "checkpoint_step_200": experiment_root / "checkpoint_step_200.pt",
            "checkpoint_last": experiment_root / "checkpoint_last.pt",
            "training_report": experiment_root / "training_report.json",
            "experiment_config": experiment_root / "effective_config.json",
            "mean_manifest": experiment_root / "mean_slice_manifest.json",
            "static_manifest": experiment_root / "static_reference_manifest.json",
        }
        return {key: path for key, path in paths.items() if path.exists()}

    def render(self, *, experiment_root: Path, manifest: Path, qc_table: Path, canonical_domain: Path, output_dir: Path, device: str) -> dict[str, dict[str, Any]]:
        for stage in STAGES:
            (output_dir / stage).mkdir(parents=True, exist_ok=True)
        return {"V8_dynamic_zero_motion": stage_status("partial", "Exp2-A has no canonical-only boundary schedule"), "V9_dynamic_boundaries": self.render_v9_dynamic_boundaries(experiment_root=experiment_root, output=output_dir / "V9_dynamic_boundaries", device=device)}

    def render_v9_dynamic_boundaries(self, *, experiment_root: Path, output: Path, device: str = "cpu", **_: Any) -> dict[str, Any]:
        output.mkdir(parents=True, exist_ok=True)
        report_path = experiment_root / "training_report.json"
        if not report_path.exists():
            return stage_status("partial", "Exp2-A training_report.json is unavailable")
        report = json.loads(report_path.read_text())
        payload = {"experiment": self.name, "device": device, "step_0": report.get("diagnostics", {}).get("step_0", {}), "step_200": report.get("diagnostics", {}).get("step_200", {}), "runner_report": report.get("runner_report", {}), "provenance": report.get("provenance", {}), "source_artifact_metadata": {"checkpoint_step_200": str(experiment_root / "checkpoint_step_200.pt"), "training_report": str(report_path), "effective_config": str(experiment_root / "effective_config.json"), "mean_manifest_sha256": report.get("provenance", {}).get("mean_manifest_sha256"), "static_manifest_sha256": report.get("provenance", {}).get("static_manifest_sha256")}, "shared_dynamic_core": {"fixed_grid": "dynamic_audit.fixed_physical_grid", "representative_selection": "dynamic_audit.select_fixed_location_frames", "field_labels": dynamic_field_labels()}, "step_0_visualization": "metrics_only; no step-0 model checkpoint exists", "policy": "reuse Exp1 representative locations/grid when source observations are supplied"}
        (output / "exp2a_diagnostics.json").write_text(json.dumps(payload, indent=2) + "\n")
        return stage_status("rendered", diagnostics_file="exp2a_diagnostics.json", step_0_policy=payload["step_0_visualization"], score_policy="raw, reference and effective gauge diagnostics are recorded explicitly")
