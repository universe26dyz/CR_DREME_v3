from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.c4_diagnostics.exp3b0_factor_diagnostics import audit


def test_summary_curve_audit_preserves_dvf_primary_and_reports_stage_regression(tmp_path: Path) -> None:
    summary = {"final_metrics": {"aggregate": {"full_fov_dynamic_cosine": {"mean": 0.82}, "full_fov_dynamic_rms_ratio": {"mean": 0.63}, "full_fov_dynamic_normalized_epe": {"mean": 0.91}}}, "interpretation_contract": "score metrics are diagnostics only"}
    summary_path = tmp_path / "summary.json"; summary_path.write_text(json.dumps(summary))
    curve_path = tmp_path / "curve.csv"
    with curve_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["global_step", "phase", "full_batch_reconstruction_mse", "full_fov_dynamic_cosine"]); writer.writeheader(); writer.writerows(({"global_step": 1, "phase": "A", "full_batch_reconstruction_mse": .4, "full_fov_dynamic_cosine": .9}, {"global_step": 2, "phase": "C", "full_batch_reconstruction_mse": .2, "full_fov_dynamic_cosine": .8}))
    result = audit(summary_path, curve_path, tmp_path / "audit")
    assert result["dvf_primary_metrics"]["full_fov_dynamic_cosine"] == .82
    assert result["curve_audit"]["best_reconstruction_step"] == 2
    assert result["curve_audit"]["best_full_fov_cosine_step"] == 1
    assert (tmp_path / "audit" / "factor_diagnostics.json").is_file()
    assert (tmp_path / "audit" / "factor_diagnostics.csv").is_file()
