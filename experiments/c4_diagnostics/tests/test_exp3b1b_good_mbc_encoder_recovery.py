from __future__ import annotations

import csv
from argparse import Namespace
from pathlib import Path

import pytest
import torch

import experiments.c4_diagnostics.exp3b1b_good_mbc_encoder_recovery as b1b
from experiments.c4_diagnostics.exp3b1b_good_mbc_encoder_recovery import _classify, _formal_fresh_args_valid, _formal_resume_args_valid, _parse_existing_b1_phase_b_at_step, _phase_a_protocol, _phase_b_eval_steps, _phase_b_protocol, _write_paired_comparison, run


PRIMARY = {
    "full_fov_dynamic_cosine": .93,
    "full_fov_dynamic_rms_ratio": 1.1,
    "full_fov_dynamic_normalized_epe": .6,
}


def _write_curve(path: Path, row: dict[str, object], *, include_phase_step: bool = True) -> None:
    fields = ["global_step", "phase"] + (["phase_step"] if include_phase_step else []) + [
        "full_fov_dynamic_cosine", "full_fov_dynamic_rms_ratio", "full_fov_dynamic_normalized_epe", "full_batch_reconstruction_mse",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def test_b1b_protocols_fix_the_two_phase_budgets_and_ownership() -> None:
    phase_a, phase_b = _phase_a_protocol(), _phase_b_protocol()
    assert phase_a == {"steps": 750, "observations_per_update": 32, "pixel_mode": "fixed_grid", "pixel_samples": 256, "objective": "data_only", "trainable_modules": ["respiratory_mbc.levels.0."], "gt_respiratory_score_for_training": True, "exposure_accounting": {"optimizer_steps": 750, "observations_per_update": 32, "pixels_per_observation": 256, "observation_exposures": 24000, "pixel_observation_exposures": 6144000}}
    assert phase_b == {"steps": 1000, "observations_per_update": 450, "pixel_mode": "fixed_grid", "pixel_samples": 256, "accumulation_chunk_size": 5, "film_lr": 5e-4, "objective": "reconstruction_image_data_mse_only", "trainable_modules": ["film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp."], "gt_respiratory_score_for_training": False, "exposure_accounting": {"optimizer_steps": 1000, "observations_per_update": 450, "pixels_per_observation": 256, "observation_exposures": 450000, "pixel_observation_exposures": 115200000}}


def test_historical_parser_reads_only_phase_b_phase_step_1000(tmp_path: Path) -> None:
    curve = tmp_path / "curve.csv"
    _write_curve(curve, {"global_step": 3000, "phase": "B", "phase_step": 1000, **PRIMARY, "full_batch_reconstruction_mse": .01})
    parsed = _parse_existing_b1_phase_b_at_step(curve)
    assert {key: parsed[key] for key in PRIMARY} == PRIMARY
    assert parsed["full_batch_reconstruction_mse"] == .01
    assert parsed["full_fov_static_rms_mm"] is None


@pytest.mark.parametrize(
    ("row", "include_phase_step"),
    [
        ({"global_step": 3000, "phase": "A", "phase_step": 1000, **PRIMARY, "full_batch_reconstruction_mse": .01}, True),
        ({"global_step": 3000, "phase": "B", "phase_step": 999, **PRIMARY, "full_batch_reconstruction_mse": .01}, True),
        ({"global_step": 3000, "phase": "B", "phase_step": 1000, **{**PRIMARY, "full_fov_dynamic_cosine": "nan"}, "full_batch_reconstruction_mse": .01}, True),
        ({"global_step": 2999, "phase": "B", **PRIMARY, "full_batch_reconstruction_mse": .01}, False),
    ],
)
def test_historical_parser_rejects_non_b1_phase_b_1000_rows(tmp_path: Path, row: dict[str, object], include_phase_step: bool) -> None:
    curve = tmp_path / "curve.csv"
    _write_curve(curve, row, include_phase_step=include_phase_step)
    with pytest.raises(ValueError):
        _parse_existing_b1_phase_b_at_step(curve)


def test_historical_parser_uses_global_step_3000_only_without_phase_step_column(tmp_path: Path) -> None:
    curve = tmp_path / "curve.csv"
    _write_curve(curve, {"global_step": 3000, "phase": "B", **PRIMARY, "full_batch_reconstruction_mse": .01}, include_phase_step=False)
    assert _parse_existing_b1_phase_b_at_step(curve)["full_fov_dynamic_cosine"] == .93


def test_classification_uses_good_mbc_gate_and_phase_b_dvf_thresholds() -> None:
    failure = {"full_fov_dynamic_cosine": .7, "full_fov_dynamic_rms_ratio": .2, "full_fov_dynamic_normalized_epe": 2.}
    escape = {"full_fov_dynamic_cosine": .91, "full_fov_dynamic_rms_ratio": 1.1, "full_fov_dynamic_normalized_epe": .7}
    strict = {"full_fov_dynamic_cosine": .96, "full_fov_dynamic_rms_ratio": 1., "full_fov_dynamic_normalized_epe": .4}
    assert _classify(failure, escape)["causal_classification"] == "good_mbc_bootstrap_failure"
    assert _classify(escape, failure)["causal_classification"] == "encoder_given_good_mbc_failure"
    assert _classify(escape, escape)["causal_classification"] == "good_mbc_enables_encoder_escape"
    assert _classify(escape, strict)["causal_classification"] == "good_mbc_enables_encoder_strict_recovery"


def _smoke_args(tmp_path: Path, name: str):
    root = Path(__file__).resolve().parents[3]
    return __import__("argparse").Namespace(source_config=root / "configs/source_first_change4_paperaligned.yaml", experiment_config=root / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml", manifest=None, qc_table=None, canonical_domain=None, output_dir=tmp_path / name, existing_b1_curve=None, device="cpu", overwrite=False, seed=0, locations_per_view=1, frames_per_location=3, phase_a_steps=1, phase_b_steps=2, phase_a_observations_per_update=3, pixel_mode="fixed_grid", pixel_samples=256, accumulation_chunk_size=2, eval_every=1, checkpoint_every=1, evaluation_grid_shape=(3, 3, 3), resume_from=None, smoke=True)


def test_phase_a_gate_short_circuits_before_encoder_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(b1b, "_escape", lambda _metric: False)
    summary = run(_smoke_args(tmp_path, "gate-failure"))
    assert summary["causal_classification"] == "good_mbc_bootstrap_failure"
    assert summary["phase_b_entered"] is False
    assert summary["stage_parameter_change_audit"]["A"]["unexpected"] == []
    assert not (Path(summary["output_dir"]) / "checkpoint_phase_b_1.pt").exists()


def test_good_mbc_gate_enters_imported_phase_b_with_frozen_mbc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(b1b, "_escape", lambda _metric: True)
    summary = run(_smoke_args(tmp_path, "phase-b"))
    assert summary["phase_b_entered"] is True
    assert summary["stage_parameter_change_audit"]["B"]["unexpected"] == []
    assert summary["runtime_audit"]["phase_a_encoder_calls"] == 0
    assert summary["runtime_audit"]["phase_b_encoder_calls"] > 0
    assert summary["git_sha"] == b1b.old_b1._git_sha()
    assert summary["historical_pair_available"] is False
    assert summary["exposure_accounting"]["total_observation_exposures"] == 474000
    assert summary["frozen_module_audit"]["respiratory_mbc_unchanged_during_b"] is True
    assert (Path(summary["output_dir"]) / "checkpoint_phase_b_2.pt").is_file()
    assert not (Path(summary["output_dir"]) / "paired_phase_b_1000_comparison.json").exists()


def test_phase_b_formal_evaluation_schedule_has_start_and_quarter_points() -> None:
    assert _phase_b_eval_steps(1000) == (0, 250, 500, 750, 1000)


def test_paired_comparison_keeps_missing_historical_secondary_metrics_unavailable(tmp_path: Path) -> None:
    new = {"aggregate": {"full_fov_dynamic_cosine": {"mean": .96}, "full_fov_dynamic_rms_ratio": {"mean": 1.0}, "full_fov_dynamic_normalized_epe": {"mean": .4}, "full_fov_dynamic_rms_mm": {"mean": 1.2}, "full_fov_static_rms_mm": {"mean": .1}, "pixel_center_dynamic_cosine": {"mean": .95}, "pixel_center_dynamic_rms_ratio": {"mean": 1.1}, "pixel_center_dynamic_normalized_epe": {"mean": .5}}}
    old = {**PRIMARY, "full_fov_dynamic_rms_mm": None, "full_fov_static_rms_mm": None, "pixel_center_dynamic_cosine": None, "pixel_center_dynamic_rms_ratio": None, "pixel_center_dynamic_normalized_epe": None, "full_batch_reconstruction_mse": .01}
    comparison = _write_paired_comparison(tmp_path, new, .02, old)
    assert comparison["metrics"]["full_fov_dynamic_cosine"] == {"old_b1_phase_b_1000": .93, "new_b1b_phase_b_1000": .96}
    assert comparison["metrics"]["pixel_center_dynamic_cosine"]["old_b1_phase_b_1000"] is None
    assert (tmp_path / "paired_phase_b_1000_comparison.csv").is_file()


def test_mid_phase_b_resume_continues_with_cpu_safe_checkpoint_rng(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(b1b, "_escape", lambda _metric: True)
    first = _smoke_args(tmp_path, "resume")
    first.phase_b_steps = 1
    run(first)
    checkpoint = first.output_dir / "checkpoint_phase_b_1.pt"
    resumed = _smoke_args(tmp_path, "resume")
    resumed.phase_b_steps = 2
    resumed.resume_from = checkpoint
    calls: list[str] = []; original = b1b.old_b1._assert_expected_gradients
    monkeypatch.setattr(b1b.old_b1, "_assert_expected_gradients", lambda model, phase: calls.append(phase) or original(model, phase))
    summary = run(resumed)
    assert summary["phase_b_entered"] is True
    assert calls == ["B"]
    assert (resumed.output_dir / "checkpoint_phase_b_2.pt").is_file()
    payload = b1b.old_b1._load_checkpoint(resumed.output_dir / "checkpoint_phase_b_2.pt")
    assert payload["protocol"]["phase_schedule"]["phase_b_steps"] == 2


def test_failed_phase_a_checkpoint_resume_still_short_circuits_before_phase_b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fresh = _smoke_args(tmp_path, "resume-gate")
    fresh.phase_b_steps = 1
    with monkeypatch.context() as patched:
        patched.setattr(b1b, "_escape", lambda _metric: False)
        run(fresh)
    checkpoint = fresh.output_dir / "checkpoint_phase_a_end.pt"
    payload = b1b.old_b1._load_checkpoint(checkpoint)
    aggregate = payload["phase_metrics"]["A"]["aggregate"]
    aggregate["full_fov_dynamic_cosine"]["mean"] = .1
    aggregate["full_fov_dynamic_rms_ratio"]["mean"] = 2.
    aggregate["full_fov_dynamic_normalized_epe"]["mean"] = 2.
    torch.save(payload, checkpoint)
    resumed = _smoke_args(tmp_path, "resume-gate")
    resumed.phase_b_steps = 2
    resumed.resume_from = fresh.output_dir / "checkpoint_phase_a_end.pt"
    summary = run(resumed)
    assert summary["causal_classification"] == "good_mbc_bootstrap_failure"
    assert summary["phase_b_entered"] is False
    assert not (resumed.output_dir / "checkpoint_phase_b_1.pt").exists()


def _formal_args(phase_b_steps: int) -> Namespace:
    return Namespace(phase_a_steps=750, phase_b_steps=phase_b_steps, phase_a_observations_per_update=32, pixel_mode="fixed_grid", pixel_samples=256, locations_per_view=3, frames_per_location=50, accumulation_chunk_size=5)


def test_formal_schedule_allows_b2000_only_for_resume() -> None:
    assert _formal_fresh_args_valid(_formal_args(1000))
    assert not _formal_fresh_args_valid(_formal_args(2000))
    assert _formal_resume_args_valid(_formal_args(1000))
    assert _formal_resume_args_valid(_formal_args(2000))
