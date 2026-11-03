from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch

import experiments.c4_diagnostics.exp3b1b_good_mbc_encoder_recovery as b1b
import experiments.c4_diagnostics.exp3b1c_good_state_joint_unfreeze as b1c


ESCAPE = {"full_fov_dynamic_cosine": .91, "full_fov_dynamic_rms_ratio": 1.0, "full_fov_dynamic_normalized_epe": .7, "full_fov_static_rms_mm": .2}
STRICT = {"full_fov_dynamic_cosine": .96, "full_fov_dynamic_rms_ratio": 1.0, "full_fov_dynamic_normalized_epe": .4, "full_fov_static_rms_mm": .1}
FAIL = {"full_fov_dynamic_cosine": .7, "full_fov_dynamic_rms_ratio": .2, "full_fov_dynamic_normalized_epe": 1.2, "full_fov_static_rms_mm": .6}


def _formal_source_checkpoint(*, phase: str = "B", global_step: int = 2750, phase_a_steps: int = 750, phase_b_steps: int = 2000) -> dict:
    return {
        "phase": phase,
        "global_step": global_step,
        "seed": 0,
        "protocol": {"phase_a": b1b._phase_a_protocol(), "phase_b": b1b._phase_b_protocol(), "phase_schedule": {"phase_a_steps": phase_a_steps, "phase_b_steps": phase_b_steps}, "no_phase_c": True},
        "observation_audit": {"n_locations": 9, "n_frames": 450, "frames_per_location": 50},
        "pixel_protocol": {"pixel_mode": "fixed_grid", "actual_pixel_count": 256},
    }


def test_formal_source_admission_requires_b2000_but_accepts_static_phase_b_1000_metadata() -> None:
    checkpoint = _formal_source_checkpoint()
    b1c._admit_source_checkpoint(checkpoint, seed=0, observation_audit=checkpoint["observation_audit"], pixel_protocol=checkpoint["pixel_protocol"], formal=True)
    for key, value in (("phase", "A"), ("global_step", 2749)):
        invalid = _formal_source_checkpoint(**{key: value})
        with pytest.raises(ValueError):
            b1c._admit_source_checkpoint(invalid, seed=0, observation_audit=invalid["observation_audit"], pixel_protocol=invalid["pixel_protocol"], formal=True)
    invalid_schedule = _formal_source_checkpoint(phase_b_steps=1000)
    with pytest.raises(ValueError):
        b1c._admit_source_checkpoint(invalid_schedule, seed=0, observation_audit=invalid_schedule["observation_audit"], pixel_protocol=invalid_schedule["pixel_protocol"], formal=True)


def test_joint_trajectory_classification_uses_only_escape_and_strict_thresholds() -> None:
    assert b1c._classify_joint_trajectory(FAIL, [FAIL])["causal_classification"] == "invalid_good_staged_source"
    assert b1c._classify_joint_trajectory(ESCAPE, [FAIL])["causal_classification"] == "joint_unfreeze_regression_from_good_state"
    assert b1c._classify_joint_trajectory(ESCAPE, [FAIL, STRICT])["causal_classification"] == "joint_unfreeze_escape_instability"
    assert b1c._classify_joint_trajectory(ESCAPE, [ESCAPE])["causal_classification"] == "joint_unfreeze_preserves_escape"
    assert b1c._classify_joint_trajectory(ESCAPE, [STRICT])["causal_classification"] == "joint_unfreeze_reaches_strict_recovery"


def _b1b_smoke_args(tmp_path: Path, name: str) -> Namespace:
    root = Path(__file__).resolve().parents[3]
    return Namespace(source_config=root / "configs/source_first_change4_paperaligned.yaml", experiment_config=root / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml", manifest=None, qc_table=None, canonical_domain=None, output_dir=tmp_path / name, existing_b1_curve=None, device="cpu", overwrite=False, seed=0, locations_per_view=1, frames_per_location=3, phase_a_steps=1, phase_b_steps=2, phase_a_observations_per_update=3, pixel_mode="fixed_grid", pixel_samples=256, accumulation_chunk_size=2, eval_every=1, checkpoint_every=1, evaluation_grid_shape=(3, 3, 3), resume_from=None, smoke=True)


def _b1c_smoke_args(tmp_path: Path, name: str, source_checkpoint: Path) -> Namespace:
    root = Path(__file__).resolve().parents[3]
    return Namespace(source_config=root / "configs/source_first_change4_paperaligned.yaml", experiment_config=root / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml", manifest=None, qc_table=None, canonical_domain=None, source_checkpoint=source_checkpoint, output_dir=tmp_path / name, device="cpu", overwrite=False, seed=0, locations_per_view=1, frames_per_location=3, phase_c_steps=2, pixel_mode="fixed_grid", pixel_samples=256, accumulation_chunk_size=2, film_lr=5e-5, respiratory_mbc_lr=5e-5, eval_every=1, checkpoint_every=1, evaluation_grid_shape=(3, 3, 3), resume_from=None, smoke=True)


def _make_smoke_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(b1b, "_escape", lambda _metric: True)
    source = _b1b_smoke_args(tmp_path, "source")
    b1b.run(source)
    return source.output_dir / "checkpoint_phase_b_2.pt"


def test_source_gate_failure_writes_summary_without_joint_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _make_smoke_source(tmp_path, monkeypatch)
    monkeypatch.setattr(b1b, "_escape", lambda _metric: False)
    summary = b1c.run(_b1c_smoke_args(tmp_path, "invalid", source))
    assert summary["causal_classification"] == "invalid_good_staged_source"
    assert summary["phase_c_entered"] is False
    assert not (tmp_path / "invalid" / "checkpoint_c_1.pt").exists()


def test_source_checkpoint_metric_mismatch_is_rejected_before_c_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _make_smoke_source(tmp_path, monkeypatch)
    payload = b1c.old_b1._load_checkpoint(source)
    payload["phase_metrics"]["B2"]["aggregate"]["full_fov_dynamic_cosine"]["mean"] += .1
    torch.save(payload, source)
    with pytest.raises(ValueError, match="source_checkpoint_metric_mismatch"):
        b1c.run(_b1c_smoke_args(tmp_path, "mismatch", source))


def test_joint_c_uses_fresh_phase_c_optimizer_and_writes_cadence_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _make_smoke_source(tmp_path, monkeypatch)
    monkeypatch.setattr(b1b, "_escape", lambda _metric: True)
    calls: list[tuple[str, float, float]] = []; losses: list[tuple[str, object]] = []
    configure = b1c.old_b1._configure_phase
    monkeypatch.setattr(b1c.old_b1, "_configure_phase", lambda model, phase, *, film_lr, respiratory_mbc_lr: calls.append((phase, film_lr, respiratory_mbc_lr)) or configure(model, phase, film_lr=film_lr, respiratory_mbc_lr=respiratory_mbc_lr))
    accumulate = b1c.old_b1._accumulate_full_batch
    monkeypatch.setattr(b1c.old_b1, "_accumulate_full_batch", lambda model, observations, *, phase, motion, pixel_samples, accumulation_chunk_size: losses.append((phase, motion)) or accumulate(model, observations, phase=phase, motion=motion, pixel_samples=pixel_samples, accumulation_chunk_size=accumulation_chunk_size))
    summary = b1c.run(_b1c_smoke_args(tmp_path, "joint", source))
    output = tmp_path / "joint"
    assert calls == [("C", 5e-5, 5e-5)]
    assert losses == [("C", None), ("C", None)]
    assert summary["phase_c_entered"] is True
    assert summary["fresh_optimizer_state_empty"] is True
    assert summary["gradient_connectivity_audit"] is True
    assert summary["stage_parameter_change_audit"]["C"]["unexpected"] == []
    assert summary["runtime_audit"]["cardiac_mbc_calls"] == 0
    assert summary["runtime_audit"]["uncertainty_calls"] == 0
    assert all(summary["frozen_module_audit"].values())
    assert summary["trajectory_flags"]["0"]["escape"] is True
    assert {"delta_full_fov_dynamic_cosine", "delta_full_fov_dynamic_rms_ratio", "delta_full_fov_dynamic_normalized_epe", "delta_full_fov_static_rms_mm"} <= summary.keys()
    assert summary["exposure_accounting"]["phase_c_incremental"]["observation_exposures"] == 900
    assert (output / "metrics_c_0.json").is_file()
    assert (output / "checkpoint_c_1.pt").is_file()
    assert (output / "checkpoint_c_2.pt").is_file()
    assert (output / "source_checkpoint_provenance.json").is_file()
    provenance = json.loads((output / "source_checkpoint_provenance.json").read_text())
    assert provenance["source_checkpoint_path"] == str(source)
    assert provenance["source_checkpoint_sha256"]


def test_mid_c_resume_restores_c_optimizer_rng_and_reaudits_gradients(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _make_smoke_source(tmp_path, monkeypatch)
    monkeypatch.setattr(b1b, "_escape", lambda _metric: True)
    first = _b1c_smoke_args(tmp_path, "resume", source)
    first.phase_c_steps = 1
    b1c.run(first)
    resumed = _b1c_smoke_args(tmp_path, "resume", source)
    resumed.resume_from = first.output_dir / "checkpoint_c_1.pt"
    calls: list[str] = []; original = b1c.old_b1._assert_expected_gradients
    monkeypatch.setattr(b1c.old_b1, "_assert_expected_gradients", lambda model, phase: calls.append(phase) or original(model, phase))
    summary = b1c.run(resumed)
    assert calls == ["C"]
    assert summary["phase_c_entered"] is True
    payload = b1c.old_b1._load_checkpoint(resumed.output_dir / "checkpoint_c_2.pt")
    assert payload["phase"] == "C"
    assert payload["optimizer"]["state"]
