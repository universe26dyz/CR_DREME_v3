from __future__ import annotations

import copy
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, fallback_geometry
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import _build_intervention_model, _respiratory_only_motion
from experiments.c4_diagnostics.exp3b1_staged_bootstrap_recovery import _accumulate_full_batch, _classify_stages, _configure_phase, _fixed_pixel_grid, _load_checkpoint, run
import experiments.c4_diagnostics.exp3b1_staged_bootstrap_recovery as b1
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom


def _domain() -> dict[str, object]:
    return {"world_min_mm": [-4.0, -4.0, -4.0], "world_max_mm": [4.0, 4.0, 4.0], "cardiac_box": {"min_mm": [-2.0, -2.0, -2.0], "max_mm": [2.0, 2.0, 2.0]}}


def _config() -> dict[str, object]:
    return yaml.safe_load((Path(__file__).resolve().parents[3] / "configs/source_first_change4_paperaligned.yaml").read_text())


def _observations() -> tuple[GroundTruthMotion, list]:
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(_domain()))
    frames = fallback_geometry(locations_per_view=1, frames_per_location=3, image_shape=(8, 8), motion=motion, device=torch.device("cpu"))
    return motion, _build_observations(frames, image_shape=(8, 8), phantom=AnalyticCanonicalPhantom(_domain()), motion=motion, psf_samples=1, seed=0)


def _model(n_observations: int):
    return _build_intervention_model(_config(), _domain(), n_dynamic_frames=n_observations, seed=0)


def test_phase_ownership_is_exact() -> None:
    model = _model(9)
    expected = {"A": ("respiratory_mbc.levels.0.",), "B": ("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp."), "C": ("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp.", "respiratory_mbc.levels.0.")}
    for phase, prefixes in expected.items():
        optimizer = _configure_phase(model, phase, film_lr=5e-4, respiratory_mbc_lr=5e-4)
        owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        wanted = {id(parameter) for name, parameter in model.named_parameters() if name.startswith(prefixes)}
        assert owned == wanted
        assert all(parameter.requires_grad == (id(parameter) in wanted) for parameter in model.parameters())
        assert len(optimizer.param_groups) == (2 if phase == "C" else 1)


def test_phase_b_full_batch_matches_direct_mean_and_never_uses_gt_score() -> None:
    _, observations = _observations(); accumulated = _model(len(observations))
    _configure_phase(accumulated, "B", film_lr=5e-4, respiratory_mbc_lr=5e-4); direct = copy.deepcopy(accumulated)
    with patch.object(b1, "_gt_respiratory_scores", side_effect=AssertionError("GT score entered phase B")):
        torch.manual_seed(123)
        batch = _accumulate_full_batch(accumulated, observations, phase="B", motion=None, pixel_samples=256, accumulation_chunk_size=2)
        torch.manual_seed(123)
        losses = [b1._prediction_loss_terms(direct, observation, _fixed_pixel_grid(observation, count=256))["data_mse"] for observation in observations]
        direct_loss = torch.stack(losses).mean(); direct_loss.backward()
    # PSF sampling is stochastic; the exact contract is accumulation weighting,
    # while this controlled smoke comparison bounds its rendering noise.
    torch.testing.assert_close(batch["full_batch_reconstruction_mse"], direct_loss.detach(), rtol=.25, atol=2e-4)
    assert batch["observation_ids"] == [observation.dynamic_frame_id for observation in observations]
    for (_, first), (_, second) in zip(accumulated.named_parameters(), direct.named_parameters()):
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, rtol=.3, atol=2e-4)
    assert all(parameter.grad is None for name, parameter in accumulated.named_parameters() if name.startswith("respiratory_mbc.levels.0."))


def test_causal_classification_uses_stage_dvf_not_score_gauge() -> None:
    fail = {"full_fov_dynamic_cosine": .7, "full_fov_dynamic_rms_ratio": .1, "full_fov_dynamic_normalized_epe": 2.}
    escape = {"full_fov_dynamic_cosine": .91, "full_fov_dynamic_rms_ratio": 1., "full_fov_dynamic_normalized_epe": .7, "score_mse": 999.}
    strict = {"full_fov_dynamic_cosine": .96, "full_fov_dynamic_rms_ratio": 1., "full_fov_dynamic_normalized_epe": .4, "score_mse": 0.}
    assert _classify_stages(fail, escape, escape)["causal_classification"] == "mbc_bootstrap_failure"
    assert _classify_stages(escape, fail, fail)["causal_classification"] == "encoder_given_learned_mbc_failure"
    assert _classify_stages(escape, escape, fail)["causal_classification"] == "joint_unfreeze_regression"
    assert _classify_stages(escape, escape, strict)["causal_classification"] == "staged_bootstrap_strict_recovery"


def test_cpu_smoke_writes_phase_boundaries_and_resumes_across_a_to_b(tmp_path: Path) -> None:
    root, output_dir = Path(__file__).resolve().parents[3], tmp_path / "b1"
    common = {"source_config": root / "configs/source_first_change4_paperaligned.yaml", "experiment_config": root / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml", "manifest": None, "qc_table": None, "canonical_domain": None, "output_dir": output_dir, "device": "cpu", "overwrite": False, "seed": 0, "locations_per_view": 1, "frames_per_location": 3, "phase_a_steps": 1, "phase_b_steps": 1, "phase_c_steps": 1, "eval_every": 1, "pixel_mode": "fixed_grid", "pixel_samples": 256, "accumulation_chunk_size": 2, "checkpoint_every": 1, "evaluation_grid_shape": (3, 3, 3), "smoke": True}
    summary = run(Namespace(**common, resume_from=None))
    assert summary["renderer_calls"]["total"] > 0 and all(summary["mbc_calls"][phase] > 0 for phase in ("A", "B", "C"))
    assert summary["stage_parameter_change_audit"]["B"]["unexpected"] == []
    for name in ("metrics_phase_a_end.json", "metrics_phase_b_end.json", "stage_transition_audit.json", "factor_diagnostics.json", "checkpoint_phase_a_end.pt", "checkpoint_phase_b_end.pt", "checkpoint_last.pt"):
        assert (output_dir / name).is_file()
    resumed = run(Namespace(**{**common, "resume_from": output_dir / "checkpoint_phase_a_end.pt"}))
    assert resumed["phase_schedule"]["phase_b_steps"] == 1


def test_checkpoint_loader_forces_cpu_map_location_for_rng_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"; torch.save({"rng_state": {"torch": torch.get_rng_state()}}, checkpoint)
    observed: dict[str, object] = {}; original = torch.load
    def tracked(*args, **kwargs):
        observed.update(kwargs); return original(*args, **kwargs)
    with patch.object(torch, "load", side_effect=tracked):
        payload = _load_checkpoint(checkpoint)
    assert observed["map_location"] == "cpu" and payload["rng_state"]["torch"].device.type == "cpu"


def test_eval_cadence_and_phase_c_do_not_use_gt_score(tmp_path: Path) -> None:
    root, output_dir = Path(__file__).resolve().parents[3], tmp_path / "cadence"
    common = {"source_config": root / "configs/source_first_change4_paperaligned.yaml", "experiment_config": root / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml", "manifest": None, "qc_table": None, "canonical_domain": None, "output_dir": output_dir, "device": "cpu", "overwrite": False, "seed": 0, "locations_per_view": 1, "frames_per_location": 3, "phase_a_steps": 1, "phase_b_steps": 1, "phase_c_steps": 2, "eval_every": 9, "pixel_mode": "fixed_grid", "pixel_samples": 256, "accumulation_chunk_size": 2, "checkpoint_every": 1, "evaluation_grid_shape": (3, 3, 3), "smoke": True}
    original = b1._gt_respiratory_scores
    calls = {"count": 0}
    def guarded(*args, **kwargs):
        calls["count"] += 1; return original(*args, **kwargs)
    with patch.object(b1, "_gt_respiratory_scores", side_effect=guarded):
        run(Namespace(**common, resume_from=None))
    with (output_dir / "training_curve.csv").open() as handle:
        rows = list(__import__("csv").DictReader(handle))
    assert [row["global_step"] for row in rows] == ["1", "2", "4"]
