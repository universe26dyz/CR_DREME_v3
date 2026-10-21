from __future__ import annotations

import copy
from argparse import Namespace
from pathlib import Path

import torch
import yaml

from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, fallback_geometry
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import _build_intervention_model, _respiratory_only_motion
from experiments.c4_diagnostics.exp3a_joint_fullbatch_reconstruction_recovery import (
    _accumulate_full_batch,
    _classify_dvf,
    _configure_joint_training,
    _fixed_pixel_grid,
    _prediction_loss_terms,
    run,
)
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom


def _domain() -> dict[str, object]:
    return {"world_min_mm": [-4.0, -4.0, -4.0], "world_max_mm": [4.0, 4.0, 4.0], "cardiac_box": {"min_mm": [-2.0, -2.0, -2.0], "max_mm": [2.0, 2.0, 2.0]}}


def _config() -> dict[str, object]:
    return yaml.safe_load((Path(__file__).resolve().parents[3] / "configs/source_first_change4_paperaligned.yaml").read_text())


def _observations() -> list:
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(_domain()))
    frames = fallback_geometry(locations_per_view=1, frames_per_location=3, image_shape=(8, 8), motion=motion, device=torch.device("cpu"))
    return _build_observations(frames, image_shape=(8, 8), phantom=AnalyticCanonicalPhantom(_domain()), motion=motion, psf_samples=1, seed=0)


def _model(n_observations: int):
    return _build_intervention_model(_config(), _domain(), n_dynamic_frames=n_observations, seed=0)


def test_joint_optimizer_owns_only_respiratory_encoder_and_mbc_level0() -> None:
    model = _model(9)
    optimizer = _configure_joint_training(model, film_lr=5e-4, respiratory_mbc_lr=5e-4)
    owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected = {id(parameter) for name, parameter in model.named_parameters() if name.startswith(("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp.", "respiratory_mbc.levels.0."))}
    assert owned == expected
    assert all(parameter.requires_grad == (id(parameter) in expected) for parameter in model.parameters())
    assert model.respiratory_mbc.active_level_count == 1


def test_joint_full_batch_matches_direct_mean_and_reaches_both_factors() -> None:
    observations = _observations()
    accumulated, direct = _model(len(observations)), _model(len(observations))
    _configure_joint_training(accumulated, film_lr=5e-4, respiratory_mbc_lr=5e-4)
    _configure_joint_training(direct, film_lr=5e-4, respiratory_mbc_lr=5e-4)
    batch = _accumulate_full_batch(accumulated, observations, pixel_mode="fixed_grid", pixel_samples=256, accumulation_chunk_size=2)
    losses = [_prediction_loss_terms(direct, observation, _fixed_pixel_grid(observation, count=256))["data_mse"] for observation in observations]
    direct_loss = torch.stack(losses).mean(); direct_loss.backward()
    torch.testing.assert_close(batch["full_batch_reconstruction_mse"], direct_loss.detach())
    assert batch["observation_ids"] == [observation.dynamic_frame_id for observation in observations]
    for (_, first), (_, second) in zip(accumulated.named_parameters(), direct.named_parameters()):
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, rtol=1e-2, atol=1e-4)
    assert any(parameter.grad is not None and float(parameter.grad.norm()) > 0 for name, parameter in accumulated.named_parameters() if name.startswith("film_encoder.resp."))
    assert any(parameter.grad is not None and float(parameter.grad.norm()) > 0 for name, parameter in accumulated.named_parameters() if name.startswith("respiratory_mbc.levels.0."))
    assert all(parameter.grad is None for name, parameter in accumulated.named_parameters() if not name.startswith(("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp.", "respiratory_mbc.levels.0.")))


def test_fixed_grid_is_deterministic_unique_and_covers_full_fov() -> None:
    observation = _observations()[0]
    first, second = _fixed_pixel_grid(observation, count=256), _fixed_pixel_grid(observation, count=256)
    torch.testing.assert_close(first, second)
    assert first.shape == (64, 2) and torch.unique(first, dim=0).shape[0] == 64
    assert int(first[:, 0].min()) == 0 and int(first[:, 0].max()) == 7
    assert int(first[:, 1].min()) == 0 and int(first[:, 1].max()) == 7


def test_dvf_classification_is_independent_of_score_gauge_diagnostics() -> None:
    recovered = {"full_fov_dynamic_cosine": 0.96, "full_fov_dynamic_rms_ratio": 1.0, "full_fov_dynamic_normalized_epe": 0.4, "score_mse": 10_000.0, "temporal_rms_ratio": 100.0}
    rescaled = {**recovered, "score_mse": 0.0, "temporal_rms_ratio": 0.01}
    assert _classify_dvf(recovered) == _classify_dvf(rescaled)
    assert _classify_dvf(recovered)["classification"] == "joint_fullbatch_coldstart_escape"
    assert _classify_dvf(recovered)["strict_recovery"]


def test_cpu_smoke_uses_production_renderer_learned_mbc_and_resumes(tmp_path: Path) -> None:
    root, output_dir = Path(__file__).resolve().parents[3], tmp_path / "exp3a"
    common = {"source_config": root / "configs/source_first_change4_paperaligned.yaml", "experiment_config": root / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml", "manifest": None, "qc_table": None, "canonical_domain": None, "output_dir": output_dir, "device": "cpu", "overwrite": False, "seed": 0, "locations_per_view": 1, "frames_per_location": 3, "steps": 2, "eval_every": 1, "pixel_mode": "fixed_grid", "pixel_samples": 256, "accumulation_chunk_size": 2, "checkpoint_every": 1, "evaluation_grid_shape": (3, 3, 3), "smoke": True}
    summary = run(Namespace(**common, resume_from=None))
    assert summary["renderer_calls"] > 0 and summary["training_renderer_calls"] > 0 and summary["learned_respiratory_mbc_calls"] > 0
    assert summary["cardiac_mbc_calls"] == 0 and summary["parameters_changed"]
    assert all(name.startswith(("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp.", "respiratory_mbc.levels.0.")) for name in summary["parameters_changed"])
    for name in ("summary.json", "metrics_initial.json", "metrics_final.json", "training_curve.csv", "recovery_arrays.npz", "checkpoint_last.pt", "checkpoint_1.pt", "checkpoint_2.pt"):
        assert (output_dir / name).is_file()
    checkpoint = torch.load(output_dir / "checkpoint_last.pt", map_location="cpu", weights_only=True)
    assert checkpoint["step"] == 2 and checkpoint["protocol"]["active_respiratory_levels"] == 1
    resumed = run(Namespace(**{**common, "steps": 3, "resume_from": output_dir / "checkpoint_last.pt"}))
    assert resumed["exposure_accounting"]["optimizer_steps"] == 3
