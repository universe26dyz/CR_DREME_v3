from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import inspect
import copy

import torch
import yaml

from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, fallback_geometry
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import _OracleRespiratoryMBC, _assert_oracle_matches_gt, _build_intervention_model, _configure_encoder_only, _intervention_loss_terms, _respiratory_only_motion
from experiments.c4_diagnostics.exp2c4h_oracle_field_fullbatch_reconstruction import _accumulate_full_batch, _exposure_accounting, _fixed_pixel_grid, _observation_audit, run
import experiments.c4_diagnostics.exp2c4h_oracle_field_fullbatch_reconstruction as c4h
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from cardioresp4d.training.model import SourceFirstDynamicModel


def _domain() -> dict[str, object]:
    return {"world_min_mm": [-4.0, -4.0, -4.0], "world_max_mm": [4.0, 4.0, 4.0], "cardiac_box": {"min_mm": [-2.0, -2.0, -2.0], "max_mm": [2.0, 2.0, 2.0]}}


def _config() -> dict[str, object]:
    return yaml.safe_load((Path(__file__).resolve().parents[3] / "configs/source_first_change4_paperaligned.yaml").read_text())


def _observations(locations_per_view: int = 1, frames_per_location: int = 3):
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(_domain()))
    frames = fallback_geometry(locations_per_view=locations_per_view, frames_per_location=frames_per_location, image_shape=(8, 8), motion=motion, device=torch.device("cpu"))
    return motion, _build_observations(frames, image_shape=(8, 8), phantom=AnalyticCanonicalPhantom(_domain()), motion=motion, psf_samples=1, seed=0)


def test_fixed_grid_is_deterministic_unique_and_covers_the_full_fov() -> None:
    _, observations = _observations(); first, second = _fixed_pixel_grid(observations[0], count=256), _fixed_pixel_grid(observations[0], count=256)
    torch.testing.assert_close(first, second)
    assert first.shape == (64, 2) and torch.unique(first, dim=0).shape[0] == 64
    assert int(first[:, 0].min()) == 0 and int(first[:, 0].max()) == 7
    assert int(first[:, 1].min()) == 0 and int(first[:, 1].max()) == 7


def test_oracle_reconstruction_accumulation_matches_direct_mean_gradient() -> None:
    motion, observations = _observations(); oracle = _OracleRespiratoryMBC(motion)
    accumulated = _build_intervention_model(_config(), _domain(), n_dynamic_frames=len(observations), seed=0)
    direct = copy.deepcopy(accumulated)
    _configure_encoder_only(accumulated, learning_rate=5e-4); _configure_encoder_only(direct, learning_rate=5e-4)
    result = _accumulate_full_batch(accumulated, observations, oracle, pixel_mode="fixed_grid", pixel_samples=256, accumulation_chunk_size=2)
    direct_loss = torch.stack([_intervention_loss_terms(direct, observation, oracle, pixel_uv=_fixed_pixel_grid(observation, count=256))["data_mse"] for observation in observations]).mean(); direct_loss.backward()
    torch.testing.assert_close(result["full_batch_reconstruction_mse"], direct_loss.detach())
    assert result["observation_ids"] == [observation.dynamic_frame_id for observation in observations]
    for (_, parameter_a), (_, parameter_b) in zip(accumulated.named_parameters(), direct.named_parameters()):
        if parameter_a.grad is not None:
            torch.testing.assert_close(parameter_a.grad, parameter_b.grad, rtol=1e-2, atol=1e-4)


def test_formal_observation_audit_oracle_and_exposure_accounting() -> None:
    motion, observations = _observations(locations_per_view=3, frames_per_location=50)
    audit = _observation_audit(observations)
    assert audit["n_locations"] == 9 and audit["n_frames"] == 450 and audit["frames_per_location"] == 50
    model = _build_intervention_model(_config(), _domain(), n_dynamic_frames=len(observations), seed=0)
    grid, _ = SourceFirstDynamicModel._physical_grid(model.canonical_lower_world_mm, model.canonical_upper_world_mm, (3, 3, 3))
    _assert_oracle_matches_gt(_OracleRespiratoryMBC(motion), motion, grid)
    exposure = _exposure_accounting(steps=10, n_observations=450, pixels_per_observation=256)
    assert exposure["pixel_observation_exposures"] == 10 * 450 * 256
    assert "_sample_pixels" not in inspect.getsource(c4h)


def test_cpu_smoke_uses_renderer_not_learned_mbc_and_resumes(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]; output_dir = tmp_path / "c4h"
    common = {"source_config": root / "configs/source_first_change4_paperaligned.yaml", "experiment_config": root / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml", "manifest": None, "qc_table": None, "canonical_domain": None, "output_dir": output_dir, "device": "cpu", "overwrite": False, "seed": 0, "locations_per_view": 1, "frames_per_location": 3, "steps": 2, "eval_every": 1, "pixel_mode": "fixed_grid", "pixel_samples": 256, "accumulation_chunk_size": 2, "checkpoint_every": 1, "evaluation_grid_shape": (3, 3, 3), "smoke": True}
    summary = run(Namespace(**common, resume_from=None))
    assert summary["renderer_calls"] > 0 and summary["training_renderer_calls"] > 0 and summary["learned_respiratory_mbc_calls"] == 0 and summary["parameters_changed"]
    for name in ("summary.json", "metrics_initial.json", "metrics_final.json", "training_curve.csv", "recovery_arrays.npz", "checkpoint_last.pt"):
        assert (output_dir / name).is_file()
    checkpoint = torch.load(output_dir / "checkpoint_last.pt", map_location="cpu", weights_only=True)
    assert checkpoint["step"] == 2 and "optimizer" in checkpoint and checkpoint["pixel_protocol"]["pixel_mode"] == "fixed_grid"
    resumed = run(Namespace(**{**common, "steps": 3, "resume_from": output_dir / "checkpoint_last.pt"}))
    assert resumed["exposure_accounting"]["optimizer_steps"] == 3
