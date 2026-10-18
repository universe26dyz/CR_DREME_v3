from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from cardioresp4d.training.sampler import DynamicObservation
from experiments.c4_diagnostics.exp2c4a_gt_score_mbc_only import (
    _assert_only_level0_changed,
    _build_intervention_model,
    _configure_level0_only,
    _gt_respiratory_scores,
    _intervention_loss_terms,
    _respiratory_only_motion,
    run,
)
from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, fallback_geometry
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom


def _domain() -> dict[str, object]:
    return {
        "world_min_mm": [-4.0, -4.0, -4.0],
        "world_max_mm": [4.0, 4.0, 4.0],
        "cardiac_box": {"min_mm": [-2.0, -2.0, -2.0], "max_mm": [2.0, 2.0, 2.0]},
    }


def _config() -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    return yaml.safe_load((root / "configs/source_first_change4_paperaligned.yaml").read_text())


def _observations() -> tuple[GroundTruthMotion, list[DynamicObservation]]:
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(_domain()))
    frames = fallback_geometry(locations_per_view=1, frames_per_location=3, image_shape=(8, 8), motion=motion, device=torch.device("cpu"))
    return motion, _build_observations(frames, image_shape=(8, 8), phantom=AnalyticCanonicalPhantom(_domain()), motion=motion, psf_samples=1, seed=0)


def test_gt_score_carries_amplitude_in_one_level_and_has_equal_xyz_components() -> None:
    motion = GroundTruthMotion.from_domain(_domain())
    timestamps = torch.tensor([0.0, 1.0])
    scores = _gt_respiratory_scores(motion, timestamps)
    assert scores.shape == (2, 1, 3)
    torch.testing.assert_close(scores[0], torch.zeros_like(scores[0]))
    torch.testing.assert_close(scores[1, 0], torch.full((3,), 6.0))


def test_respiratory_only_target_has_exactly_zero_cardiac_dvf() -> None:
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(_domain()))
    points = torch.randn(2, 5, 3)
    assert motion.cardiac_amplitude_mm == 0.0
    assert torch.equal(motion.cardiac_dvf_mm(points, torch.tensor([0.1, 0.2])), torch.zeros_like(points))


def test_only_level0_is_trainable_and_owned_by_optimizer() -> None:
    model = _build_intervention_model(_config(), _domain(), n_dynamic_frames=3, init_mode="fresh", checkpoint=None, seed=0)
    optimizer = _configure_level0_only(model, learning_rate=5e-4)
    owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    level0 = {id(parameter) for parameter in model.respiratory_mbc.levels[0].parameters()}
    assert owned == level0
    assert all(parameter.requires_grad == (id(parameter) in level0) for parameter in model.parameters())
    assert int(model.respiratory_mbc.active_level_count) == 1


def test_backward_reaches_only_level0_and_formal_norm_uses_active_level0() -> None:
    motion, observations = _observations()
    model = _build_intervention_model(_config(), _domain(), n_dynamic_frames=len(observations), init_mode="fresh", checkpoint=None, seed=0)
    _configure_level0_only(model, learning_rate=5e-4)
    terms = _intervention_loss_terms(model, observations[0], motion, pixel_uv=torch.tensor([[0.0, 0.0], [3.0, 4.0]]), objective="formal_mbc_norm", normalization_weight=.01)
    terms["total_loss"].backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() and float(parameter.grad.norm()) > 0.0 for parameter in model.respiratory_mbc.levels[0].parameters())
    assert all(parameter.grad is None for parameter in model.film_encoder.parameters())
    assert all(parameter.grad is None for level in model.respiratory_mbc.levels[1:] for parameter in level.parameters())
    assert all(parameter.grad is None for parameter in model.cardiac_mbc.parameters())
    assert terms["mbc_normalization"].requires_grad


def test_fresh_initialization_is_deterministic_and_strict_checkpoint_loads() -> None:
    first = _build_intervention_model(_config(), _domain(), n_dynamic_frames=3, init_mode="fresh", checkpoint=None, seed=11)
    second = _build_intervention_model(_config(), _domain(), n_dynamic_frames=3, init_mode="fresh", checkpoint=None, seed=11)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name])
    checkpoint = {name: value.detach().clone() for name, value in first.state_dict().items()}
    loaded = _build_intervention_model(_config(), _domain(), n_dynamic_frames=3, init_mode="checkpoint", checkpoint=checkpoint, seed=99)
    for name, value in checkpoint.items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


def test_cpu_smoke_writes_required_outputs_and_changes_only_level0(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    args = Namespace(
        source_config=root / "configs/source_first_change4_paperaligned.yaml",
        experiment_config=root / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml",
        manifest=None, qc_table=None, canonical_domain=None, checkpoint=None, output_dir=tmp_path / "c4a",
        device="cpu", overwrite=False, init="fresh", objective="data_only", steps=1, seed=0,
        locations_per_view=1, frames_per_location=3, observations_per_update=3, pixel_samples=4,
        evaluation_grid_shape=(3, 3, 3), smoke=True,
    )
    summary = run(args)
    assert summary["parameters_changed"]
    assert set(summary["parameters_changed"]) <= set(summary["trainable_parameter_names"])
    for name in ("summary.json", "training_curve.csv", "metrics_initial.json", "metrics_final.json", "recovery_arrays.npz"):
        assert (args.output_dir / name).is_file()
