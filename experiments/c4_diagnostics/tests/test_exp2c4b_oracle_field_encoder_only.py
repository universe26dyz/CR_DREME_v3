from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import torch
import yaml

from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, fallback_geometry
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import (
    _OracleRespiratoryMBC,
    _assert_only_encoder_changed,
    _build_intervention_model,
    _configure_encoder_only,
    _dvf_values,
    _gt_respiratory_scores,
    _intervention_loss_terms,
    _respiratory_only_motion,
    run,
)
from experiments.c4_diagnostics.exp2_checkpoint_common import parameter_snapshot
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField


def _domain() -> dict[str, object]:
    return {"world_min_mm": [-4.0, -4.0, -4.0], "world_max_mm": [4.0, 4.0, 4.0], "cardiac_box": {"min_mm": [-2.0, -2.0, -2.0], "max_mm": [2.0, 2.0, 2.0]}}


def _config() -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    return yaml.safe_load((root / "configs/source_first_change4_paperaligned.yaml").read_text())


def _observations() -> tuple[GroundTruthMotion, list]:
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(_domain()))
    frames = fallback_geometry(locations_per_view=1, frames_per_location=3, image_shape=(8, 8), motion=motion, device=torch.device("cpu"))
    return motion, _build_observations(frames, image_shape=(8, 8), phantom=AnalyticCanonicalPhantom(_domain()), motion=motion, psf_samples=1, seed=0)


def test_oracle_basis_times_gt_score_matches_analytic_respiratory_dvf() -> None:
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(_domain()))
    points = torch.randn(3, 7, 3)
    timestamps = torch.tensor([0.0, 0.37, 0.81])
    field = ScoreWeightedMBCField(_OracleRespiratoryMBC(motion), _gt_respiratory_scores(motion, timestamps))
    torch.testing.assert_close(field(points), motion.respiratory_dvf_mm(points, timestamps), atol=1e-6, rtol=1e-6)


def test_respiratory_only_target_has_exactly_zero_cardiac_dvf() -> None:
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(_domain()))
    points = torch.randn(2, 5, 3)
    assert motion.cardiac_amplitude_mm == 0.0
    assert torch.equal(motion.cardiac_dvf_mm(points, torch.tensor([0.1, 0.2])), torch.zeros_like(points))


def test_only_intended_encoder_path_is_trainable_and_owned_by_optimizer() -> None:
    model = _build_intervention_model(_config(), _domain(), n_dynamic_frames=3, seed=0)
    optimizer = _configure_encoder_only(model, learning_rate=5e-4)
    owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    allowed = {id(parameter) for name, parameter in model.named_parameters() if name.startswith(("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp."))}
    assert owned == allowed
    assert all(parameter.requires_grad == (id(parameter) in allowed) for parameter in model.parameters())


def test_data_loss_reaches_only_encoder_path_and_frozen_modules_remain_unchanged() -> None:
    motion, observations = _observations()
    model = _build_intervention_model(_config(), _domain(), n_dynamic_frames=len(observations), seed=0)
    _configure_encoder_only(model, learning_rate=5e-4)
    before = parameter_snapshot(model)
    terms = _intervention_loss_terms(model, observations[0], _OracleRespiratoryMBC(motion), pixel_uv=torch.tensor([[0.0, 0.0], [3.0, 4.0]]))
    terms["data_mse"].backward()
    for prefix in ("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp."):
        assert any(name.startswith(prefix) and parameter.grad is not None and torch.isfinite(parameter.grad).all() and float(parameter.grad.norm()) > 0.0 for name, parameter in model.named_parameters())
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if not name.startswith(("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp.")))
    _assert_only_encoder_changed(model, before, require_change=False)


def test_dynamic_normalized_epe_ignores_constant_spatial_offset() -> None:
    target = torch.tensor([[[1.0, -2.0, 0.5]], [[2.0, -1.0, 1.5]]])
    values = _dvf_values(target + torch.tensor([5.0, -3.0, 2.0]), target)
    assert values["endpoint_error_mm"] > 0.0
    assert values["dynamic_normalized_epe"] == pytest.approx(0.0, abs=1e-7)


def test_fresh_initialization_is_deterministic() -> None:
    first = _build_intervention_model(_config(), _domain(), n_dynamic_frames=3, seed=11)
    second = _build_intervention_model(_config(), _domain(), n_dynamic_frames=3, seed=11)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name])


def test_cpu_smoke_writes_required_outputs_and_changes_only_encoder(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    args = Namespace(source_config=root / "configs/source_first_change4_paperaligned.yaml", experiment_config=root / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml", manifest=None, qc_table=None, canonical_domain=None, output_dir=tmp_path / "c4b", device="cpu", overwrite=False, init="fresh", steps=1, seed=0, locations_per_view=1, frames_per_location=3, observations_per_update=3, pixel_samples=4, evaluation_grid_shape=(3, 3, 3), smoke=True)
    summary = run(args)
    assert summary["parameters_changed"]
    assert all(name.startswith(("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp.")) for name in summary["parameters_changed"])
    assert all(summary["frozen_module_audit"].values())
    assert summary["learned_respiratory_mbc_calls"] == 0
    for name in ("summary.json", "training_curve.csv", "metrics_initial.json", "metrics_final.json", "recovery_arrays.npz"):
        assert (args.output_dir / name).is_file()
