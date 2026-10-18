from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cardioresp4d.models.cardioresp_motion import ScoreWeightedMBCField
from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, fallback_geometry
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import _OracleRespiratoryMBC, _gt_respiratory_scores, _respiratory_only_motion
from experiments.c4_diagnostics.exp2c4c_oracle_score_loss_landscape import _aggregate, _build_model, _evaluate_frame, run
from experiments.c4_diagnostics.exp2_checkpoint_common import parameter_snapshot
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom


def _domain() -> dict[str, object]:
    return {"world_min_mm": [-4.0, -4.0, -4.0], "world_max_mm": [4.0, 4.0, 4.0], "cardiac_box": {"min_mm": [-2.0, -2.0, -2.0], "max_mm": [2.0, 2.0, 2.0]}}


def _config() -> dict[str, object]:
    return yaml.safe_load((Path(__file__).resolve().parents[3] / "configs/source_first_change4_paperaligned.yaml").read_text())


def _fixture() -> tuple[object, GroundTruthMotion, object]:
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(_domain()))
    frame = fallback_geometry(locations_per_view=1, frames_per_location=1, image_shape=(8, 8), motion=motion, device=torch.device("cpu"))[0]
    observation = _build_observations([frame], image_shape=(8, 8), phantom=AnalyticCanonicalPhantom(_domain()), motion=motion, psf_samples=1, seed=0)[0]
    return _build_model(_config(), _domain(), n_dynamic_frames=1, seed=0), motion, observation


def test_oracle_times_gt_coefficient_matches_analytic_respiratory_dvf() -> None:
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(_domain()))
    points = torch.randn(3, 7, 3); times = torch.tensor([0.0, .37, .81])
    field = ScoreWeightedMBCField(_OracleRespiratoryMBC(motion), _gt_respiratory_scores(motion, times))
    torch.testing.assert_close(field(points), motion.respiratory_dvf_mm(points, times), atol=1e-6, rtol=1e-6)


def test_exact_gt_evaluation_is_independent_of_grid_discretization() -> None:
    model, motion, observation = _fixture(); oracle = _OracleRespiratoryMBC(motion)
    first, _ = _evaluate_frame(model, observation, oracle, motion, np.array([-9.0, 0.0, 9.0]), finite_difference_delta_mm=.05, seed=0)
    second, _ = _evaluate_frame(model, observation, oracle, motion, np.arange(-8.9, 9.0, .2), finite_difference_delta_mm=.05, seed=0)
    assert first["mse_at_gt"] == pytest.approx(second["mse_at_gt"], abs=1e-10)


def test_selected_frame_losses_are_finite_and_leave_model_unchanged() -> None:
    model, motion, observation = _fixture(); before = parameter_snapshot(model)
    row, curve = _evaluate_frame(model, observation, _OracleRespiratoryMBC(motion), motion, np.array([-1.0, 0.0, 1.0]), finite_difference_delta_mm=.05, seed=0)
    assert np.isfinite(curve).all() and all(np.isfinite(value) for value in row.values() if isinstance(value, float))
    assert all(torch.equal(value, model.state_dict()[name].detach().cpu()) for name, value in before.items())


def test_aggregation_regression_metrics_are_correct_for_artificial_fixture() -> None:
    rows = [{"view": "SAX", "slice_id": "a", "gt_coefficient_mm": -2.0, "argmin_coefficient_mm": -1.0, "mse_at_gt": 2.0, "mse_at_zero": 4.0, "argmin_mse": 1.0, "dL_da_at_gt": .2, "d2L_da2_at_gt": 1.0, "loss_dynamic_range": 1.0}, {"view": "SAX", "slice_id": "a", "gt_coefficient_mm": 2.0, "argmin_coefficient_mm": 3.0, "mse_at_gt": 2.0, "mse_at_zero": 4.0, "argmin_mse": 1.0, "dL_da_at_gt": -.2, "d2L_da2_at_gt": 1.0, "loss_dynamic_range": 1.0}]
    values = _aggregate(rows)
    assert values["mae_argmin_vs_gt_mm"] == pytest.approx(1.0)
    assert values["slope"] == pytest.approx(1.0)
    assert values["intercept_mm"] == pytest.approx(1.0)
    assert values["fraction_curvature_gt_positive"] == pytest.approx(1.0)


def test_smoke_writes_outputs_and_never_calls_encoder_or_learned_mbc(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    args = Namespace(source_config=root / "configs/source_first_change4_paperaligned.yaml", experiment_config=root / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml", manifest=None, qc_table=None, canonical_domain=None, output_dir=tmp_path / "c4c", device="cpu", overwrite=False, seed=0, locations_per_view=1, frames_per_location=1, coefficient_min_mm=-1., coefficient_max_mm=1., coefficient_step_mm=1., finite_difference_delta_mm=.05, smoke=True)
    summary = run(args)
    assert summary["encoder_calls"] == 0 and summary["learned_respiratory_mbc_calls"] == 0 and summary["parameters_unchanged"]
    for name in ("summary.json", "per_frame_landscape.csv", "aggregate_by_location.csv", "aggregate_by_view.csv", "landscape_arrays.npz"):
        assert (args.output_dir / name).is_file()
