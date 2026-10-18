from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch
import yaml
import numpy as np
import pytest

from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, fallback_geometry
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import _respiratory_only_motion, _score_values
from experiments.c4_diagnostics.exp2c4d_supervised_score_encoder_probe import _build_model, _configure_encoder_only, _direct_score_loss, _gt_score, _parameter_audit, run
from experiments.c4_diagnostics.exp2_checkpoint_common import parameter_snapshot
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom


def _domain() -> dict[str, object]:
    return {"world_min_mm": [-4.0, -4.0, -4.0], "world_max_mm": [4.0, 4.0, 4.0], "cardiac_box": {"min_mm": [-2.0, -2.0, -2.0], "max_mm": [2.0, 2.0, 2.0]}}


def _config() -> dict[str, object]:
    return yaml.safe_load((Path(__file__).resolve().parents[3] / "configs/source_first_change4_paperaligned.yaml").read_text())


def _observation():
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(_domain()))
    frame = fallback_geometry(locations_per_view=1, frames_per_location=1, image_shape=(8, 8), motion=motion, device=torch.device("cpu"))[0]
    observation = _build_observations([frame], image_shape=(8, 8), phantom=AnalyticCanonicalPhantom(_domain()), motion=motion, psf_samples=1, seed=0)[0]
    return motion, observation


def test_gt_score_construction_is_exact() -> None:
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(_domain()))
    score = _gt_score(motion, 1.0)
    assert score.shape == (1, 1, 3)
    torch.testing.assert_close(score, torch.full((1, 1, 3), 6.0))


def test_only_intended_encoder_parameters_are_trainable_and_optimizer_owned() -> None:
    model = _build_model(_config(), _domain(), n_dynamic_frames=1, seed=0)
    optimizer = _configure_encoder_only(model, learning_rate=5e-4)
    owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    allowed = {id(parameter) for name, parameter in model.named_parameters() if name.startswith(("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp."))}
    assert owned == allowed
    assert all(parameter.requires_grad == (id(parameter) in allowed) for parameter in model.parameters())


def test_direct_score_loss_reaches_all_intended_encoder_submodules_and_preserves_frozen_parameters() -> None:
    motion, observation = _observation(); model = _build_model(_config(), _domain(), n_dynamic_frames=1, seed=0); _configure_encoder_only(model, learning_rate=5e-4); before = parameter_snapshot(model)
    loss, _ = _direct_score_loss(model, observation, motion); loss.backward()
    for prefix in ("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp."):
        assert any(name.startswith(prefix) and parameter.grad is not None and torch.isfinite(parameter.grad).all() and float(parameter.grad.norm()) > 0.0 for name, parameter in model.named_parameters())
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if not name.startswith(("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp.")))
    assert _parameter_audit(model, before, require_change=False) == []


def test_fresh_initialization_is_deterministic() -> None:
    first = _build_model(_config(), _domain(), n_dynamic_frames=3, seed=11); second = _build_model(_config(), _domain(), n_dynamic_frames=3, seed=11)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name])


def test_cpu_smoke_writes_outputs_and_only_changes_encoder(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    args = Namespace(source_config=root / "configs/source_first_change4_paperaligned.yaml", experiment_config=root / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml", manifest=None, qc_table=None, canonical_domain=None, output_dir=tmp_path / "c4d", device="cpu", overwrite=False, seed=0, locations_per_view=1, frames_per_location=3, steps=1, observations_per_update=3, smoke=True)
    summary = run(args)
    assert summary["parameters_changed"]
    assert all(name.startswith(("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp.")) for name in summary["parameters_changed"])
    assert all(summary["frozen_module_audit"].values())
    arrays = np.load(args.output_dir / "recovery_arrays.npz")
    predicted = torch.cat([torch.from_numpy(arrays[key]) for key in arrays.files if key.endswith("_predicted_scores") and not key.startswith("final_")])
    target = torch.cat([torch.from_numpy(arrays[key]) for key in arrays.files if key.endswith("_gt_scores") and not key.startswith("final_")])
    expected = _score_values(predicted, target)
    for name, value in expected.items():
        assert summary["initial_metrics"]["overall"][name] == pytest.approx(value)
    for name in ("summary.json", "training_curve.csv", "metrics_initial.json", "metrics_final.json", "recovery_arrays.npz"):
        assert (args.output_dir / name).is_file()
