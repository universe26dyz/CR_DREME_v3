from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import csv
import torch
import yaml

from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, fallback_geometry
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import _respiratory_only_motion
from experiments.c4_diagnostics.exp2c4e_single_location_score_overfit import _build_model, _classify, _configure_encoder_only, _direct_score_loss, _gt_score, _select_location, run
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom


def _domain() -> dict[str, object]:
    return {"world_min_mm": [-4.0, -4.0, -4.0], "world_max_mm": [4.0, 4.0, 4.0], "cardiac_box": {"min_mm": [-2.0, -2.0, -2.0], "max_mm": [2.0, 2.0, 2.0]}}


def _config() -> dict[str, object]:
    return yaml.safe_load((Path(__file__).resolve().parents[3] / "configs/source_first_change4_paperaligned.yaml").read_text())


def _observations():
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(_domain()))
    frames = fallback_geometry(locations_per_view=1, frames_per_location=3, image_shape=(8, 8), motion=motion, device=torch.device("cpu"))
    return motion, _build_observations(frames, image_shape=(8, 8), phantom=AnalyticCanonicalPhantom(_domain()), motion=motion, psf_samples=1, seed=0)


def test_selected_location_has_requested_frames_increasing_timestamps_and_constant_geometry() -> None:
    _, observations = _observations()
    sequence, audit = _select_location(observations, view="2CH", slice_id=None, frames_per_location=3)
    assert len(sequence) == 3 and all(item.view == "2CH" for item in sequence)
    assert all(second.timestamp_s > first.timestamp_s for first, second in zip(sequence, sequence[1:]))
    assert audit["geometry_constant"]


def test_gt_score_matches_c4d_definition() -> None:
    motion, _ = _observations()
    torch.testing.assert_close(_gt_score(motion, 1.0), torch.full((1, 1, 3), 6.0))


def test_optimizer_and_direct_loss_are_limited_to_intended_encoder_path() -> None:
    motion, observations = _observations(); model = _build_model(_config(), _domain(), n_dynamic_frames=len(observations), seed=0); optimizer = _configure_encoder_only(model, learning_rate=5e-4)
    owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert owned == {id(parameter) for name, parameter in model.named_parameters() if name.startswith(("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp."))}
    loss, _ = _direct_score_loss(model, observations[0], motion); loss.backward()
    for prefix in ("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp."):
        assert any(name.startswith(prefix) and parameter.grad is not None and float(parameter.grad.norm()) > 0 for name, parameter in model.named_parameters())
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if not name.startswith(("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp.")))


def test_classifier_thresholds() -> None:
    assert _classify({"gt_correlation": .99, "temporal_rms_ratio": .9, "score_mse": .01})["classification"] == "single_location_overfit_pass"
    assert _classify({"gt_correlation": .99, "temporal_rms_ratio": .5, "score_mse": .001})["classification"] == "single_location_overfit_fail"


def test_cpu_smoke_writes_outputs_curve_and_checkpoint_without_renderer_or_mbc(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    args = Namespace(source_config=root / "configs/source_first_change4_paperaligned.yaml", experiment_config=root / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml", manifest=None, qc_table=None, canonical_domain=None, output_dir=tmp_path / "c4e", device="cpu", overwrite=False, seed=0, view="2CH", slice_id=None, locations_per_view=1, frames_per_location=3, steps=2, film_lr=None, eval_every=1, smoke=True)
    summary = run(args)
    assert summary["renderer_calls"] == 0 and summary["learned_respiratory_mbc_calls"] == 0 and summary["parameters_changed"]
    assert all(summary["frozen_module_audit"].values())
    for name in ("summary.json", "metrics_initial.json", "metrics_final.json", "training_curve.csv", "recovery_arrays.npz", "checkpoint_last.pt"):
        assert (args.output_dir / name).is_file()
    with (args.output_dir / "training_curve.csv").open() as handle:
        assert {"step", "score_mse", "gt_correlation", "temporal_rms_ratio", "raw_prediction_rms"} <= set(csv.DictReader(handle).fieldnames or ())
