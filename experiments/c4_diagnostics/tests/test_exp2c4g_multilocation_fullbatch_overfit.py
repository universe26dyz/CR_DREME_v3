from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import csv
import inspect

import torch
import yaml

from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, fallback_geometry
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import _respiratory_only_motion
from experiments.c4_diagnostics.exp2c4d_supervised_score_encoder_probe import _build_model, _configure_encoder_only, _direct_score_loss
import experiments.c4_diagnostics.exp2c4g_multilocation_fullbatch_overfit as c4g
from experiments.c4_diagnostics.exp2c4g_multilocation_fullbatch_overfit import _accumulate_full_batch, _classify, _exposure_accounting, _observation_audit, _update_thresholds, run
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom


def _domain() -> dict[str, object]:
    return {"world_min_mm": [-4.0, -4.0, -4.0], "world_max_mm": [4.0, 4.0, 4.0], "cardiac_box": {"min_mm": [-2.0, -2.0, -2.0], "max_mm": [2.0, 2.0, 2.0]}}


def _config() -> dict[str, object]:
    return yaml.safe_load((Path(__file__).resolve().parents[3] / "configs/source_first_change4_paperaligned.yaml").read_text())


def _observations(locations_per_view: int = 1, frames_per_location: int = 3):
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(_domain()))
    frames = fallback_geometry(locations_per_view=locations_per_view, frames_per_location=frames_per_location, image_shape=(8, 8), motion=motion, device=torch.device("cpu"))
    return motion, _build_observations(frames, image_shape=(8, 8), phantom=AnalyticCanonicalPhantom(_domain()), motion=motion, psf_samples=1, seed=0)


def test_observation_audit_accepts_formal_nine_location_450_frame_set() -> None:
    _, observations = _observations(locations_per_view=3, frames_per_location=50)
    audit = _observation_audit(observations)
    assert audit["n_locations"] == 9 and audit["n_frames"] == 450
    assert audit["frames_per_location"] == 50 and audit["timestamps_strictly_increasing"]


def test_accumulation_uses_each_observation_once_and_matches_direct_batch_mean_gradient() -> None:
    motion, observations = _observations()
    accumulated = _build_model(_config(), _domain(), n_dynamic_frames=len(observations), seed=0)
    direct = _build_model(_config(), _domain(), n_dynamic_frames=len(observations), seed=0)
    _configure_encoder_only(accumulated, learning_rate=5e-4); _configure_encoder_only(direct, learning_rate=5e-4)
    result = _accumulate_full_batch(accumulated, observations, motion, accumulation_chunk_size=2)
    direct_loss = torch.stack([_direct_score_loss(direct, observation, motion)[0] for observation in observations]).mean(); direct_loss.backward()
    assert result["observation_ids"] == [observation.dynamic_frame_id for observation in observations]
    torch.testing.assert_close(result["full_batch_score_mse"], direct_loss.detach())
    for (name_a, parameter_a), (name_b, parameter_b) in zip(accumulated.named_parameters(), direct.named_parameters()):
        assert name_a == name_b
        if parameter_a.grad is not None:
            torch.testing.assert_close(parameter_a.grad, parameter_b.grad, rtol=1e-5, atol=1e-7)


def test_thresholds_classifier_and_exposure_accounting() -> None:
    metrics = {"gt_correlation": .96, "temporal_rms_ratio": 1.0, "score_mse": 1.0}
    assert _classify(metrics)["classification"] == "multilocation_fullbatch_recovery"
    thresholds = _update_thresholds(None, metrics, step=7)
    assert thresholds["first_corr_ge_0p90_step"] == 7 and thresholds["first_joint_near_recovery_step"] == 7
    exposure = _exposure_accounting(steps=10, n_observations=450)
    assert exposure["total_observation_exposures"] == 4500 and exposure["per_frame_exposures"] == 10
    assert "ViewLocationBalancedSampler" not in inspect.getsource(c4g)


def test_cpu_smoke_writes_outputs_checkpoint_and_uses_no_renderer_or_mbc(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    args = Namespace(source_config=root / "configs/source_first_change4_paperaligned.yaml", experiment_config=root / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml", manifest=None, qc_table=None, canonical_domain=None, output_dir=tmp_path / "c4g", device="cpu", overwrite=False, seed=0, locations_per_view=1, frames_per_location=3, steps=2, eval_every=1, accumulation_chunk_size=2, smoke=True)
    summary = run(args)
    assert summary["renderer_calls"] == 0 and summary["learned_respiratory_mbc_calls"] == 0 and summary["parameters_changed"]
    assert summary["exposure_accounting"]["observations_per_optimizer_step"] == 9
    assert all(summary["frozen_module_audit"].values())
    for name in ("summary.json", "metrics_initial.json", "metrics_final.json", "training_curve.csv", "recovery_arrays.npz", "checkpoint_last.pt"):
        assert (args.output_dir / name).is_file()
    checkpoint = torch.load(args.output_dir / "checkpoint_last.pt", map_location="cpu", weights_only=True)
    assert checkpoint["step"] == 2 and checkpoint["protocol"]["exact_full_batch"]
    with (args.output_dir / "training_curve.csv").open() as handle:
        assert {"step", "full_batch_score_mse", "gt_correlation", "temporal_rms_ratio"} <= set(csv.DictReader(handle).fieldnames or ())
