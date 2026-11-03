from __future__ import annotations

from pathlib import Path
from argparse import Namespace

import torch
import yaml
import csv
import pytest

from experiments.c4_diagnostics.exp1_synthetic_gt_recovery import _build_observations, fallback_geometry
from experiments.c4_diagnostics.exp2c4b_oracle_field_encoder_only import _respiratory_only_motion
from experiments.c4_diagnostics.exp3b1a_c4a_bridge import _classify_bridge, _existing_b1_at_step, _exposure_accounting, _fixed_pixel_grid, _pixel_uv, _protocol, run
from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom


def _observation():
    domain = {"world_min_mm": [-4.0, -4.0, -4.0], "world_max_mm": [4.0, 4.0, 4.0], "cardiac_box": {"min_mm": [-2.0, -2.0, -2.0], "max_mm": [2.0, 2.0, 2.0]}}
    motion = _respiratory_only_motion(GroundTruthMotion.from_domain(domain))
    frames = fallback_geometry(locations_per_view=1, frames_per_location=3, image_shape=(8, 8), motion=motion, device=torch.device("cpu"))
    return _build_observations(frames, image_shape=(8, 8), phantom=AnalyticCanonicalPhantom(domain), motion=motion, psf_samples=1, seed=0)[0]


def test_fixed_grid_is_deterministic_while_random_mode_uses_non_grid_sampling() -> None:
    observation = _observation()
    first, second = _pixel_uv(observation, "fixed_grid", 256), _pixel_uv(observation, "fixed_grid", 256)
    torch.testing.assert_close(first, second); torch.testing.assert_close(first, _fixed_pixel_grid(observation, count=256))
    random_uv = _pixel_uv(observation, "random", 256)
    assert random_uv.shape == first.shape


def test_bridge_protocols_only_differ_by_pixel_mode_and_exposure_is_c4a_budget() -> None:
    r0, r1 = _protocol("random"), _protocol("fixed_grid")
    assert {key: value for key, value in r0.items() if key != "pixel_mode"} == {key: value for key, value in r1.items() if key != "pixel_mode"}
    assert _exposure_accounting(steps=750, observations_per_update=32, pixels_per_observation=256) == {"optimizer_steps": 750, "observations_per_update": 32, "pixels_per_observation": 256, "observation_exposures": 24000, "pixel_observation_exposures": 6144000}


def test_bridge_classification_is_based_on_full_fov_dvf_contract() -> None:
    passed = {"full_fov_dynamic_cosine": .96, "full_fov_dynamic_rms_ratio": 1., "full_fov_dynamic_normalized_epe": .4}
    failed = {"full_fov_dynamic_cosine": .7, "full_fov_dynamic_rms_ratio": .2, "full_fov_dynamic_normalized_epe": 2.}
    assert _classify_bridge(failed, passed, passed) == "c4a_reproduction_failure"
    assert _classify_bridge(passed, failed, failed) == "fixed_pixel_sampling_degrades_mbc_recovery"
    assert _classify_bridge(passed, passed, failed) == "exact_fullbatch_optimization_is_primary_bridge_difference"


def test_cpu_smoke_runs_both_arms_with_only_level0_changes(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    args = Namespace(source_config=root / "configs/source_first_change4_paperaligned.yaml", experiment_config=root / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml", manifest=None, qc_table=None, canonical_domain=None, output_dir=tmp_path / "bridge", existing_b1_summary=None, existing_b1_curve=None, arm="all", device="cpu", seed=0, locations_per_view=1, frames_per_location=3, steps=1, observations_per_update=3, pixel_samples=256, evaluation_grid_shape=(3, 3, 3), smoke=True)
    results = run(args)
    assert set(results) == {"R0", "R1"}
    for result in results.values():
        assert result["parameters_changed"] and all(name.startswith("respiratory_mbc.levels.0.") for name in result["parameters_changed"])
        assert result["encoder_calls"] == 0


def test_existing_b1_curve_parser_reads_the_step_750_row(tmp_path: Path) -> None:
    curve = tmp_path / "curve.csv"
    with curve.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["global_step", "full_fov_dynamic_cosine", "full_fov_dynamic_rms_ratio", "full_fov_dynamic_normalized_epe"]); writer.writeheader(); writer.writerow({"global_step": 750, "full_fov_dynamic_cosine": .9272, "full_fov_dynamic_rms_ratio": 1.3617, "full_fov_dynamic_normalized_epe": .7980})
    assert _existing_b1_at_step(tmp_path / "unused.json", curve) == {"full_fov_dynamic_cosine": .9272, "full_fov_dynamic_rms_ratio": 1.3617, "full_fov_dynamic_normalized_epe": .7980}


def test_formal_bridge_requires_c4a_budget_and_geometry_inputs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    args = Namespace(source_config=root / "configs/source_first_change4_paperaligned.yaml", experiment_config=root / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml", manifest=None, qc_table=None, canonical_domain=None, output_dir=tmp_path / "bridge", existing_b1_summary=None, existing_b1_curve=None, arm="all", device="cpu", seed=0, locations_per_view=3, frames_per_location=50, steps=750, observations_per_update=32, pixel_samples=256, evaluation_grid_shape=(3, 3, 3), smoke=False)
    with pytest.raises(ValueError, match="manifest, QC table, and canonical domain"):
        run(args)
