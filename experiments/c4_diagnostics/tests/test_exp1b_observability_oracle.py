from __future__ import annotations

import math
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from experiments.c4_diagnostics.synthetic.motion import GroundTruthMotion
from experiments.c4_diagnostics.synthetic.observability import (
    image_sequence_metrics,
    legacy_flattened_temporal_std,
)
from experiments.c4_diagnostics.synthetic.oracle import (
    DirectMBCFit,
    OracleScoreGenerator,
    build_oracle_motion_model,
    derive_oracle_score_scales,
)
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from experiments.c4_diagnostics.synthetic.render import GeometryFrame, render_synthetic_observation
from experiments.c4_diagnostics.exp1b_observability_oracle import (
    _aggregate_by_amplitude_scale,
    _cardiac_support_mask,
    _direct_recovery_label,
    _normalization_grid_shapes,
    _oracle_optimizer,
    _oracle_render,
    _observability_label,
    _roi_mse,
    run,
)


def _domain() -> dict[str, object]:
    return {
        "world_min_mm": [-12.0, -12.0, -12.0],
        "world_max_mm": [12.0, 12.0, 12.0],
        "cardiac_box": {"min_mm": [-7.0, -7.0, -7.0], "max_mm": [7.0, 7.0, 7.0]},
    }


def _frame(timestamp_s: float) -> GeometryFrame:
    return GeometryFrame(
        "SAX", "test", torch.zeros(3), torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]),
        torch.tensor([1.5, 1.5]), 4.0, timestamp_s, 8, 8,
    )


def _scales(domain: dict[str, object], motion: GroundTruthMotion):
    cardiac = domain["cardiac_box"]
    return derive_oracle_score_scales(motion, lower_world_mm=torch.tensor(domain["world_min_mm"]), upper_world_mm=torch.tensor(domain["world_max_mm"]), cardiac_lower_world_mm=torch.tensor(cardiac["min_mm"]), cardiac_upper_world_mm=torch.tensor(cardiac["max_mm"]), respiratory_grid_shape=(3, 3, 3), cardiac_grid_shape=(3, 3, 3))


def test_true_temporal_std_excludes_static_spatial_contrast() -> None:
    image = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    sequence = torch.stack((image, image, image))
    values = image_sequence_metrics(sequence, sequence)
    assert values["image_true_temporal_std_mean_pred"] == pytest.approx(0.0, abs=1e-7)
    assert legacy_flattened_temporal_std(sequence) > 0.0


def test_gt_renderer_is_deterministic_and_zero_motion_is_observable() -> None:
    domain = _domain()
    phantom = AnalyticCanonicalPhantom(domain)
    motion = GroundTruthMotion.from_domain(domain, respiratory_amplitude_mm=5.0, cardiac_amplitude_mm=3.0)
    target = render_synthetic_observation(_frame(0.5), height=8, width=8, phantom=phantom, motion=motion, psf_samples=1, seed=3, dynamic_frame_id=0)
    repeated = render_synthetic_observation(_frame(0.5), height=8, width=8, phantom=phantom, motion=motion, psf_samples=1, seed=3, dynamic_frame_id=0)
    zero = GroundTruthMotion.from_domain(domain, respiratory_amplitude_mm=0.0, cardiac_amplitude_mm=0.0)
    static = render_synthetic_observation(_frame(0.5), height=8, width=8, phantom=phantom, motion=zero, psf_samples=1, seed=3, dynamic_frame_id=0)
    assert torch.allclose(target.image, repeated.image, atol=1e-6, rtol=0.0)
    assert (target.image - static.image).square().mean() > 1e-8


def test_oracle_scores_are_timestamp_only_and_match_shapes() -> None:
    domain, motion = _domain(), GroundTruthMotion.from_domain(_domain())
    generator = OracleScoreGenerator(motion, _scales(domain, motion), active_respiratory_levels=3)
    timestamps = torch.arange(0.0, 16.0, 0.125)
    scores_a = generator(timestamps)
    scores_b = generator(timestamps.clone() + 0.0, image=torch.randn(2, 3), geometry=torch.randn(4, 3))
    assert scores_a["resp_scores"].shape == (len(timestamps), 3, 3)
    assert scores_a["card_scores"].shape == (len(timestamps), 1, 3)
    assert torch.equal(scores_a["resp_scores"], scores_b["resp_scores"])
    peak = torch.fft.rfftfreq(len(timestamps), d=0.125)[torch.fft.rfft(scores_a["resp_scores"][:, 0, 0]).abs()[1:].argmax() + 1]
    cardiac_peak = torch.fft.rfftfreq(len(timestamps), d=0.125)[torch.fft.rfft(scores_a["card_scores"][:, 0, 0]).abs()[1:].argmax() + 1]
    assert float(peak) == pytest.approx(motion.respiratory_frequency_hz, abs=0.05)
    assert float(cardiac_peak) == pytest.approx(motion.cardiac_frequency_hz, abs=0.05)


def test_oracle_score_physical_scales_come_from_gt_normalization_grids() -> None:
    domain, motion = _domain(), GroundTruthMotion.from_domain(_domain())
    scales = derive_oracle_score_scales(motion, lower_world_mm=torch.tensor(domain["world_min_mm"]), upper_world_mm=torch.tensor(domain["world_max_mm"]), cardiac_lower_world_mm=torch.tensor(domain["cardiac_box"]["min_mm"]), cardiac_upper_world_mm=torch.tensor(domain["cardiac_box"]["max_mm"]), respiratory_grid_shape=(4, 4, 4), cardiac_grid_shape=(5, 5, 5))
    resp_target = motion.respiratory_dvf_mm(scales.respiratory_grid, torch.tensor([1.0 / (4.0 * motion.respiratory_frequency_hz)]))
    card_target = motion.cardiac_dvf_mm(scales.cardiac_grid, torch.tensor([(math.pi / 2.0 - .37) / (2.0 * math.pi * motion.cardiac_frequency_hz)]))
    assert torch.allclose(scales.respiratory_component_rms_mm, resp_target.square().mean(dim=(0, 1)).sqrt())
    assert torch.allclose(scales.cardiac_component_rms_mm, card_target.square().mean(dim=(0, 1)).sqrt())


def test_unit_normalized_mbc_factorization_with_scaled_oracle_scores_recovers_gt() -> None:
    domain, motion = _domain(), GroundTruthMotion.from_domain(_domain())
    scales = derive_oracle_score_scales(motion, lower_world_mm=torch.tensor(domain["world_min_mm"]), upper_world_mm=torch.tensor(domain["world_max_mm"]), cardiac_lower_world_mm=torch.tensor(domain["cardiac_box"]["min_mm"]), cardiac_upper_world_mm=torch.tensor(domain["cardiac_box"]["max_mm"]), respiratory_grid_shape=(4, 4, 4), cardiac_grid_shape=(4, 4, 4))
    target = motion.respiratory_dvf_mm(scales.respiratory_grid, torch.tensor([1.0 / (4.0 * motion.respiratory_frequency_hz)]))
    unit_mbc = target / scales.respiratory_component_rms_mm
    scores = OracleScoreGenerator(motion, scales, active_respiratory_levels=3)(torch.tensor([1.0 / (4.0 * motion.respiratory_frequency_hz)]))["resp_scores"]
    recovered = (unit_mbc[:, None] * scores[:, :, None]).sum(dim=1)
    assert torch.allclose(unit_mbc.square().mean(dim=(0, 1)).sqrt(), torch.ones(3), atol=1e-6)
    assert torch.allclose(recovered, target, atol=1e-6)


def test_formal_normalization_grid_is_c4_16_cubed() -> None:
    root = Path(__file__).resolve().parents[3]
    config = __import__("yaml").safe_load((root / "configs/source_first_change4_paperaligned.yaml").read_text())
    assert _normalization_grid_shapes(config, smoke=False) == ((16, 16, 16), (16, 16, 16))
    assert _normalization_grid_shapes(config, smoke=True) == ((3, 3, 3), (3, 3, 3))


def test_direct_fit_recovery_label_is_gt_relative() -> None:
    thresholds = {"max_normalized_epe": .25, "min_spatial_cosine": .90, "min_rms_ratio": .75, "max_rms_ratio": 1.25}
    assert _direct_recovery_label({"normalized_epe": .20, "spatial_cosine": .95, "rms_ratio": 1.1}, thresholds) == "pass"
    assert _direct_recovery_label({"normalized_epe": .30, "spatial_cosine": .99, "rms_ratio": 1.0}, thresholds) == "fail"


def test_observability_is_quantitative_when_gt_rerender_mse_is_zero() -> None:
    aggregate = {"zero_motion_observability_fraction": {"median": 1.0}, "zero_motion_mse_over_target_temporal_variance": {"median": 1e-6}}
    assert _observability_label(aggregate) == "quantitative_only"


def test_cli_path_import_bootstrap_from_repository_root() -> None:
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run([sys.executable, "experiments/c4_diagnostics/exp1b_observability_oracle.py", "--help"], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_oracle_only_trains_selected_mbc_and_keeps_canonical_frozen() -> None:
    model = build_oracle_motion_model(_domain(), AnalyticCanonicalPhantom(_domain()), device=torch.device("cpu"), active_respiratory_levels=1, hidden_dim=4, logical_shape=(2, 2, 2), psf_samples=1, normalization_grid_shapes=((3, 3, 3), (3, 3, 3)))
    model.configure_trainable("resp")
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert names and all(name.startswith("respiratory_mbc") for name in names)
    assert not any("encoder" in name for name, _ in model.named_parameters())
    assert not any(parameter.requires_grad for parameter in model.canonical.parameters())


def test_direct_fit_updates_only_selected_mbc() -> None:
    domain = _domain()
    model = build_oracle_motion_model(domain, AnalyticCanonicalPhantom(domain), device=torch.device("cpu"), active_respiratory_levels=1, hidden_dim=4, logical_shape=(2, 2, 2), psf_samples=1, normalization_grid_shapes=((3, 3, 3), (3, 3, 3)))
    fitter = DirectMBCFit(model, GroundTruthMotion.from_domain(domain), branch="resp", grid_shape=(3, 3, 3), lr=1e-3, normalization_weight=1e-2)
    before_card = [value.detach().clone() for value in model.cardiac_mbc.parameters()]
    before_resp = [value.detach().clone() for value in model.respiratory_mbc.parameters()]
    loss = fitter.step()
    assert math.isfinite(loss)
    assert all(torch.equal(before, after) for before, after in zip(before_card, model.cardiac_mbc.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(before_resp, model.respiratory_mbc.parameters()))


def test_same_motion_different_psf_seed_has_self_noise_but_same_seed_is_deterministic() -> None:
    domain, motion = _domain(), GroundTruthMotion.from_domain(_domain())
    phantom = AnalyticCanonicalPhantom(domain)
    first = render_synthetic_observation(_frame(.4), height=8, width=8, phantom=phantom, motion=motion, psf_samples=1, seed=10, dynamic_frame_id=0)
    same = render_synthetic_observation(_frame(.4), height=8, width=8, phantom=phantom, motion=motion, psf_samples=1, seed=10, dynamic_frame_id=0)
    different = render_synthetic_observation(_frame(.4), height=8, width=8, phantom=phantom, motion=motion, psf_samples=1, seed=11, dynamic_frame_id=0)
    assert torch.equal(first.image, same.image)
    assert (first.image - different.image).square().mean() > 0.0


def test_matched_crn_reuses_target_psf_and_stochastic_redraws_samples() -> None:
    domain = _domain(); phantom = AnalyticCanonicalPhantom(domain)
    motion = GroundTruthMotion.from_domain(domain, respiratory_amplitude_mm=0.0, cardiac_amplitude_mm=0.0)
    target = render_synthetic_observation(_frame(.4), height=8, width=8, phantom=phantom, motion=motion, psf_samples=1, seed=13, dynamic_frame_id=0)
    model = build_oracle_motion_model(domain, phantom, device=torch.device("cpu"), active_respiratory_levels=1, hidden_dim=4, logical_shape=(2, 2, 2), psf_samples=1, normalization_grid_shapes=((3, 3, 3), (3, 3, 3)))
    uv = torch.stack(torch.meshgrid(torch.arange(8), torch.arange(8), indexing="xy"), dim=-1).reshape(-1, 2).float()
    matched = _oracle_render(model, target, {}, "zero", uv, psf_training_mode="matched_crn", target_seed=13).reshape_as(target.image[0])
    stochastic_a = _oracle_render(model, target, {}, "zero", uv, psf_training_mode="stochastic", target_seed=13)
    stochastic_b = _oracle_render(model, target, {}, "zero", uv, psf_training_mode="stochastic", target_seed=13)
    assert torch.allclose(matched, target.image[0], atol=1e-6, rtol=0.0)
    assert not torch.equal(stochastic_a, stochastic_b)


def test_cardiac_roi_mse_is_not_diluted_by_whole_fov() -> None:
    predicted = torch.zeros(1, 1, 8, 8); target = predicted.clone(); target[..., 3:5, 3:5] = 1.0
    mask = torch.zeros_like(target, dtype=torch.bool); mask[..., 3:5, 3:5] = True
    assert _roi_mse(predicted, target, mask) > (predicted - target).square().mean()


def test_amplitude_scale_aggregates_are_separate() -> None:
    rows = {"a|scale=0": {"amplitude_scale": 0.0, "metric": 1.0}, "b|scale=0": {"amplitude_scale": 0.0, "metric": 3.0}, "a|scale=1": {"amplitude_scale": 1.0, "metric": 9.0}}
    aggregate = _aggregate_by_amplitude_scale(rows)
    assert aggregate["0"]["metric"]["median"] == 2.0
    assert aggregate["1"]["metric"]["median"] == 9.0


def test_direct_mbc_fit_uses_supplied_normalization_weight() -> None:
    domain = _domain(); model = build_oracle_motion_model(domain, AnalyticCanonicalPhantom(domain), device=torch.device("cpu"), active_respiratory_levels=1, hidden_dim=4, logical_shape=(2, 2, 2), psf_samples=1, normalization_grid_shapes=((3, 3, 3), (3, 3, 3)))
    fitter = DirectMBCFit(model, GroundTruthMotion.from_domain(domain), branch="resp", grid_shape=(3, 3, 3), lr=1e-3, normalization_weight=.37)
    terms = fitter.loss_terms()
    assert float(terms["loss"].detach()) == pytest.approx(float((terms["data_loss"] + .37 * terms["mbc_normalization"]).detach()))


def test_combined_oracle_optimizer_keeps_resp_and_card_learning_rates_separate() -> None:
    domain = _domain(); model = build_oracle_motion_model(domain, AnalyticCanonicalPhantom(domain), device=torch.device("cpu"), active_respiratory_levels=1, hidden_dim=4, logical_shape=(2, 2, 2), psf_samples=1, normalization_grid_shapes=((3, 3, 3), (3, 3, 3)))
    config = {"training": {"optimizer": {"stage2": {"respiratory_mbc_lr": 1e-3}, "stage3a": {"cardiac_mbc_lr": 2e-3}, "stage3b": {"respiratory_mbc_lr": 3e-3, "cardiac_mbc_lr": 4e-3}}}}
    optimizer = _oracle_optimizer(model, "combined", config)
    assert sorted(group["lr"] for group in optimizer.param_groups) == [3e-3, 4e-3]


def test_cpu_tiny_smoke_covers_observability_direct_and_oracle(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    def arguments(mode: str, output: Path) -> Namespace:
        return Namespace(
            source_config=root / "configs/source_first_change4_paperaligned.yaml",
            experiment_config=root / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml",
            manifest=None, qc_table=None, canonical_domain=None, mode=mode, output_dir=output,
            device="cpu", overwrite=False, smoke=True, locations_per_view=1,
            frames_per_location=3, steps=1, pixel_samples=4, observations_per_update=1,
            amplitude_scales=[1.0], psf_training_mode="stochastic", seed=7,
        )
    observed = run(arguments("observability", tmp_path / "observability"))
    direct = run(arguments("direct_mbc", tmp_path / "direct"))
    oracle_resp = run(arguments("oracle_resp_only", tmp_path / "oracle_resp"))
    oracle_card = run(arguments("oracle_card_only", tmp_path / "oracle_card"))
    oracle_combined = run(arguments("oracle_combined", tmp_path / "oracle_combined"))
    assert observed["observability"] == "quantitative_only"
    assert direct["resp_direct_mbc_fit"] in {"pass", "fail"}
    assert oracle_resp["oracle_resp_recovery"] in {"pass", "fail"}
    assert oracle_card["oracle_card_recovery"] in {"pass", "fail"}
    assert oracle_combined["oracle_combined_recovery"] in {"pass", "fail"}
