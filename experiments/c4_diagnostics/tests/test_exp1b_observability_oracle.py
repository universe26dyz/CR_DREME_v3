from __future__ import annotations

import math
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
)
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from experiments.c4_diagnostics.synthetic.render import GeometryFrame, render_synthetic_observation
from experiments.c4_diagnostics.exp1b_observability_oracle import run


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
    motion = GroundTruthMotion()
    generator = OracleScoreGenerator(motion, active_respiratory_levels=3)
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


def test_oracle_only_trains_selected_mbc_and_keeps_canonical_frozen() -> None:
    model = build_oracle_motion_model(_domain(), AnalyticCanonicalPhantom(_domain()), device=torch.device("cpu"), active_respiratory_levels=1, hidden_dim=4, logical_shape=(2, 2, 2), psf_samples=1)
    model.configure_trainable("resp")
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert names and all(name.startswith("respiratory_mbc") for name in names)
    assert not any("encoder" in name for name, _ in model.named_parameters())
    assert not any(parameter.requires_grad for parameter in model.canonical.parameters())


def test_direct_fit_updates_only_selected_mbc() -> None:
    domain = _domain()
    model = build_oracle_motion_model(domain, AnalyticCanonicalPhantom(domain), device=torch.device("cpu"), active_respiratory_levels=1, hidden_dim=4, logical_shape=(2, 2, 2), psf_samples=1)
    fitter = DirectMBCFit(model, GroundTruthMotion.from_domain(domain), branch="resp", grid_shape=(3, 3, 3), lr=1e-3)
    before_card = [value.detach().clone() for value in model.cardiac_mbc.parameters()]
    before_resp = [value.detach().clone() for value in model.respiratory_mbc.parameters()]
    loss = fitter.step()
    assert math.isfinite(loss)
    assert all(torch.equal(before, after) for before, after in zip(before_card, model.cardiac_mbc.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(before_resp, model.respiratory_mbc.parameters()))


def test_cpu_tiny_smoke_covers_observability_direct_and_oracle(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    def arguments(mode: str, output: Path) -> Namespace:
        return Namespace(
            source_config=root / "configs/source_first_change4_paperaligned.yaml",
            experiment_config=root / "experiments/c4_diagnostics/configs/exp1_synthetic_gt.yaml",
            manifest=None, qc_table=None, canonical_domain=None, mode=mode, output_dir=output,
            device="cpu", overwrite=False, smoke=True, locations_per_view=1,
            frames_per_location=3, steps=1, pixel_samples=4, observations_per_update=1,
            amplitude_scales=[1.0], seed=7,
        )
    observed = run(arguments("observability", tmp_path / "observability"))
    direct = run(arguments("direct_mbc", tmp_path / "direct"))
    oracle_resp = run(arguments("oracle_resp_only", tmp_path / "oracle_resp"))
    oracle_card = run(arguments("oracle_card_only", tmp_path / "oracle_card"))
    oracle_combined = run(arguments("oracle_combined", tmp_path / "oracle_combined"))
    assert observed["observability"] in {"strong", "weak"}
    assert direct["resp_direct_mbc_fit"] in {"pass", "fail"}
    assert oracle_resp["oracle_resp_recovery"] in {"pass", "fail"}
    assert oracle_card["oracle_card_recovery"] in {"pass", "fail"}
    assert oracle_combined["oracle_combined_recovery"] in {"pass", "fail"}
