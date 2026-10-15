from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from experiments.c4_diagnostics.exp2b_score_dc_gradient_audit import (
    _loss_terms,
    gradient_cosine,
    gradient_statistics,
)
from experiments.c4_diagnostics.exp2_checkpoint_common import (
    assert_parameters_unchanged,
    parameter_snapshot,
    reconstruct_motion_only_model,
)
from experiments.c4_diagnostics.exp2a_encoder_temporal_identifiability import validate_source_timestamps
from experiments.c4_diagnostics.synthetic.phantom import AnalyticCanonicalPhantom
from experiments.c4_diagnostics.synthetic.prior import build_synthetic_frequency_prior


class _Scores(torch.nn.Module):
    def __init__(self, parameter: torch.nn.Parameter) -> None:
        super().__init__(); self.parameter = parameter

    def forward(self, image: torch.Tensor, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.parameter.reshape(1, 1, 1)
        return {"resp_scores": value.expand(1, 3, 3), "card_scores": (2.0 * value).expand(1, 1, 3)}


class _MBC(torch.nn.Module):
    def raw_active(self, grid: torch.Tensor, *, active_levels: int) -> torch.Tensor:
        return torch.ones((1, active_levels, grid.shape[1], 3), dtype=grid.dtype, device=grid.device)


class _RandomizedDataModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.film_encoder = _Scores(self.parameter)
        self.respiratory_mbc = _MBC()
        self.canonical_lower_world_mm = torch.tensor([-1.0, -1.0, -1.0])
        self.canonical_upper_world_mm = torch.tensor([1.0, 1.0, 1.0])
        self.respiratory_smoothness_shape = (2, 2, 2)

    def predict(self, observation, pixel_uv: torch.Tensor, stage: str) -> dict[str, torch.Tensor]:
        assert stage == "stage3b"
        return {"predicted_intensity": self.parameter * torch.randn(pixel_uv.shape[0])}


def _randomized_sequence() -> list[SimpleNamespace]:
    geometry = {"center_mm": torch.zeros(3), "row_direction": torch.tensor([1.0, 0.0, 0.0]), "column_direction": torch.tensor([0.0, 1.0, 0.0]), "normal": torch.tensor([0.0, 0.0, 1.0]), "pixel_spacing_mm": torch.ones(2), "slice_thickness_mm": 1.0}
    return [SimpleNamespace(image=torch.zeros(1, 3, 3), view="SAX", slice_id="fixed", timestamp_s=float(index), **geometry) for index in range(3)]


def _config() -> dict[str, object]:
    return {
        "model": {
            "canonical": {"coarsest_resolution": 8, "finest_resolution": 1, "level_scale": 1.5, "features_per_level": 2, "log2_hashmap_size": 12, "latent_dim": 4, "width": 4, "depth": 1},
            "psf": {"n_samples": 2},
            "film": {"channels": 4, "position_bands": 2},
            "uncertainty": {"frame_embedding_dim": 2, "width": 4, "depth": 1},
            "respiratory_mbc": {"hidden_dim": 4, "logical_control_shapes": [[3, 3, 3], [3, 3, 3], [3, 3, 3]], "cps": 1},
            "cardiac_mbc": {"logical_control_shape": [3, 3, 3], "cps": 1, "taper_mm": 1.0},
        },
        "training": {"motion_regularization": {"respiratory_evaluation_grid": {"shape": [3, 3, 3]}, "cardiac_evaluation_grid": {"shape": [3, 3, 3]}}, "image_regularization": {"mode": "edge", "delta": 1.0}},
    }


def _domain() -> dict[str, object]:
    return {"world_min_mm": [-8.0, -8.0, -8.0], "world_max_mm": [8.0, 8.0, 8.0], "cardiac_box": {"min_mm": [-4.0, -4.0, -4.0], "max_mm": [4.0, 4.0, 4.0]}}


def test_motion_only_checkpoint_strict_loads_after_phantom_replacement() -> None:
    config, domain = _config(), _domain()
    source = reconstruct_motion_only_model(config, domain, {}, n_dynamic_frames=2, allow_empty_state=True)
    source.canonical = source.canonical.__class__(AnalyticCanonicalPhantom(domain))
    source.respiratory_mbc.set_active_levels(3)
    checkpoint = {name: value.detach().clone() for name, value in source.state_dict().items()}
    loaded = reconstruct_motion_only_model(config, domain, checkpoint, n_dynamic_frames=2)
    for name, value in checkpoint.items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


def test_weighted_gradient_statistics_use_actual_weighted_loss() -> None:
    parameter = torch.nn.Parameter(torch.tensor([3.0]))
    values = gradient_statistics(0.25 * parameter.square().sum(), [parameter])
    assert values["l2_norm"] == pytest.approx(1.5)
    assert values["rms_per_parameter"] == pytest.approx(1.5)
    assert values["max_abs"] == pytest.approx(1.5)


def test_gradient_cosine_is_safe_when_a_gradient_is_zero() -> None:
    assert torch.isnan(torch.tensor(gradient_cosine(torch.zeros(3), torch.ones(3))))


def test_parameter_snapshot_detects_no_diagnostic_mutation() -> None:
    module = torch.nn.Linear(2, 1)
    before = parameter_snapshot(module)
    assert_parameters_unchanged(module, before)
    with torch.no_grad():
        module.weight.add_(1.0)
    with pytest.raises(RuntimeError, match="mutated"):
        assert_parameters_unchanged(module, before)


def test_source_timestamps_are_validated_against_checkpoint_arrays(tmp_path) -> None:
    path = tmp_path / "recovery_arrays.npz"
    import numpy as np

    np.savez(path, SAX_fixed_timestamps_s=np.array([0.0, 1.0]))
    group = {"SAX/fixed": [SimpleNamespace(timestamp_s=0.0), SimpleNamespace(timestamp_s=1.0)]}
    assert validate_source_timestamps(path, group)


def test_data_loss_and_gradient_replay_exactly_with_the_same_psf_seed() -> None:
    model, sequence = _RandomizedDataModel(), _randomized_sequence()
    prior = build_synthetic_frequency_prior(.25, 1.1, {("SAX", "fixed")})
    def construct(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        terms, _ = _loss_terms(model, sequence, data_frame_count=3, pixels_per_frame=4, prior=prior, weights={"zero_mean_score": 1e-4, "cardiac_leakage_in_resp": .1, "mbc_normalization": .01}, data_psf_seed=seed)
        return terms["data"].detach(), torch.autograd.grad(terms["data"], model.parameter)[0].detach()
    first_loss, first_gradient = construct(17)
    second_loss, second_gradient = construct(17)
    changed_loss, changed_gradient = construct(18)
    torch.testing.assert_close(first_loss, second_loss)
    torch.testing.assert_close(first_gradient, second_gradient)
    assert not (torch.equal(first_loss, changed_loss) and torch.equal(first_gradient, changed_gradient))


def test_stage3b_zero_mean_includes_cardiac_scores() -> None:
    model, sequence = _RandomizedDataModel(), _randomized_sequence()
    prior = build_synthetic_frequency_prior(.25, 1.1, {("SAX", "fixed")})
    terms, _ = _loss_terms(model, sequence, data_frame_count=3, pixels_per_frame=4, prior=prior, weights={"zero_mean_score": 1e-4, "cardiac_leakage_in_resp": .1, "mbc_normalization": .01}, data_psf_seed=17)
    assert float(terms["stage3b_zero_mean_score"]) > float(terms["resp_only_zero_mean_score"])
