from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from experiments.c4_diagnostics.exp2b_score_dc_gradient_audit import (
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
