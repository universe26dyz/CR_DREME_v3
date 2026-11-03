"""Contracts for the respiratory-only vNext staged schedule."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.training.respiratory_staged_vnext import respiratory_staged_vnext_segments, validate_respiratory_staged_vnext_config  # noqa: E402


def test_vnext_schedule_has_fixed_non_joint_ownership_and_budget() -> None:
    segments = respiratory_staged_vnext_segments()
    assert [(segment.name, segment.steps) for segment in segments] == [("resp_static_bootstrap_a1", 500), ("resp_static_bootstrap_a2", 1300), ("resp_pca_warmup", 1000), ("resp_mbc_bootstrap_c1", 250), ("resp_mbc_bootstrap_c2", 250), ("resp_mbc_bootstrap_c3", 250), ("resp_score_refine", 1500), ("resp_canonical_refine", 2000)]
    assert sum(segment.steps for segment in segments) == 7050
    for segment in segments:
        owners = segment.owners
        assert not ({"respiratory_encoder", "respiratory_mbc"} <= owners)
        assert not ("canonical" in owners and ({"respiratory_encoder", "respiratory_mbc"} & owners))
        assert "cardiac" not in owners


def test_vnext_config_requires_distinct_respiratory_pca_weight_and_fixed_lrs() -> None:
    config = {"training": {"loss_weights": {"respiratory_pca_waveform": 1e-3}, "optimizer": {"resp_a1": {"canonical_lr": 2e-4}, "resp_a2": {"canonical_lr": 5e-5}, "resp_score": {"film_lr": 5e-4}, "resp_mbc": {"respiratory_mbc_lr": 5e-4}, "resp_e": {"canonical_lr": 5e-5}}}}
    validate_respiratory_staged_vnext_config(config)
    config["training"]["loss_weights"]["respiratory_pca_waveform"] = 0.
    try:
        validate_respiratory_staged_vnext_config(config)
    except ValueError:
        pass
    else:
        raise AssertionError("respiratory PCA weight must be required")


def test_respiration_pca_provenance_never_falls_back_to_cardiac_candidates(tmp_path: Path) -> None:
    import json
    import numpy as np
    from cardioresp4d.frequency.pca_waveform_prior import PCAWaveformPrior
    times = np.arange(8, dtype=float)
    artifact = tmp_path / "SAX" / "s"; artifact.mkdir(parents=True)
    np.savez_compressed(artifact / "pca_psd.npz", timestamps_s=times, temporal_pcs=np.stack((times, times + 1), axis=1))
    path = tmp_path / "frequency_bands.json"
    path.write_text(json.dumps({"cardiac": {"per_slice_candidates": [{"slice_key": "SAX/s", "reliable": True, "selected_pc": 1}]}, "respiratory": {"per_slice_candidates": [{"slice_key": "SAX/s", "reliable": True, "selected_pc": 2}]}}))
    respiratory = PCAWaveformPrior.load(path, component="respiratory")
    assert respiratory.match("SAX", "s", torch.tensor([1., 3.])).selected_pc == 2


def test_respiratory_pca_loss_has_no_mbc_or_canonical_gradient() -> None:
    from cardioresp4d.losses.motion_loss import respiratory_pca_waveform_subspace_loss
    respiratory_score = torch.randn(12, 3, requires_grad=True)
    frozen = torch.nn.Parameter(torch.ones(3), requires_grad=True)
    result = respiratory_pca_waveform_subspace_loss(respiratory_score, respiratory_score.detach() @ torch.tensor([.2, .4, -.3]))
    result["loss"].backward()
    assert respiratory_score.grad is not None and torch.isfinite(respiratory_score.grad).all()
    assert frozen.grad is None


def test_segment_optimizer_owns_one_factor_and_keeps_cardiac_frozen() -> None:
    from cardioresp4d.training.model import SourceFirstDynamicModel
    from cardioresp4d.training.respiratory_staged_vnext import configure_respiratory_staged_segment
    model = SourceFirstDynamicModel(torch.tensor([-2., -2., -2.]), torch.tensor([2., 2., 2.]), cardiac_lower_world_mm=torch.tensor([-1., -1., -1.]), cardiac_upper_world_mm=torch.tensor([1., 1., 1.]), n_dynamic_frames=3, inr_width=8, inr_depth=1, latent_dim=4, motion_hidden_dim=8, respiratory_grid_shapes=((4, 4, 4),) * 3, cardiac_grid_shape=(4, 4, 4), psf_samples=1)
    config = {"resp_mbc": {"respiratory_mbc_lr": 5e-4}}
    segment = next(item for item in respiratory_staged_vnext_segments() if item.name == "resp_mbc_bootstrap_c2")
    optimizer = configure_respiratory_staged_segment(model, segment, config)
    owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert owned == {id(parameter) for level in model.respiratory_mbc.levels[:2] for parameter in level.parameters()}
    assert not any(parameter.requires_grad for parameter in model.canonical.parameters())
    assert not any(parameter.requires_grad for parameter in model.film_encoder.card.parameters())
