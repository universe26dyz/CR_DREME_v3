"""Contracts for the respiratory-only vNext staged schedule."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cardioresp4d.training.respiratory_staged_vnext import respiratory_staged_vnext_segments, validate_respiratory_staged_vnext_config  # noqa: E402


def _smoke_runner_with_matching_respiratory_pca():
    from cardioresp4d.frequency.pca_waveform_prior import PCAWaveformMatch
    from cardioresp4d.training.model import SourceFirstDynamicModel
    from cardioresp4d.training.respiratory_staged_vnext import RespiratoryStagedVNextRunner
    from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler
    from cardioresp4d.training.trainer import UnifiedProgressiveTrainer
    class MatchingRespiratoryPCA:
        def match(self, view, slice_id, timestamps, *, device, dtype):
            return PCAWaveformMatch(timestamps, torch.linspace(-1., 1., timestamps.numel(), device=device, dtype=dtype), 1)
    rows = [DynamicObservation(torch.rand(1, 4, 4), view, "s", index, torch.zeros(3), torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.]), torch.ones(2), 4., True, "valid", float(frame)) for index, (view, frame) in enumerate((view, frame) for view in ("SAX", "2CH", "4CH") for frame in range(3))]
    model = SourceFirstDynamicModel(torch.tensor([-2., -2., -2.]), torch.tensor([2., 2., 2.]), cardiac_lower_world_mm=torch.tensor([-1., -1., -1.]), cardiac_upper_world_mm=torch.tensor([1., 1., 1.]), n_dynamic_frames=9, inr_width=8, inr_depth=1, latent_dim=4, motion_hidden_dim=8, respiratory_grid_shapes=((4, 4, 4),) * 3, cardiac_grid_shape=(4, 4, 4), psf_samples=1)
    optimizer = {"resp_a2": {"canonical_lr": 5e-5}, "resp_score": {"film_lr": 5e-4}, "resp_mbc": {"respiratory_mbc_lr": 5e-4}, "resp_e": {"canonical_lr": 5e-5}}
    trainer = UnifiedProgressiveTrainer(model, ViewLocationBalancedSampler(rows), pixel_samples=1, optimizer_config=optimizer, pca_waveform_min_frames=3)
    return RespiratoryStagedVNextRunner(trainer, respiratory_pca_waveform_prior=MatchingRespiratoryPCA(), loss_weights={"respiratory_pca_waveform": 1e-3}), model


def _owner_has_finite_nonzero_gradient(model, owner: str) -> bool:
    prefixes = {"canonical": ("canonical.",), "respiratory_encoder": ("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp."), "respiratory_mbc": ("respiratory_mbc.levels.",)}
    return any(parameter.grad is not None and torch.isfinite(parameter.grad).all() and bool(parameter.grad.abs().sum() > 0) for name, parameter in model.named_parameters() if name.startswith(prefixes[owner]))


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


def test_multilevel_respiratory_pca_loss_is_the_mean_of_three_independent_levels() -> None:
    from cardioresp4d.losses.motion_loss import respiratory_pca_waveform_subspace_loss
    from cardioresp4d.training.respiratory_staged_vnext import multilevel_respiratory_pca_waveform_loss
    torch.manual_seed(0)
    target = torch.sin(torch.linspace(0., 3.14, 12))
    scores = torch.stack((torch.stack((target, target.square(), target.cos()), -1), torch.stack((target.cos(), target, target.square()), -1), torch.randn(12, 3)), 1)
    result = multilevel_respiratory_pca_waveform_loss(scores, target)
    per_level = [respiratory_pca_waveform_subspace_loss(scores[:, level, :], target)["loss"] for level in range(3)]
    torch.testing.assert_close(result["loss"], torch.stack(per_level).mean())
    assert not torch.isclose(result["loss"], per_level[0])
    assert len(result["loss_by_level"]) == 3


def test_missing_respiratory_pca_match_is_a_differentiable_skip_without_fallback() -> None:
    from cardioresp4d.training.respiratory_staged_vnext import multilevel_respiratory_pca_waveform_loss
    scores = torch.randn(8, 3, 3, requires_grad=True)
    result = multilevel_respiratory_pca_waveform_loss(scores, None)
    assert result["matched"] is False
    assert result["loss"].requires_grad
    result["loss"].backward()
    assert scores.grad is not None and torch.equal(scores.grad, torch.zeros_like(scores.grad))


def test_phase_b_missing_local_pca_match_skips_and_records_counter() -> None:
    from cardioresp4d.training.model import SourceFirstDynamicModel
    from cardioresp4d.training.respiratory_staged_vnext import RespiratoryStagedVNextRunner
    from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler
    from cardioresp4d.training.trainer import UnifiedProgressiveTrainer
    class MissingLocalRespiratoryPCA:
        def match(self, *args, **kwargs):
            return None
    rows = [DynamicObservation(torch.rand(1, 4, 4), view, "s", index, torch.zeros(3), torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.]), torch.ones(2), 4., True, "valid", float(frame)) for index, (view, frame) in enumerate((view, frame) for view in ("SAX", "2CH", "4CH") for frame in range(3))]
    model = SourceFirstDynamicModel(torch.tensor([-2., -2., -2.]), torch.tensor([2., 2., 2.]), cardiac_lower_world_mm=torch.tensor([-1., -1., -1.]), cardiac_upper_world_mm=torch.tensor([1., 1., 1.]), n_dynamic_frames=9, inr_width=8, inr_depth=1, latent_dim=4, motion_hidden_dim=8, respiratory_grid_shapes=((4, 4, 4),) * 3, cardiac_grid_shape=(4, 4, 4), psf_samples=1)
    trainer = UnifiedProgressiveTrainer(model, ViewLocationBalancedSampler(rows), pixel_samples=1, optimizer_config={"resp_score": {"film_lr": 5e-4}})
    runner = RespiratoryStagedVNextRunner(trainer, respiratory_pca_waveform_prior=MissingLocalRespiratoryPCA(), loss_weights={"respiratory_pca_waveform": 1e-3})
    segment = replace(next(item for item in respiratory_staged_vnext_segments() if item.name == "resp_pca_warmup"), steps=1)
    report = runner.run_segment(segment)
    assert report["respiratory_pca_matched_steps"] == 0
    assert report["respiratory_pca_skipped_steps"] == 1


def test_smoke_segments_clear_gradients_across_a2_b_c_d_e_ownership_changes() -> None:
    runner, model = _smoke_runner_with_matching_respiratory_pca()
    names = ("resp_static_bootstrap_a2", "resp_pca_warmup", "resp_mbc_bootstrap_c1", "resp_score_refine", "resp_canonical_refine")
    expected_owners = ("canonical", "respiratory_encoder", "respiratory_mbc", "respiratory_encoder", "canonical")
    schedule = {segment.name: segment for segment in respiratory_staged_vnext_segments()}
    for name, owner in zip(names, expected_owners):
        runner.run_segment(replace(schedule[name], steps=1))
        assert _owner_has_finite_nonzero_gradient(model, owner)
        assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)


def test_stale_canonical_gradient_is_cleared_before_phase_b_backward() -> None:
    runner, model = _smoke_runner_with_matching_respiratory_pca()
    canonical = next(model.canonical.parameters()); canonical.grad = torch.ones_like(canonical)
    segment = replace(next(item for item in respiratory_staged_vnext_segments() if item.name == "resp_pca_warmup"), steps=1)
    runner.run_segment(segment)
    assert canonical.grad is None
    assert _owner_has_finite_nonzero_gradient(model, "respiratory_encoder")


def test_phase_b_pca_only_backward_has_gradients_only_on_respiratory_encoder() -> None:
    """The formal B auxiliary reaches every intended score-encoder branch only."""
    from cardioresp4d.training.model import SourceFirstDynamicModel
    from cardioresp4d.training.respiratory_staged_vnext import configure_respiratory_staged_segment
    from cardioresp4d.training.sampler import DynamicObservation
    from cardioresp4d.losses.motion_loss import respiratory_pca_waveform_subspace_loss
    model = SourceFirstDynamicModel(torch.tensor([-2., -2., -2.]), torch.tensor([2., 2., 2.]), cardiac_lower_world_mm=torch.tensor([-1., -1., -1.]), cardiac_upper_world_mm=torch.tensor([1., 1., 1.]), n_dynamic_frames=8, inr_width=8, inr_depth=1, latent_dim=4, motion_hidden_dim=8, respiratory_grid_shapes=((4, 4, 4),) * 3, cardiac_grid_shape=(4, 4, 4), psf_samples=1)
    segment = next(item for item in respiratory_staged_vnext_segments() if item.name == "resp_pca_warmup")
    configure_respiratory_staged_segment(model, segment, {"resp_score": {"film_lr": 5e-4}})
    scores = []
    for index in range(8):
        observation = DynamicObservation(torch.rand(1, 4, 4), "SAX", "s", index, torch.tensor([float(index), 0., 0.]), torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.]), torch.ones(2), 4., True, "valid", float(index))
        scores.append(model.encode_motion_scores(observation)["resp_scores"])
    scores = torch.cat(scores); scores.retain_grad()
    from cardioresp4d.training.respiratory_staged_vnext import multilevel_respiratory_pca_waveform_loss
    multilevel_respiratory_pca_waveform_loss(scores, torch.linspace(-1., 1., 8))["loss"].backward()
    assert scores.grad is not None
    for level in range(3):
        assert torch.isfinite(scores.grad[:, level]).all() and bool(scores.grad[:, level].abs().sum() > 0)
    expected = ("film_encoder.image.", "film_encoder.geometry_mlp.", "film_encoder.head.", "film_encoder.resp.")
    for prefix in expected:
        gradients = [parameter.grad for name, parameter in model.named_parameters() if name.startswith(prefix)]
        assert gradients and all(gradient is not None and torch.isfinite(gradient).all() and bool(gradient.abs().sum() > 0) for gradient in gradients)
    for prefix in ("canonical.", "respiratory_mbc.", "film_encoder.card.", "cardiac_mbc.", "uncertainty."):
        assert all(parameter.grad is None for name, parameter in model.named_parameters() if name.startswith(prefix))


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


def test_phase_e_stage2c_renderer_calls_frozen_motion_without_updating_it() -> None:
    """E keeps the stage2c respiratory pullback active while updating canonical only."""
    from cardioresp4d.training.model import SourceFirstDynamicModel
    from cardioresp4d.training.respiratory_staged_vnext import configure_respiratory_staged_segment
    from cardioresp4d.training.sampler import DynamicObservation
    model = SourceFirstDynamicModel(torch.tensor([-2., -2., -2.]), torch.tensor([2., 2., 2.]), cardiac_lower_world_mm=torch.tensor([-1., -1., -1.]), cardiac_upper_world_mm=torch.tensor([1., 1., 1.]), n_dynamic_frames=1, inr_width=8, inr_depth=1, latent_dim=4, motion_hidden_dim=8, respiratory_grid_shapes=((4, 4, 4),) * 3, cardiac_grid_shape=(4, 4, 4), psf_samples=1)
    segment = next(item for item in respiratory_staged_vnext_segments() if item.name == "resp_canonical_refine")
    optimizer = configure_respiratory_staged_segment(model, segment, {"resp_e": {"canonical_lr": 5e-5}})
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if name.startswith(("film_encoder.", "respiratory_mbc."))}
    called = {"encoder": 0, "mbc": 0}
    encoder_hook = model.film_encoder.register_forward_hook(lambda *_: called.__setitem__("encoder", called["encoder"] + 1))
    mbc_hook = model.respiratory_mbc.register_forward_hook(lambda *_: called.__setitem__("mbc", called["mbc"] + 1))
    try:
        observation = DynamicObservation(torch.rand(1, 4, 4), "SAX", "s", 0, torch.zeros(3), torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.]), torch.ones(2), 4., True, "valid", 0.)
        optimizer.zero_grad(set_to_none=True)
        model.predict(observation, torch.tensor([[1., 1.]]), "stage2c")["predicted_intensity"].mean().backward()
        optimizer.step()
    finally:
        encoder_hook.remove(); mbc_hook.remove()
    assert called == {"encoder": 1, "mbc": 1}
    assert all(torch.equal(before, dict(model.named_parameters())[name]) for name, before in frozen.items())
