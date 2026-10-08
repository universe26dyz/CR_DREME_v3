"""Paper-aligned C4 scheduling, batching, and diagnostic contracts."""
from __future__ import annotations

import copy
import sys
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler  # noqa: E402
from cardioresp4d.training.source_first_config import validate_source_first_config  # noqa: E402
from cardioresp4d.training.stage_contract import paperaligned_segment  # noqa: E402
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer, accumulate_mean_loss_backward, effective_loss_weights  # noqa: E402
import train_source_first  # noqa: E402


def _item(view: str, frame: int) -> DynamicObservation:
    return DynamicObservation(
        image=torch.zeros(1, 2, 2), view=view, slice_id=f"{view}_s{frame % 2}", dynamic_frame_id=frame,
        center_mm=torch.zeros(3), row_direction=torch.tensor([1., 0., 0.]), column_direction=torch.tensor([0., 1., 0.]), normal=torch.tensor([0., 0., 1.]), pixel_spacing_mm=torch.ones(2), slice_thickness_mm=1., qc_valid=True, qc_reason="valid", timestamp_s=float(frame),
    )


class _Resp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.levels = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(3)]); self.active = 0

    def set_active_levels(self, levels: int) -> None:
        self.active = levels


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.canonical = torch.nn.Linear(1, 1); self.film_encoder = torch.nn.Module()
        self.film_encoder.shared = torch.nn.Linear(1, 1); self.film_encoder.resp = torch.nn.Linear(1, 1); self.film_encoder.card = torch.nn.Linear(1, 1)
        self.respiratory_mbc = _Resp(); self.cardiac_mbc = torch.nn.Linear(1, 1); self.uncertainty = torch.nn.Linear(1, 1)


def _trainer() -> UnifiedProgressiveTrainer:
    items = [_item(view, index) for index, view in enumerate(("SAX", "2CH", "4CH"))]
    config = yaml.safe_load((ROOT / "configs" / "source_first_change4_paperaligned.yaml").read_text(encoding="utf-8"))
    return UnifiedProgressiveTrainer(_Model(), ViewLocationBalancedSampler(items), pixel_samples=1, optimizer_config=config["training"]["optimizer"])


def test_paperaligned_segments_have_requested_freeze_lr_and_progression() -> None:
    trainer = _trainer()
    s1a, s1b = paperaligned_segment("s1a"), paperaligned_segment("s1b")
    trainer._configure_stage(s1a.stage, segment=s1a)
    assert trainer.optimizer is not None
    assert next(group["lr"] for group in trainer.optimizer.param_groups if group["name"] == "canonical") == 2e-4
    trainer._configure_stage(s1b.stage, segment=s1b)
    assert next(group["lr"] for group in trainer.optimizer.param_groups if group["name"] == "canonical") == 5e-5
    for name, levels, canonical in (("s2a_init", 1, False), ("s2a_joint", 1, True), ("s2b_init", 2, False), ("s2b_joint", 2, True), ("s2c_init", 3, False), ("s2c_joint", 3, True)):
        segment = paperaligned_segment(name); trainer._configure_stage(segment.stage, segment=segment)
        assert trainer.model.respiratory_mbc.active == levels
        assert all(parameter.requires_grad is canonical for parameter in trainer.model.canonical.parameters())
    trainer._configure_stage("stage3a", segment=paperaligned_segment("s3a"))
    assert not any(parameter.requires_grad for parameter in trainer.model.canonical.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.film_encoder.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.respiratory_mbc.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.cardiac_mbc.parameters())
    assert not any(parameter.requires_grad for parameter in trainer.model.uncertainty.parameters())
    trainer._configure_stage("stage3b", segment=paperaligned_segment("s3b_full"))
    assert all(parameter.requires_grad for parameter in trainer.model.canonical.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.film_encoder.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.respiratory_mbc.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.cardiac_mbc.parameters())


def test_regular_stage3a_preserves_card_head_only_semantics() -> None:
    trainer = _trainer()
    trainer._configure_stage("stage3a")
    assert not any(parameter.requires_grad for parameter in trainer.model.canonical.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.film_encoder.card.parameters())
    assert not any(parameter.requires_grad for parameter in trainer.model.film_encoder.shared.parameters())
    assert not any(parameter.requires_grad for parameter in trainer.model.film_encoder.resp.parameters())
    assert not any(parameter.requires_grad for parameter in trainer.model.respiratory_mbc.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.cardiac_mbc.parameters())
    assert not any(parameter.requires_grad for parameter in trainer.model.uncertainty.parameters())


def test_paperaligned_s3a_trains_full_motion_with_canonical_frozen() -> None:
    trainer = _trainer()
    trainer._configure_stage("stage3a", segment=paperaligned_segment("s3a"))
    assert not any(parameter.requires_grad for parameter in trainer.model.canonical.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.film_encoder.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.respiratory_mbc.parameters())
    assert all(parameter.requires_grad for parameter in trainer.model.cardiac_mbc.parameters())
    assert not any(parameter.requires_grad for parameter in trainer.model.uncertainty.parameters())


def test_paperaligned_s3a_all_three_resp_levels_active_and_trainable() -> None:
    trainer = _trainer()
    trainer._configure_stage("stage3a", segment=paperaligned_segment("s3a"))
    assert trainer.model.respiratory_mbc.active == 3
    assert all(all(parameter.requires_grad for parameter in level.parameters()) for level in trainer.model.respiratory_mbc.levels)


def test_sampler_balances_32_observations_and_restores_rotation() -> None:
    items = [_item(view, index) for index, view in enumerate(("SAX", "2CH", "4CH") * 3)]
    first = ViewLocationBalancedSampler(items, seed=4); selected = first.sample_step(observations_per_update=32)
    counts = {view: sum(item.view == view for item in selected) for view in ("SAX", "2CH", "4CH")}
    assert max(counts.values()) - min(counts.values()) <= 1
    state = copy.deepcopy(first.state_dict()); expected = [(item.view, item.slice_id, item.dynamic_frame_id) for item in first.sample_step(observations_per_update=32)]
    resumed = ViewLocationBalancedSampler(items, seed=99); resumed.load_state_dict(state)
    assert expected == [(item.view, item.slice_id, item.dynamic_frame_id) for item in resumed.sample_step(observations_per_update=32)]


def test_accumulated_mean_gradient_matches_single_mean_loss() -> None:
    accumulated_parameter, direct_parameter = torch.tensor(2., requires_grad=True), torch.tensor(2., requires_grad=True)
    accumulated = accumulate_mean_loss_backward([(accumulated_parameter - target).square() for target in (1., 3., 5.)])
    direct = torch.stack([(direct_parameter - target).square() for target in (1., 3., 5.)]).mean(); direct.backward()
    torch.testing.assert_close(accumulated, direct.detach())
    torch.testing.assert_close(accumulated_parameter.grad, direct_parameter.grad)


def test_paperaligned_config_has_only_requested_c4_losses_and_batching() -> None:
    config = yaml.safe_load((ROOT / "configs" / "source_first_change4_paperaligned.yaml").read_text(encoding="utf-8"))
    validate_source_first_config(config, ROOT)
    weights = effective_loss_weights(config["training"]["loss_weights"])
    assert config["training"]["observations_per_update"] == 32
    assert config["training"]["image_regularization"]["mode"] == "dreme_tv_stable"
    assert {name: weights[name] for name in ("image", "mbc_normalization", "zero_mean_score", "cardiac_leakage_in_resp", "respiratory_leakage_in_card", "smooth_resp", "smooth_card", "cardiac_target_concentration", "cardiac_pca_waveform")} == {"image": 2e-6, "mbc_normalization": 1e-2, "zero_mean_score": 1e-4, "cardiac_leakage_in_resp": 1e-1, "respiratory_leakage_in_card": 5e-2, "smooth_resp": 0., "smooth_card": 0., "cardiac_target_concentration": 0., "cardiac_pca_waveform": 0.}
    with patch.object(train_source_first.PCAWaveformPrior, "load") as load:
        assert train_source_first.optional_pca_waveform_prior(config["training"]["loss_weights"], Path("unused.json")) is None
        load.assert_not_called()
