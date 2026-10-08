"""CPU contracts for C4's read-only backward-finiteness gate."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cardioresp4d.training.sampler import DynamicObservation, ViewLocationBalancedSampler  # noqa: E402
from cardioresp4d.training.stage_contract import paperaligned_segment  # noqa: E402
from cardioresp4d.training.trainer import UnifiedProgressiveTrainer  # noqa: E402
import preflight_c4_paperaligned_loss_scale as preflight  # noqa: E402


class _Resp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.levels = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(3)]); self.active = 0

    def set_active_levels(self, levels: int) -> None:
        self.active = levels


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.canonical = torch.nn.Linear(1, 1)
        self.film_encoder = torch.nn.Module(); self.film_encoder.shared = torch.nn.Linear(1, 1); self.film_encoder.resp = torch.nn.Linear(1, 1); self.film_encoder.card = torch.nn.Linear(1, 1)
        self.respiratory_mbc = _Resp(); self.cardiac_mbc = torch.nn.Linear(1, 1); self.uncertainty = torch.nn.Linear(1, 1)


def _observation(view: str, frame: int) -> DynamicObservation:
    return DynamicObservation(torch.zeros(1, 2, 2), view, f"{view}_s001", frame, torch.zeros(3), torch.tensor([1., 0., 0.]), torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.]), torch.ones(2), 1., True, "valid", float(frame))


def _trainer() -> UnifiedProgressiveTrainer:
    trainer = UnifiedProgressiveTrainer(_Model(), ViewLocationBalancedSampler([_observation(view, index) for index, view in enumerate(("SAX", "2CH", "4CH"))]), pixel_samples=1)

    def total() -> torch.Tensor:
        return sum((parameter.square().sum() for parameter in trainer.model.parameters()))

    def observation_components(_observation: DynamicObservation, stage: str) -> dict[str, torch.Tensor]:
        value = total(); zero = value * 0.
        result = {"data": value, "image": zero}
        if stage != "stage1": result.update({"mbc_normalization": zero, "smooth_resp": zero})
        if stage in {"stage3a", "stage3b", "stage3c"}: result["smooth_card"] = zero
        return result

    def temporal_components(_stage: str) -> dict[str, torch.Tensor]:
        value = total()
        return {"zero_mean_score": value, "cardiac_leakage_in_resp": value, "respiratory_leakage_in_card": value, "cardiac_target_concentration": value, "cardiac_pca_waveform": value}

    trainer._observation_components = observation_components  # type: ignore[method-assign]
    trainer._temporal_components = temporal_components  # type: ignore[method-assign]
    return trainer


def test_critical_paperaligned_segments_pass_cpu_backward_preflight() -> None:
    trainer = _trainer()
    checks = {name: preflight.backward_finite_check(trainer, paperaligned_segment(name)) for name in ("s1a", "s2a_init", "s3a", "s3b_full")}
    assert checks == {name: {"finite": True, "bad_parameter_names": []} for name in checks}


def test_preflight_detects_deliberately_injected_nonfinite_gradient() -> None:
    model = torch.nn.Linear(1, 1)
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    model.bias.grad = torch.zeros_like(model.bias)
    assert preflight.nonfinite_trainable_gradient_names(model) == ["weight"]
