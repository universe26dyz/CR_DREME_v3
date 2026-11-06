"""Optional score-gauge policy used only by the isolated Exp2-A experiment."""
from __future__ import annotations

from typing import Any, Callable, Mapping

import torch


def effective_reference_scores(raw: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "resp_scores" not in raw or "resp_scores" not in reference:
        raise KeyError("reference gauge requires resp_scores in both branches")
    result = dict(raw)
    result["resp_scores"] = raw["resp_scores"] - reference["resp_scores"]
    result["raw_resp_scores"] = raw["resp_scores"]
    result["reference_resp_scores"] = reference["resp_scores"]
    return result


class ReferenceScoreGauge:
    """Non-trainable provider wrapper; scores are always recomputed by caller."""

    def __init__(self, reference_provider: Callable[[Any], torch.Tensor]) -> None:
        self._reference_provider = reference_provider

    def reference_image(self, observation: Any) -> torch.Tensor:
        return self._reference_provider(observation)

    def __call__(self, raw: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return effective_reference_scores(raw, reference)
