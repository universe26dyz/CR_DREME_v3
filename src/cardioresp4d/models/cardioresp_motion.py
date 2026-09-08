"""功能：按 DREME-style sequential composition 生成 observation-to-reference pullback coordinates。
论文来源：DREME-MR sequential cardiac/respiratory deformation。
输入：observation world-mm points 与 cardiac/respiratory world-mm DVF fields。
输出：cardiac-step points、reference query points、各步 DVF（均为 mm）。
主要步骤：先 cardiac pullback，再在 cardiac-shifted position 评估 respiratory pullback。
是否属于原论文直接实现 / 必要适配 / 可选实验：Direct sequential principle；明确采用 renderer 所需 pullback convention。
命令行使用示例：由 thick-slice renderer 导入，无独立 CLI。

DVF convention：``d_c(y)`` and ``d_r(.)`` map an observation coordinate toward
canonical reference anatomy. Thus ``x_ref = y + d_c(y) + d_r(y+d_c(y))``.
"""
from __future__ import annotations
import torch
from torch import nn
from .bspline_mbc import dvf_from_scores


class ScoreWeightedMBCField(nn.Module):
    """Turn a DREME MBC plus ``[B, level, xyz]`` scores into a mm DVF.

    The flattened calculation follows ``sum_l score_l * MBC_l(x)``.  It restores
    arbitrary spatial query dimensions so the result is directly callable by
    the renderer and by :class:`SequentialPullbackMotion`.
    """
    def __init__(self, mbc: nn.Module, scores: torch.Tensor) -> None:
        super().__init__()
        if scores.ndim != 3 or scores.shape[-1] != 3:
            raise ValueError("scores must have shape [B, levels, 3]")
        self.mbc = mbc
        self.register_buffer("scores", scores.detach().clone().float())

    def forward(self, points_mm: torch.Tensor) -> torch.Tensor:
        if points_mm.ndim < 3 or points_mm.shape[-1] != 3:
            raise ValueError("points_mm must have shape [B, ..., 3]")
        batch, spatial_shape = points_mm.shape[0], points_mm.shape[1:-1]
        if self.scores.shape[0] not in (1, batch):
            raise ValueError("score batch must be one or match point batch")
        fields = self.mbc(points_mm.reshape(batch, -1, 3))
        # A respiratory MBC exposes levels explicitly; the one-level cardiac
        # MBC returns a single field and is promoted to the same contract.
        if fields.ndim == 3:
            fields = fields[:, None]
        scores = self.scores.to(points_mm).expand(batch, -1, -1)
        return dvf_from_scores(fields, scores).reshape(batch, *spatial_shape, 3)


class SequentialPullbackMotion(nn.Module):
    """Compose callable world-mm cardiac then respiratory pullback fields."""
    def __init__(self, respiratory_field: nn.Module, cardiac_field: nn.Module) -> None:
        super().__init__(); self.respiratory_field = respiratory_field; self.cardiac_field = cardiac_field

    def forward(self, observation_points_mm: torch.Tensor) -> dict[str, torch.Tensor]:
        if observation_points_mm.shape[-1] != 3: raise ValueError("observation_points_mm must end in xyz")
        cardiac_dvf_mm = self.cardiac_field(observation_points_mm)
        cardiac_points_mm = observation_points_mm + cardiac_dvf_mm
        respiratory_dvf_mm = self.respiratory_field(cardiac_points_mm)
        return {"cardiac_dvf_mm": cardiac_dvf_mm, "respiratory_dvf_mm": respiratory_dvf_mm,
                "cardiac_points_mm": cardiac_points_mm, "reference_points_mm": cardiac_points_mm + respiratory_dvf_mm}
