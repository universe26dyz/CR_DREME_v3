"""功能：以物理 mm 坐标的 cubic B-spline control grids 表示 DREME-MR motion basis components。
论文来源：Direct from DREME-MR global multi-resolution respiratory 与 local cardiac MBC。
输入：world-mm query points、per-level Cartesian scores。
输出：MBC fields 与单位为 mm 的 respiratory/cardiac DVF。
主要步骤：cubic B-spline interpolation；cardiac box 边界输出强制为零。
是否属于原论文直接实现 / 必要适配 / 可选实验：Direct principle；physical-mm grid 是 canonical normalization 的必要尺度契约。
命令行使用示例：由 Phase-2 motion modules 导入，无独立 CLI。
"""
from __future__ import annotations
import torch
from torch import nn


def _cubic(t: torch.Tensor) -> torch.Tensor:
    return torch.stack(((1-t)**3/6, (3*t**3-6*t**2+4)/6, (-3*t**3+3*t**2+3*t+1)/6, t**3/6), -1)


class CubicBSplineMBC(nn.Module):
    """One 3-component cubic B-spline field in an explicit world-mm box."""
    def __init__(self, domain_min_mm: torch.Tensor, domain_max_mm: torch.Tensor, resolution: int, *, cardiac: bool = False) -> None:
        super().__init__()
        if resolution < 4: raise ValueError("cubic B-spline resolution must be at least four")
        lower, upper = torch.as_tensor(domain_min_mm, dtype=torch.float32), torch.as_tensor(domain_max_mm, dtype=torch.float32)
        if lower.shape != (3,) or upper.shape != (3,) or torch.any(upper <= lower): raise ValueError("domain bounds must be increasing three-vectors")
        self.register_buffer("domain_min_mm", lower); self.register_buffer("domain_max_mm", upper); self.cardiac = cardiac
        self.controls = nn.Parameter(torch.zeros(3, resolution, resolution, resolution))
        mask = torch.ones_like(self.controls)
        if cardiac:
            mask[:, 0] = mask[:, -1] = 0; mask[:, :, 0] = mask[:, :, -1] = 0; mask[:, :, :, 0] = mask[:, :, :, -1] = 0
        self.register_buffer("control_mask", mask)

    def forward(self, points_mm: torch.Tensor) -> torch.Tensor:
        if points_mm.shape[-1] != 3: raise ValueError("points_mm must end in xyz")
        shape = points_mm.shape[:-1]; p = points_mm.reshape(-1, 3)
        normalized = (p - self.domain_min_mm) / (self.domain_max_mm - self.domain_min_mm)
        inside = torch.all((normalized >= 0) & (normalized <= 1), -1)
        coordinate = normalized.clamp(0, 1) * (self.controls.shape[1] - 3) + 1
        base = torch.floor(coordinate).long() - 1; frac = coordinate - torch.floor(coordinate)
        base = base.clamp(0, self.controls.shape[1] - 4); weights = _cubic(frac)
        field = torch.zeros(p.shape[0], 3, dtype=p.dtype, device=p.device); controls = self.controls * self.control_mask
        for ix in range(4):
            for iy in range(4):
                for iz in range(4):
                    value = controls[:, base[:,0]+ix, base[:,1]+iy, base[:,2]+iz].T
                    field = field + value * (weights[:,0,ix]*weights[:,1,iy]*weights[:,2,iz])[:,None]
        if self.cardiac:
            inside = inside & ~torch.any((normalized <= 0) | (normalized >= 1), -1)
        return (field * inside[:, None].to(field.dtype)).reshape(*shape, 3)


class RespiratoryMBC(nn.Module):
    def __init__(self, domain_min_mm: torch.Tensor, domain_max_mm: torch.Tensor, resolutions: tuple[int, int, int] = (8, 12, 16)) -> None:
        super().__init__(); self.levels = nn.ModuleList([CubicBSplineMBC(domain_min_mm, domain_max_mm, res) for res in resolutions])
    def forward(self, points_mm: torch.Tensor) -> torch.Tensor:
        return torch.stack([level(points_mm) for level in self.levels], dim=-3)


class CardiacMBC(CubicBSplineMBC):
    def __init__(self, domain_min_mm: torch.Tensor, domain_max_mm: torch.Tensor, resolution: int = 16) -> None:
        super().__init__(domain_min_mm, domain_max_mm, resolution, cardiac=True)


def dvf_from_scores(fields_mm: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Combine ``[B,L,N,3]`` fields with ``[B,L,3]`` scores into a world-mm DVF."""
    if fields_mm.ndim != 4 or scores.ndim != 3 or fields_mm.shape[:2] != scores.shape[:2] or fields_mm.shape[-1] != 3 or scores.shape[-1] != 3: raise ValueError("fields [B,L,N,3] and scores [B,L,3] are required")
    return (fields_mm * scores[:, :, None, :]).sum(1)
