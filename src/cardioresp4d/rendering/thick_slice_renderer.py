"""功能：直接 query continuous INR 的 5-point Gauss-Legendre thick-slice forward renderer。
论文来源：Direct from SIMPLE-4D finite-thickness quadrature；DREME-style pullback motion。
输入：slice world geometry、DICOM SliceThickness、canonical INR、可选 sequential pullback motion。
输出：predicted slice、quadrature latent features、world/reference sample coordinates。
主要步骤：沿真实 plane normal 在 5 个 Gauss-Legendre nodes 采样，warp 后 query INR 并加权。
是否属于原论文直接实现 / 必要适配 / 可选实验：SIMPLE-4D direct physics；world-to-normalized conversion 是必要适配。
命令行使用示例：由训练 forward 导入，无独立 CLI。
"""
from __future__ import annotations
import torch
from torch import nn


class ThickSliceRenderer(nn.Module):
    """5-point deterministic Gauss-Legendre renderer; no dense-volume grid_sample path."""
    def __init__(self, world_to_normalized: torch.Tensor) -> None:
        super().__init__()
        transform = torch.as_tensor(world_to_normalized, dtype=torch.float32)
        if transform.shape != (4, 4): raise ValueError("world_to_normalized must be 4x4")
        self.register_buffer("world_to_normalized", transform)
        nodes, weights = torch.tensor([-0.9061798459, -0.5384693101, 0., 0.5384693101, 0.9061798459]), torch.tensor([0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851])
        self.register_buffer("nodes", nodes); self.register_buffer("weights", weights * .5)

    def forward(self, inr: nn.Module, *, center_mm: torch.Tensor, row_direction: torch.Tensor, column_direction: torch.Tensor, normal: torch.Tensor, pixel_spacing_mm: torch.Tensor, thickness_mm: torch.Tensor, height: int, width: int, motion: nn.Module | None = None) -> dict[str, torch.Tensor]:
        batch = center_mm.shape[0]
        if any(t.shape != (batch, 3) for t in (center_mm, row_direction, column_direction, normal)) or pixel_spacing_mm.shape != (batch, 2) or thickness_mm.shape != (batch,):
            raise ValueError("batched geometry shapes must be center/directions [B,3], spacing [B,2], thickness [B]")
        rows = torch.arange(height, device=center_mm.device, dtype=center_mm.dtype) - (height - 1.) / 2.
        cols = torch.arange(width, device=center_mm.device, dtype=center_mm.dtype) - (width - 1.) / 2.
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        plane = center_mm[:, None, None, :] + col_grid[None,:,:,None] * pixel_spacing_mm[:,None,None,1,None] * row_direction[:,None,None,:] + row_grid[None,:,:,None] * pixel_spacing_mm[:,None,None,0,None] * column_direction[:,None,None,:]
        sample_world = plane[:, None] + self.nodes[None,:,None,None,None] * (thickness_mm[:,None,None,None,None] / 2.) * normal[:,None,None,None,:]
        reference_world = motion(sample_world)["reference_points_mm"] if motion is not None else sample_world
        normalized = self._normalize(reference_world)
        intensity, latent = inr(normalized, return_features=True)
        intensity = intensity.squeeze(-1)
        predicted = (intensity * self.weights[None,:,None,None]).sum(1).unsqueeze(1)
        return {"predicted_slice": predicted, "latent_samples": latent, "sample_world_mm": sample_world, "reference_world_mm": reference_world, "query_normalized": normalized, "quadrature_weights": self.weights}

    def _normalize(self, points_mm: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((*points_mm.shape[:-1], 1), dtype=points_mm.dtype, device=points_mm.device)
        matrix = self.world_to_normalized.to(dtype=points_mm.dtype)
        return torch.einsum("ij,...j->...i", matrix, torch.cat((points_mm, ones), -1))[..., :3]
