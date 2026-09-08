"""功能：以多分辨率 hash encoding 表示连续 canonical anatomy。
论文来源：SIMPLE-4D Instant-NGP-style canonical INR；NeSVoR INR latent feature。
输入：canonical_domain.json 的 normalized coordinates ``[...,3]``。
输出：intensity ``[...,1]``，可选 latent feature ``[...,D]``。
主要步骤：trilinear hash-grid encoding 后由紧凑 MLP 查询连续信号。
是否属于原论文直接实现 / 必要适配 / 可选实验：SIMPLE-4D principle 的纯 PyTorch necessary adaptation。
命令行使用示例：由训练模块导入 ``CanonicalINR``，无独立 CLI。
"""
from __future__ import annotations
import torch
from torch import nn


class HashGridEncoder(nn.Module):
    """Pure-PyTorch trilinear multi-resolution hash grid for normalized ``[-1,1]^3`` points."""
    def __init__(self, *, levels: int, features_per_level: int, hash_size: int, min_resolution: int, max_resolution: int) -> None:
        super().__init__()
        if min(levels, features_per_level, hash_size, min_resolution, max_resolution) <= 0 or max_resolution < min_resolution:
            raise ValueError("hash-grid sizes must be positive with max_resolution >= min_resolution")
        self.levels, self.features_per_level, self.hash_size = levels, features_per_level, hash_size
        self.resolutions = tuple(round(min_resolution * (max_resolution / min_resolution) ** (level / max(levels - 1, 1))) for level in range(levels))
        self.tables = nn.ParameterList([nn.Parameter(torch.empty(hash_size, features_per_level)) for _ in range(levels)])
        for table in self.tables: nn.init.uniform_(table, -1e-4, 1e-4)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.shape[-1] != 3: raise ValueError("xyz must end in 3 normalized coordinates")
        flat = xyz.reshape(-1, 3); result = []
        offsets = torch.tensor([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]], device=xyz.device, dtype=torch.long)
        for resolution, table in zip(self.resolutions, self.tables):
            scaled = ((flat.clamp(-1., 1.) + 1.) * .5) * (resolution - 1)
            lower = torch.floor(scaled).long(); frac = scaled - lower
            corners = lower[:, None, :] + offsets[None]
            hashed = self._hash(corners) % self.hash_size
            values = table[hashed]
            weights = torch.where(offsets[None].bool(), frac[:, None, :], 1. - frac[:, None, :]).prod(-1)
            result.append((values * weights[..., None]).sum(1))
        return torch.cat(result, -1).reshape(*xyz.shape[:-1], -1)

    @staticmethod
    def _hash(indices: torch.Tensor) -> torch.Tensor:
        return (indices[..., 0] * 1540863) ^ (indices[..., 1] * 1259921) ^ (indices[..., 2] * 1936771)


class CanonicalINR(nn.Module):
    """Continuous image-domain anatomy whose domain is supplied externally by canonical-domain geometry."""
    def __init__(self, *, levels: int = 12, features_per_level: int = 2, hash_size: int = 2**15, min_resolution: int = 16, max_resolution: int = 256, hidden_dim: int = 64, latent_dim: int = 32) -> None:
        super().__init__()
        self.encoder = HashGridEncoder(levels=levels, features_per_level=features_per_level, hash_size=hash_size, min_resolution=min_resolution, max_resolution=max_resolution)
        encoded_dim = levels * features_per_level
        self.backbone = nn.Sequential(nn.Linear(encoded_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim), nn.ReLU())
        self.intensity_head = nn.Linear(latent_dim, 1)

    def forward(self, xyz_normalized: torch.Tensor, return_features: bool = False):
        latent = self.backbone(self.encoder(xyz_normalized))
        intensity = torch.sigmoid(self.intensity_head(latent))
        return (intensity, latent) if return_features else intensity
