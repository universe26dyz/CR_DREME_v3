"""功能：预测 NeSVoR-style pixel/frame variance，供后续 Gaussian NLL 使用。
论文来源：NeSVoR sigma_net 与 per-slice variance；SIMPLE-4D quadrature necessary adaptation。
输入：INR quadrature latent features ``[B,Q,H,W,D]`` 与 stable acquired-frame indices。
输出：positive pixel/frame/total variance；不在本模块施加独立 variance penalty。
主要步骤：latent+frame embedding 经 MLP softplus，按 renderer quadrature 聚合，再加 frame variance。
是否属于原论文直接实现 / 必要适配 / 可选实验：NeSVoR likelihood principle 的 dynamic-frame adaptation。
命令行使用示例：由后续 Gaussian-NLL loss 导入，无独立 CLI。
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class SliceUncertainty(nn.Module):
    """Positive pixel and acquired-frame variance with no standalone loss term."""
    def __init__(self, latent_dim: int, num_frames: int, embedding_dim: int = 8, epsilon: float = 1e-6) -> None:
        super().__init__()
        if min(latent_dim, num_frames, embedding_dim) <= 0 or epsilon <= 0: raise ValueError("uncertainty dimensions and epsilon must be positive")
        self.frame_embedding = nn.Embedding(num_frames, embedding_dim); self.pixel_head = nn.Sequential(nn.Linear(latent_dim + embedding_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 1)); self.frame_log_variance = nn.Embedding(num_frames, 1); self.epsilon = epsilon

    def forward(self, latent_samples: torch.Tensor, frame_indices: torch.Tensor, quadrature_weights: torch.Tensor) -> dict[str, torch.Tensor]:
        if latent_samples.ndim != 5 or frame_indices.ndim != 1 or latent_samples.shape[0] != frame_indices.shape[0]: raise ValueError("latent_samples [B,Q,H,W,D] and frame_indices [B] are required")
        batch, quadrature, height, width, _ = latent_samples.shape
        if quadrature_weights.shape != (quadrature,): raise ValueError("quadrature_weights must match latent Q")
        embedding = self.frame_embedding(frame_indices)[:,None,None,None,:].expand(batch, quadrature, height, width, -1)
        pixel_samples = F.softplus(self.pixel_head(torch.cat((latent_samples, embedding), -1)).squeeze(-1)) + self.epsilon
        pixel_variance = (pixel_samples * quadrature_weights[None,:,None,None]).sum(1)
        frame_variance = F.softplus(self.frame_log_variance(frame_indices)).view(batch,1,1) + self.epsilon
        total = pixel_variance + frame_variance
        return {"pixel_variance": pixel_variance, "frame_variance": frame_variance, "total_variance": total}
