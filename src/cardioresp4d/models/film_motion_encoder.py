"""功能：single-frame、full-geometry conditioned FiLM encoder，预测 DREME 12 scores。
论文来源：S2V-DREME FiLM score inference；FiLM gamma/beta primitive。
输入：one acquired image [B,1,H,W] 和 center/row/column/normal/spacing/thickness geometry。
输出：resp_scores [B,3,3]、card_scores [B,1,3]。
主要步骤：CNN image feature；geometry Fourier encoding→MLP gamma/beta；FiLM modulation→score heads。
是否属于原论文直接实现 / 必要适配 / 可选实验：FiLM direct primitive；asynchronous single-view geometry conditioning is necessary adaptation。
命令行使用示例：由 dynamic training forward 导入，无独立 CLI。
"""
from __future__ import annotations
import math
import torch
from torch import nn
from cardioresp4d.adapters.film import FiLMAdapter

class GeometryFiLMMotionEncoder(nn.Module):
    def __init__(self, channels: int = 24, *, canonical_lower_world_mm: torch.Tensor | None = None, canonical_upper_world_mm: torch.Tensor | None = None, position_bands: int = 4) -> None:
        """Condition a FiLM score head on physical geometry without mixing units.

        # 参考源码（NISF++，commit 1f6f9b3feba7c3c757d1f4111418415cd1cafeaa）：
        # https://github.com/NILOIDE/CMR_representations/blob/1f6f9b3feba7c3c757d1f4111418415cd1cafeaa/pos_encoding.py
        # 参考源码（Nerfstudio，commit 50e0e3c70c775e89333256213363badbf074f29d）：
        # https://github.com/nerfstudio-project/nerfstudio/blob/50e0e3c70c775e89333256213363badbf074f29d/nerfstudio/data/scene_box.py
        # 本项目适配：仅将 world-mm center 用 canonical full-FOV 映射到 [-1,1]
        # 后做 per-dimension NeRF Fourier encoding；方向保持无量纲 unit-vector，
        # spacing/thickness 是相对 canonical extent 的低频 acquisition scalar。
        """
        super().__init__()
        if position_bands <= 0:
            raise ValueError("position_bands must be positive")
        lower = torch.full((3,), -1.0) if canonical_lower_world_mm is None else torch.as_tensor(canonical_lower_world_mm, dtype=torch.float32)
        upper = torch.full((3,), 1.0) if canonical_upper_world_mm is None else torch.as_tensor(canonical_upper_world_mm, dtype=torch.float32)
        if lower.shape != (3,) or upper.shape != (3,) or torch.any(upper <= lower):
            raise ValueError("canonical full-FOV bounds must be ordered [3] millimetres")
        self.register_buffer("canonical_lower_world_mm", lower)
        self.register_buffer("canonical_upper_world_mm", upper)
        self.position_bands = position_bands
        self.position_feature_dim = 3 * (1 + 2 * position_bands)
        self.image=nn.Sequential(nn.Conv2d(1,channels,3,padding=1),nn.ReLU(),nn.Conv2d(channels,channels,3,padding=1),nn.ReLU()); self.film=FiLMAdapter()
        geo_dim = self.position_feature_dim + 9 + 3
        self.geometry_mlp=nn.Sequential(nn.Linear(geo_dim,channels*2),nn.ReLU(),nn.Linear(channels*2,channels*2))
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(channels,channels),nn.ReLU()); self.resp=nn.Linear(channels,9); self.card=nn.Linear(channels,3)
    def forward(self,image:torch.Tensor,*,center_mm:torch.Tensor,row_direction:torch.Tensor,column_direction:torch.Tensor,normal:torch.Tensor,pixel_spacing_mm:torch.Tensor,slice_thickness_mm:torch.Tensor)->dict[str,torch.Tensor]:
        if image.ndim==3: image=image[:,None]
        batch=image.shape[0]
        if image.ndim!=4 or image.shape[1]!=1 or any(x.shape!=(batch,3) for x in (center_mm,row_direction,column_direction,normal)) or pixel_spacing_mm.shape!=(batch,2) or slice_thickness_mm.shape!=(batch,1): raise ValueError('image [B,1,H,W] and full batched geometry required')
        encoded = self.encode_geometry(center_mm, row_direction, column_direction, normal, pixel_spacing_mm, slice_thickness_mm)
        gamma,beta=self.geometry_mlp(encoded).chunk(2,-1); feat=self.image(image); feat=self.film(feat,1+gamma,beta); feature=self.head(feat)
        return {'resp_scores':self.resp(feature).reshape(batch,3,3),'card_scores':self.card(feature).reshape(batch,1,3)}
    def encode_geometry(self, center_mm: torch.Tensor, row_direction: torch.Tensor, column_direction: torch.Tensor, normal: torch.Tensor, pixel_spacing_mm: torch.Tensor, slice_thickness_mm: torch.Tensor) -> torch.Tensor:
        directions = (row_direction, column_direction, normal)
        norms = [torch.linalg.vector_norm(vector, dim=-1) for vector in directions]
        if any(not torch.allclose(norm, torch.ones_like(norm), atol=1e-4, rtol=0.) for norm in norms):
            raise ValueError("DICOM row/column/normal directions must be unit vectors")
        if (torch.abs((row_direction * column_direction).sum(-1)) > 1e-4).any() or (torch.abs((row_direction * normal).sum(-1)) > 1e-4).any() or (torch.abs((column_direction * normal).sum(-1)) > 1e-4).any():
            raise ValueError("DICOM row/column/normal directions must be orthogonal")
        if torch.any(pixel_spacing_mm <= 0) or torch.any(slice_thickness_mm <= 0):
            raise ValueError("pixel spacing and slice thickness must be positive")
        position = 2. * (center_mm - self.canonical_lower_world_mm.to(center_mm)) / (self.canonical_upper_world_mm - self.canonical_lower_world_mm).to(center_mm) - 1.
        bands=(2.0**torch.arange(self.position_bands, device=center_mm.device, dtype=center_mm.dtype))*math.pi
        phase = position[..., None] * bands
        position_encoded = torch.cat((position, torch.sin(phase).flatten(-2), torch.cos(phase).flatten(-2)), dim=-1)
        extent = (self.canonical_upper_world_mm - self.canonical_lower_world_mm).to(center_mm)
        acquisition = torch.log(torch.cat((pixel_spacing_mm / extent[:2], slice_thickness_mm / extent.mean()), dim=-1).clamp_min(torch.finfo(center_mm.dtype).eps))
        return torch.cat((position_encoded, row_direction, column_direction, normal, acquisition), dim=-1)
