"""功能：single-frame、full-geometry conditioned FiLM encoder，预测 DREME 12 scores。
论文来源：S2V-DREME FiLM score inference；FiLM gamma/beta primitive。
输入：one acquired image [B,1,H,W] 和 center/row/column/normal/spacing/thickness geometry。
输出：resp_scores [B,3,3]、card_scores [B,1,3]。
主要步骤：CNN image feature；geometry Fourier encoding→MLP gamma/beta；FiLM modulation→score heads。
是否属于原论文直接实现 / 必要适配 / 可选实验：FiLM direct primitive；asynchronous single-view geometry conditioning is necessary adaptation。
命令行使用示例：由 dynamic training forward 导入，无独立 CLI。
"""
from __future__ import annotations
import math, torch
from torch import nn

class GeometryFiLMMotionEncoder(nn.Module):
    def __init__(self, channels:int=24, geometry_bands:int=4) -> None:
        super().__init__(); self.geometry_bands=geometry_bands
        self.image=nn.Sequential(nn.Conv2d(1,channels,3,padding=1),nn.ReLU(),nn.Conv2d(channels,channels,3,padding=1),nn.ReLU())
        geo_dim=15*(1+2*geometry_bands); self.geometry_mlp=nn.Sequential(nn.Linear(geo_dim,channels*2),nn.ReLU(),nn.Linear(channels*2,channels*2))
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(channels,channels),nn.ReLU()); self.resp=nn.Linear(channels,9); self.card=nn.Linear(channels,3)
    def forward(self,image:torch.Tensor,*,center_mm:torch.Tensor,row_direction:torch.Tensor,column_direction:torch.Tensor,normal:torch.Tensor,pixel_spacing_mm:torch.Tensor,slice_thickness_mm:torch.Tensor)->dict[str,torch.Tensor]:
        if image.ndim==3: image=image[:,None]
        batch=image.shape[0]
        if image.ndim!=4 or image.shape[1]!=1 or any(x.shape!=(batch,3) for x in (center_mm,row_direction,column_direction,normal)) or pixel_spacing_mm.shape!=(batch,2) or slice_thickness_mm.shape!=(batch,1): raise ValueError('image [B,1,H,W] and full batched geometry required')
        geometry=torch.cat((center_mm,row_direction,column_direction,normal,pixel_spacing_mm,slice_thickness_mm),-1); encoded=self._fourier(geometry)
        gamma,beta=self.geometry_mlp(encoded).chunk(2,-1); feat=self.image(image); feat=feat*(1+gamma[:,:,None,None])+beta[:,:,None,None]; feature=self.head(feat)
        return {'resp_scores':self.resp(feature).reshape(batch,3,3),'card_scores':self.card(feature).reshape(batch,1,3)}
    def _fourier(self,x:torch.Tensor)->torch.Tensor:
        bands=(2.0**torch.arange(self.geometry_bands,device=x.device,dtype=x.dtype))*math.pi
        value=x[...,None]*bands; return torch.cat((x,torch.sin(value).flatten(-2),torch.cos(value).flatten(-2)),-1)
