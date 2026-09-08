"""功能：Stage 1A/1B/2/3 的最小 stage-aware data fidelity and motion regularizers。
论文来源：S2V-DREME warm start、NeSVoR Gaussian likelihood、DREME-MR score/MBC regularization。
输入：predicted/observed images、variance、mask、scores、DVF/INR grids和 Phase-1 timestamp priors。
输出：scalar differentiable losses；Stage 1A mask=0 produces exactly zero gradient。
主要步骤：masked MSE、Gaussian NLL、normalization/zero-mean/smoothness/TV和 irregular-time spectral leakage。
是否属于原论文直接实现 / 必要适配 / 可选实验：likelihood direct primitive；nonuniform timestamp DFT is necessary adaptation。
命令行使用示例：由训练 stage wrapper 导入，无独立 CLI。
"""
from __future__ import annotations
import math, torch

def masked_reference_mse(predicted:torch.Tensor,target:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
    if predicted.shape!=target.shape or mask.shape!=target.shape: raise ValueError('prediction/target/mask shapes must match')
    return (((predicted-target)**2)*mask).sum()/mask.sum().clamp_min(1.0)
def gaussian_nll(predicted:torch.Tensor,target:torch.Tensor,variance:torch.Tensor)->torch.Tensor:
    if predicted.shape!=target.shape or variance.shape!=target.shape or torch.any(variance<=0): raise ValueError('matching positive variance required')
    return 0.5*((predicted-target).square()/variance+variance.log()).mean()
def mbc_normalization(fields_mm:torch.Tensor)->torch.Tensor: return fields_mm.square().mean()
def zero_mean_scores(scores:torch.Tensor)->torch.Tensor: return scores.mean(dim=0).square().mean()
def dvf_smoothness(dvf_mm:torch.Tensor, spacing_mm:torch.Tensor)->torch.Tensor:
    if dvf_mm.ndim!=5 or dvf_mm.shape[-1]!=3 or spacing_mm.shape!=(3,): raise ValueError('dvf [B,D,H,W,3], spacing [3] required')
    return sum(((dvf_mm.diff(dim=axis)/spacing_mm[axis-1]).square().mean()) for axis in (1,2,3))
def inr_tv(values:torch.Tensor)->torch.Tensor: return sum(values.diff(dim=axis).abs().mean() for axis in range(1,values.ndim))
def frequency_leakage(scores:torch.Tensor,timestamps_s:torch.Tensor,forbidden_band_hz:tuple[float,float])->torch.Tensor:
    if scores.shape[0]!=timestamps_s.numel() or timestamps_s.ndim!=1: raise ValueError('time-leading scores and timestamps required')
    duration=(timestamps_s.max()-timestamps_s.min()).clamp_min(torch.finfo(timestamps_s.dtype).eps); frequencies=torch.arange(1,max(2,timestamps_s.numel()//2+1),device=timestamps_s.device,dtype=timestamps_s.dtype)/duration
    spectrum=torch.abs(torch.einsum('t...,tf->f...',scores-scores.mean(0),torch.exp(-2j*math.pi*timestamps_s[:,None]*frequencies[None])))**2
    forbidden=(frequencies>=forbidden_band_hz[0])&(frequencies<=forbidden_band_hz[1]); return spectrum[forbidden].mean()/(spectrum.mean()+torch.finfo(scores.dtype).eps)
