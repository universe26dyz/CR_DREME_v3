"""功能：NeSVoR-style anisotropic 3D Gaussian PSF slice renderer。
论文来源：NeSVoR PSF sampling；NiftyMIC oriented Gaussian FWHM cross-check。
输入：DICOM world-mm slice geometry、canonical INR、可选 observation-to-reference motion。
输出：differentiable PSF-integrated slice和每个 sample 的 latent/world coordinates。
主要步骤：以 in-plane FWHM=1.2*spacing、through-plane FWHM=thickness 构造 local covariance；每 sample warp 后 query INR。
是否属于原论文直接实现 / 必要适配 / 可选实验：NeSVoR/NiftyMIC physical primitive；world-space local axes 是必要 adaptation。
命令行使用示例：由训练 forward 导入，无独立 CLI。
"""
from __future__ import annotations
import math, torch
from torch import nn

_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))

def psf_sigma_mm(pixel_spacing_mm: torch.Tensor, thickness_mm: torch.Tensor) -> torch.Tensor:
    """Return [B,3] local (row-axis, column-axis, normal-axis) Gaussian sigmas in mm."""
    if pixel_spacing_mm.ndim != 2 or pixel_spacing_mm.shape[1] != 2 or thickness_mm.shape != (pixel_spacing_mm.shape[0],): raise ValueError('spacing [B,2] and thickness [B] required')
    return torch.cat((1.2 * pixel_spacing_mm, thickness_mm[:, None]), -1) * _FWHM_TO_SIGMA

def psf_covariance_mm2(row_direction: torch.Tensor, column_direction: torch.Tensor, normal: torch.Tensor, pixel_spacing_mm: torch.Tensor, thickness_mm: torch.Tensor) -> torch.Tensor:
    """Return the NiftyMIC-equivalent oriented world-mm Gaussian covariance [B,3,3]."""
    batch = row_direction.shape[0]
    if any(value.shape != (batch, 3) for value in (row_direction, column_direction, normal)): raise ValueError('axes must be [B,3]')
    axes = torch.stack((row_direction, column_direction, normal), -1)
    sigma2 = psf_sigma_mm(pixel_spacing_mm, thickness_mm).square()
    return axes @ torch.diag_embed(sigma2) @ axes.transpose(-1, -2)

class AnisotropicPSFRenderer(nn.Module):
    """Tensor-product three-node Gauss-Hermite approximation of an oriented Gaussian PSF."""
    def __init__(self, world_to_normalized: torch.Tensor) -> None:
        super().__init__(); transform = torch.as_tensor(world_to_normalized, dtype=torch.float32)
        if transform.shape != (4,4): raise ValueError('world_to_normalized must be 4x4')
        nodes, weights = torch.tensor([-1.2247448714, 0., 1.2247448714]), torch.tensor([0.2954089752, 1.1816359006, 0.2954089752]) / math.sqrt(math.pi)
        grid = torch.cartesian_prod(nodes, nodes, nodes); w = torch.cartesian_prod(weights, weights, weights).prod(-1)
        self.register_buffer('world_to_normalized', transform); self.register_buffer('standard_nodes', grid); self.register_buffer('weights', w / w.sum())

    def forward(self, inr: nn.Module, *, center_mm: torch.Tensor, row_direction: torch.Tensor, column_direction: torch.Tensor, normal: torch.Tensor, pixel_spacing_mm: torch.Tensor, thickness_mm: torch.Tensor, height: int, width: int, motion: nn.Module | None = None) -> dict[str, torch.Tensor]:
        batch = center_mm.shape[0]
        if any(t.shape != (batch,3) for t in (center_mm,row_direction,column_direction,normal)) or pixel_spacing_mm.shape != (batch,2) or thickness_mm.shape != (batch,): raise ValueError('invalid batched geometry')
        rows = torch.arange(height, device=center_mm.device, dtype=center_mm.dtype)-(height-1)/2; cols=torch.arange(width,device=center_mm.device,dtype=center_mm.dtype)-(width-1)/2
        rr, cc = torch.meshgrid(rows, cols, indexing='ij')
        plane = center_mm[:,None,None,:] + cc[None,:,:,None]*pixel_spacing_mm[:,None,None,1,None]*row_direction[:,None,None,:] + rr[None,:,:,None]*pixel_spacing_mm[:,None,None,0,None]*column_direction[:,None,None,:]
        sigma = psf_sigma_mm(pixel_spacing_mm.to(center_mm), thickness_mm.to(center_mm))
        axes = torch.stack((row_direction,column_direction,normal), -1)
        offsets = torch.einsum('bik,bsk->bsi', axes, self.standard_nodes[None].to(center_mm) * math.sqrt(2.0) * sigma[:,None,:])
        samples = plane[:,None] + offsets[:,:,None,None,:]
        reference = motion(samples)['reference_points_mm'] if motion is not None else samples
        normalized = self._normalize(reference); intensity, latent = inr(normalized, return_features=True)
        predicted = (intensity.squeeze(-1)*self.weights[None,:,None,None]).sum(1).unsqueeze(1)
        return {'predicted_slice': predicted, 'latent_samples': latent, 'sample_world_mm': samples, 'reference_world_mm': reference, 'query_normalized': normalized, 'psf_weights': self.weights, 'psf_sigma_mm': sigma}

    def _normalize(self, points_mm: torch.Tensor) -> torch.Tensor:
        one=torch.ones((*points_mm.shape[:-1],1),dtype=points_mm.dtype,device=points_mm.device); return torch.einsum('ij,...j->...i',self.world_to_normalized.to(points_mm),torch.cat((points_mm,one),-1))[...,:3]
