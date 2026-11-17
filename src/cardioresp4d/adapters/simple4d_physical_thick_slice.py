"""Physical-plane counterpart of SIMPLE4D.predict's thick-slice quadrature."""
import torch
from torch import nn
from third_party.simple4d.simple4d.model import GL_NODES, GL_WEIGHTS


class PhysicalUpstreamThickSliceAdapter(nn.Module):
    """Integrate along the physical normal; no in-plane PSF is applied."""
    def sample(self, centers_world_mm, resolution_mm, *, row_direction, column_direction, normal):
        thickness = resolution_mm[:, 2]
        if torch.any(thickness <= 0) or not torch.isfinite(thickness).all():
            raise ValueError('slice thickness must be finite and positive')
        if normal.shape != centers_world_mm.shape or not torch.allclose(torch.linalg.vector_norm(normal, dim=-1), torch.ones_like(thickness), atol=1e-4, rtol=0):
            raise ValueError('normal must be a unit vector per pixel')
        nodes = centers_world_mm.new_tensor(GL_NODES)
        return centers_world_mm[:, None, :] + normal[:, None, :] * (thickness[:, None, None] * nodes[None, :, None] / 2)

    def forward(self, canonical, centers_world_mm, resolution_mm, *, row_direction, column_direction, normal, motion=None):
        samples = self.sample(centers_world_mm, resolution_mm, row_direction=row_direction, column_direction=column_direction, normal=normal)
        reference = samples if motion is None else motion(samples)['reference_points_mm']
        intensity, latent = canonical(reference, return_features=True)
        nodes = centers_world_mm.new_tensor(GL_NODES)
        weights = centers_world_mm.new_tensor(GL_WEIGHTS) * torch.exp(-0.5 * (nodes * 2) ** 2)
        weights = weights / weights.sum()
        return {'predicted_intensity': (intensity * weights).sum(-1), 'intensity_samples': intensity, 'latent_samples': latent, 'observation_samples_world_mm': samples, 'reference_samples_world_mm': reference}
