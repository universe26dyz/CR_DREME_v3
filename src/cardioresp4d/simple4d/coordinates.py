"""Upstream SIMPLE-4D: https://github.com/ttlmh/simple-4d
commit c466b276a438b3e052ad3b44d38f398872d8aeae; vendored at third_party/simple4d/.
Project adaptation: external patient-world millimetres to normalized coordinates.
"""
import torch
from torch import nn


class Simple4DWorldCoordinateAdapter(nn.Module):
    """Map external LPS xyz to network xyz, not upstream array-axis order.

    No LPS/RAS conversion occurs here; the canonical adapter owns that boundary.
    Reference spacing is explicitly supplied from the canonical protocol.
    """
    def __init__(self, lower_world_mm, upper_world_mm, reference_spacing_mm, *, canonical_spacing_mm=None):
        super().__init__()
        lower, upper, spacing = (torch.as_tensor(x, dtype=torch.float32) for x in (lower_world_mm, upper_world_mm, reference_spacing_mm))
        if any(x.shape != (3,) or not torch.isfinite(x).all() for x in (lower, upper, spacing)) or torch.any(upper <= lower) or torch.any(spacing <= 0):
            raise ValueError('bounds and spacing must be finite 3-vectors with positive extent and spacing')
        if canonical_spacing_mm is not None:
            canonical = torch.as_tensor(canonical_spacing_mm, dtype=spacing.dtype)
            if canonical.shape != (3,) or not torch.isfinite(canonical).all() or not torch.allclose(spacing, canonical, atol=1e-6, rtol=0):
                raise ValueError('reference spacing disagrees with canonical provenance')
        self.register_buffer('lower_world_mm', lower)
        self.register_buffer('extent_mm', upper - lower)
        self.register_buffer('reference_spacing_mm', spacing)

    @staticmethod
    def _check(x):
        if x.ndim < 1 or x.shape[-1] != 3 or not torch.isfinite(x).all():
            raise ValueError('coordinates must be finite [...,3]')
        return x

    def world_to_normalized(self, x):
        return 2 * (self._check(x) - self.lower_world_mm) / self.extent_mm - 1

    def normalized_to_world(self, x):
        return (self._check(x) + 1) * self.extent_mm / 2 + self.lower_world_mm

    def raw_deformation_to_normalized(self, raw):
        return 2 * self._check(raw) / (self.extent_mm / self.reference_spacing_mm)

    def normalized_deformation_to_world_mm(self, dvf):
        return self._check(dvf) * self.extent_mm / 2

    def raw_deformation_to_world_mm(self, raw):
        return self.normalized_deformation_to_world_mm(self.raw_deformation_to_normalized(raw))
