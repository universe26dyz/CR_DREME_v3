"""Upstream SIMPLE-4D: https://github.com/ttlmh/simple-4d
commit c466b276a438b3e052ad3b44d38f398872d8aeae; vendored at third_party/simple4d/.
Project adaptation: normalized SIREN displacement to external-world pullback.
"""
from torch import nn
from third_party.simple4d.simple4d.networks import DeformCodeAdaptive


class Simple4DRespiratoryDeformation(nn.Module):
    def __init__(self, coordinates):
        super().__init__()
        self.coordinates = coordinates
        self.network = DeformCodeAdaptive([3, 256, 256, 256, 256, 3], code_dim=32, omega=7, residual=False)

    def forward_normalized(self, points_normalized, code):
        raw = self.network(points_normalized, code)
        dvf = self.coordinates.raw_deformation_to_normalized(raw)
        return raw, dvf

    def forward(self, points_world_mm, respiratory_code):
        if points_world_mm.ndim != 3 or points_world_mm.shape[-1] != 3 or respiratory_code.shape != (points_world_mm.shape[0], 32):
            raise ValueError('expected points [B,N,3] and code [B,32]')
        normalized = self.coordinates.world_to_normalized(points_world_mm)
        raw, dvf = self.forward_normalized(normalized, respiratory_code)
        mm = self.coordinates.normalized_deformation_to_world_mm(dvf)
        return {'raw_dvf': raw, 'dvf_normalized': dvf, 'displacement_mm': mm, 'reference_points_mm': points_world_mm + mm, 'points_normalized': normalized}
