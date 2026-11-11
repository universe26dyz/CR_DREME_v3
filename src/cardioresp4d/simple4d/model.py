"""Upstream SIMPLE-4D: https://github.com/ttlmh/simple-4d
commit c466b276a438b3e052ad3b44d38f398872d8aeae; vendored at third_party/simple4d/.
Project adaptation: NeSVoR canonical and PSF respiratory-only integration.
"""
import json
from pathlib import Path
import torch
from torch import nn
from cardioresp4d.adapters.nesvor_psf import NeSVoRPSFAdapter
from cardioresp4d.training.model import SourceFirstDynamicModel
from cardioresp4d.training.sampler import view_id_for
from .encoder import Simple4DViewConditionedMotionCodeVAE
from .deformation import Simple4DRespiratoryDeformation
from .coordinates import Simple4DWorldCoordinateAdapter


class Simple4DRespiratoryModel(nn.Module):
    def __init__(self, canonical, lower_world_mm, upper_world_mm, reference_spacing_mm, *, canonical_spacing_mm=None, psf_samples=8):
        super().__init__()
        self.canonical = canonical
        self.respiratory_encoder = Simple4DViewConditionedMotionCodeVAE()
        coordinates = Simple4DWorldCoordinateAdapter(lower_world_mm, upper_world_mm, reference_spacing_mm, canonical_spacing_mm=canonical_spacing_mm)
        self.forward_deformation = Simple4DRespiratoryDeformation(coordinates)
        self.inverse_deformation = Simple4DRespiratoryDeformation(Simple4DWorldCoordinateAdapter(lower_world_mm, upper_world_mm, reference_spacing_mm, canonical_spacing_mm=canonical_spacing_mm))
        self.psf = NeSVoRPSFAdapter(psf_samples)

    def freeze_r0(self):
        self.canonical.requires_grad_(False)
        self.respiratory_encoder.requires_grad_(True)
        self.forward_deformation.requires_grad_(True)
        self.inverse_deformation.requires_grad_(True)

    def predict(self, observation, pixel_uv):
        if observation.normalized_slice_position is None:
            raise ValueError('normalized slice position is required')
        image = observation.image.unsqueeze(0)
        encoded = self.respiratory_encoder(image, image.new_tensor([observation.normalized_slice_position]), torch.tensor([view_id_for(observation.view)], device=image.device))
        points = SourceFirstDynamicModel._pixel_world(observation, pixel_uv.to(image))
        count = points.shape[0]
        resolution = image.new_tensor([observation.pixel_spacing_mm[1], observation.pixel_spacing_mm[0], observation.slice_thickness_mm]).expand(count, 3)
        def motion(samples):
            return self.forward_deformation(samples, encoded['code'].expand(samples.shape[0], -1))
        render = self.psf(self.canonical, points, resolution, row_direction=observation.row_direction.to(image).expand(count, -1), column_direction=observation.column_direction.to(image).expand(count, -1), normal=observation.normal.to(image).expand(count, -1), motion=motion)
        render.update(encoded)
        return render

    def provenance(self):
        manifest = json.loads((Path(__file__).resolve().parents[3] / 'third_party/simple4d_VENDOR.json').read_text())
        return {'upstream': manifest, 'encoder_class': 'MotionCodeVAE', 'deformation_class': 'DeformCodeAdaptive', 'coordinate_policy': 'external LPS xyz mm to normalized xyz; canonical owns internal frame conversion', 'reference_spacing_mm': self.forward_deformation.coordinates.reference_spacing_mm.detach().cpu().tolist(), 'canonical_source': 'static NeSVoR', 'renderer_source': 'NeSVoRPSFAdapter'}
