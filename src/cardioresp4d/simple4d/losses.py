"""Upstream SIMPLE-4D: https://github.com/ttlmh/simple-4d
commit c466b276a438b3e052ad3b44d38f398872d8aeae; vendored at third_party/simple4d/.
Project adaptation: normalized-coordinate loss and millimetre diagnostics.
"""
import torch
import torch.nn.functional as F
from third_party.simple4d.simple4d.losses import jacobian_loss, dvf_magnitude_loss
from third_party.simple4d.simple4d.networks import MotionCodeVAE


def respiratory_loss(prediction, target, mu, logvar, points_normalized, forward_deformation, inverse_deformation, code, step):
    if not points_normalized.requires_grad:
        raise ValueError('normalized coordinates must require gradients')
    _, dvf = forward_deformation.forward_normalized(points_normalized, code)
    canonical = points_normalized + dvf
    xc = canonical.detach().clone().requires_grad_(True)
    _, inverse_dvf = inverse_deformation.forward_normalized(xc, code)
    back = xc + inverse_dvf
    photo = F.l1_loss(prediction, target)
    kl = MotionCodeVAE.kl_loss(mu, logvar)
    jac = jacobian_loss(points_normalized, dvf)
    mag = dvf_magnitude_loss(dvf)
    cycle = F.l1_loss(back, points_normalized)
    error_mm = forward_deformation.coordinates.normalized_deformation_to_world_mm(back - points_normalized).abs().mean()
    total = photo + 1e-3 * kl + min(0.2 * step, 10) * (1e-4 * jac + 1e-2 * mag) + 0.1 * cycle
    return {'total': total, 'photo': photo, 'kl': kl, 'jacobian': jac, 'magnitude': mag, 'cycle_loss_normalized': cycle, 'cycle_error_mm': error_mm, 'dvf_rms_mm': forward_deformation.coordinates.normalized_deformation_to_world_mm(dvf).square().mean().sqrt()}
