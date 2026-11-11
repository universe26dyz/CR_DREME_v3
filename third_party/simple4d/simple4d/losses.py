"""Regularisers of the deformation field."""
from __future__ import annotations

import torch


def jacobian_matrix(input_coords, output, add_identity=True):
    """d(output)/d(input_coords) for [B, N, 3] tensors -> [B, N, 3, 3] (plus identity)."""
    B, N, _ = input_coords.shape
    jac = torch.zeros(B, N, 3, 3, device=input_coords.device)
    for i in range(3):
        grad = torch.autograd.grad(outputs=output[:, :, i], inputs=input_coords,
                                   grad_outputs=torch.ones_like(output[:, :, i]),
                                   create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]
        if grad is not None:
            jac[:, :, i, :] = grad
            if add_identity:
                jac[:, :, i, i] += 1
    return jac


def jacobian_loss(input_coords, dvf):
    """Mean of d + 1/d - 2 with d = det(I + grad DVF) clamped to [0.1, 10].

    A weak prior against folding (not a volume-preservation constraint).
    ``input_coords`` must be the tensor the DVF was computed from.
    """
    j = jacobian_matrix(input_coords, dvf)
    a, b, c = j[:, :, 0, 0], j[:, :, 0, 1], j[:, :, 0, 2]
    d, e, f = j[:, :, 1, 0], j[:, :, 1, 1], j[:, :, 1, 2]
    g, h, i = j[:, :, 2, 0], j[:, :, 2, 1], j[:, :, 2, 2]
    det = torch.clamp(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g), min=0.1, max=10)
    return torch.mean(det - 1 + torch.ones_like(det) / det - 1)


def dvf_magnitude_loss(dvf):
    """Mean squared displacement (in normalised coordinates)."""
    return torch.mean(dvf ** 2)
