"""Evaluation on held-out frames.

V1  held-out slice self-consistency: the encoder sees the held-out slice; the slice
    at the same position is rendered from the estimate and compared with it.
V2  prediction with the target withheld: the code is interpolated from the
    temporally adjacent frames, z = (mu_{j-1} + mu_{j+1}) / 2.
Both use slice positions inside the depth crop. PSNR / MAE on [0, 1] intensities,
SSIM with a 7x7 window.

``dvf_regularity`` reports the Jacobian determinant of the forward field and the
inverse-consistency error in physical units (mm).
"""
from __future__ import annotations

import numpy as np
import torch
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

from .coords import index_to_norm


def slice_metrics(pred, gt):
    p = torch.from_numpy(np.asarray(pred))[None, None].float()
    g = torch.from_numpy(np.asarray(gt))[None, None].float()
    return dict(psnr=float(peak_signal_noise_ratio(p, g, data_range=1.0)),
                ssim=float(structural_similarity_index_measure(p, g, data_range=1.0, kernel_size=7)),
                mae=float((p - g).abs().mean()))


def heldout_metrics(model, series, held_idx):
    codes = model.encode_frames(series.frames, series.slice_index)
    D, (c0, c1) = series.num_positions, series.depth_crop
    held = [int(j) for j in held_idx if c0 <= series.slice_index[j] < D - c1]
    pos = series.slice_index[held]
    n = len(series)
    z2 = np.stack([0.5 * (codes[j - 1] + codes[j + 1]) if j + 1 < n else codes[j - 1] for j in held])
    p1 = model.render_slices(codes[held], pos)
    p2 = model.render_slices(z2, pos)
    rows = []
    for i, j in enumerate(held):
        r = dict(frame=j, slice_position=int(pos[i]))
        r.update({f"v1_{k}": v for k, v in slice_metrics(p1[i], series.frames[j]).items()})
        r.update({f"v2_{k}": v for k, v in slice_metrics(p2[i], series.frames[j]).items()})
        rows.append(r)
    keys = [k for k in rows[0] if k.startswith(("v1_", "v2_"))]
    return dict(summary={k: float(np.mean([r[k] for r in rows])) for k in keys}, n_frames=len(rows),
                per_frame=rows)


@torch.no_grad()
def dvf_regularity(model, code, chunk=400_000):
    """Jacobian determinant of the forward field (mm), folding fraction and inverse-consistency
    error (mm) on the full grid for one code."""
    model.eval_mode()
    h, w, d = model.h, model.w, model.d
    P, Hh, Ww = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    x = np.stack([index_to_norm(P, d), index_to_norm(Ww, w), index_to_norm(Hh, h)], -1).reshape(-1, 3)
    x = torch.from_numpy(x).float().to(model.device)
    sz, sr, sc = model.spacing
    half = torch.tensor([d / 2 * sz, w / 2 * sc, h / 2 * sr], device=model.device)
    z = torch.as_tensor(code).float().view(1, -1).to(model.device)
    u_mm, ice = [], []
    for i in range(0, len(x), chunk):
        xi = x[i:i + chunk][None]
        u = model.dvf(xi, z)
        u_mm.append((u[0] * half).cpu())
        if model.inverse_deform_net is not None:
            xt = xi + u
            ice.append(((xt + model.dvf(xt, z, inverse=True) - xi)[0] * half).norm(dim=-1).cpu())
    u = torch.cat(u_mm).numpy().reshape(d, h, w, 3).astype(np.float64)
    axis = {0: (0, sz), 1: (2, sc), 2: (1, sr)}           # component (d, w, h) -> array axis, spacing
    J = np.empty((d, h, w, 3, 3))
    for i in range(3):
        for j in range(3):
            a, s = axis[j]
            J[..., i, j] = np.gradient(u[..., i], s, axis=a) + (1.0 if i == j else 0.0)
    det = np.linalg.det(J)
    out = dict(jacobian_min=float(det.min()), jacobian_p001=float(np.percentile(det, 0.1)),
               folding_percent=float(100 * (det <= 0).mean()))
    if ice:
        e = torch.cat(ice).numpy()
        out.update(ice_mean_mm=float(e.mean()), ice_p99_mm=float(np.percentile(e, 99)))
    return out
