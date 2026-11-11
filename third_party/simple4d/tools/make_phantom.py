"""Synthetic breathing phantom in the SIMPLE-4D input format, for testing the pipeline.

A torso with two lungs, a liver with a lesion and a spine is deformed by an irregular
breathing signal (cranio-caudal motion growing towards the diaphragm, smaller
antero-posterior motion) and sampled as coronal slices swept cyclically through D
positions. The ground-truth volumes of the held-out frames are stored as well.

Example:
    python tools/make_phantom.py -o phantom.npz
    python train.py --data phantom.npz --out_dir runs/phantom --n_holdout 16 --iters 3000
"""
from __future__ import annotations

import argparse

import numpy as np


def anatomy(r, c, z):
    """Intensity at physical coordinates (mm): r cranio-caudal (down), c left-right, z antero-posterior."""
    v = np.zeros(np.broadcast(r, c, z).shape, np.float32)
    body = (c / 85) ** 2 + (z / 55) ** 2 < 1
    v[body] = 0.45
    for sgn in (-1, 1):
        lung = ((c - sgn * 40) / 30) ** 2 + ((r + 25) / 55) ** 2 + (z / 40) ** 2 < 1
        v[lung & body] = 0.08
    liver = ((c + 25) / 55) ** 2 + ((r - 55) / 35) ** 2 + (z / 42) ** 2 < 1
    liver &= ~((((c + 40) / 30) ** 2 + ((r + 25) / 55) ** 2 + (z / 40) ** 2) < 1)
    v[liver & body] = 0.72
    v[((c + 30) ** 2 + (r - 60) ** 2 + (z - 5) ** 2) < 9 ** 2] = 0.95                 # lesion
    v[(np.abs(c) < 9) & (z > 30) & (z < 48)] = 0.85                                    # spine
    return v


def breathing(t, rng):
    """Irregular breathing signal in [0, ~1.2] (period 3.5-5 s, varying amplitude)."""
    s, phase = np.zeros_like(t), 0.0
    period, amp, t_next = 4.0, 1.0, 0.0
    for i, ti in enumerate(t):
        if ti >= t_next:
            period, amp = rng.uniform(3.5, 5.0), rng.uniform(0.7, 1.2)
            t_next = ti + period
        phase += 2 * np.pi * (t[1] - t[0]) / period if i else 0.0
        s[i] = amp * 0.5 * (1 - np.cos(phase))
    return s


def volume(signal, H, W, D, spacing, amp_cc=15.0, amp_ap=4.0):
    dz, dr, dc = spacing
    r = (np.arange(H) - H / 2 + 0.5)[:, None, None] * dr
    c = (np.arange(W) - W / 2 + 0.5)[None, :, None] * dc
    z = (np.arange(D) - D / 2 + 0.5)[None, None, :] * dz
    profile = 1 / (1 + np.exp(-(r + 10) / 15))             # motion grows towards the diaphragm
    return anatomy(r - amp_cc * signal * profile, c, z + amp_ap * signal * profile)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", default="phantom.npz")
    p.add_argument("--size", type=int, nargs=2, default=(96, 96), metavar=("H", "W"))
    p.add_argument("--positions", type=int, default=16)
    p.add_argument("--sweeps", type=int, default=30)
    p.add_argument("--spacing", type=float, nargs=3, default=(8.0, 2.5, 2.0), metavar=("SLICE", "ROW", "COL"))
    p.add_argument("--frame_time", type=float, default=0.3, help="s per slice")
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    H, W = a.size
    D = a.positions
    N = D * a.sweeps
    t = np.arange(N) * a.frame_time
    s = breathing(t, rng)
    idx = np.arange(N) % D
    frames = np.empty((N, H, W), np.float32)
    for i in range(N):
        frames[i] = volume(s[i], H, W, D, a.spacing)[:, :, idx[i]]
    frames = np.clip(frames + a.noise * rng.standard_normal(frames.shape), 0, 1).astype(np.float32)
    gt = np.stack([volume(s[i], H, W, D, a.spacing) for i in range(N - D, N)])
    np.savez_compressed(a.output, frames=frames, slice_index=idx, spacing=np.array(a.spacing), num_positions=D,
                        depth_crop=np.array([0, 0]), breathing=s, gt_volumes_last_sweep=gt)
    print(f"{a.output}: {N} frames of {H}x{W}, {D} positions, {a.sweeps} sweeps")


if __name__ == "__main__":
    main()
