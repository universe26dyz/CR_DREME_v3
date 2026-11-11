"""Fit SIMPLE-4D to one scan.

Example:
    python train.py --data scan.npz --out_dir runs/scan01 --n_holdout 64
"""
from __future__ import annotations

import json
import os
import time

import torch
import yaml

from simple4d.config import parse_args
from simple4d.coords import seed_worker
from simple4d.data import SliceDataset, SliceSeries
from simple4d.evaluation import heldout_metrics
from simple4d.model import SIMPLE4D


def main(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    series = SliceSeries.load(cfg.data, cfg.depth_crop)
    train_idx, held_idx = series.split(cfg.n_holdout)
    H, W = series.frames.shape[1:]
    print(f"{cfg.data}: {len(series)} frames, {series.num_positions} slice positions, {H}x{W} pixels, "
          f"spacing {series.spacing} mm; {len(train_idx)} for optimisation, {len(held_idx)} held out")

    g = torch.Generator()
    g.manual_seed(cfg.loader_seed)
    loader = torch.utils.data.DataLoader(
        SliceDataset(series, train_idx), batch_size=cfg.batch_size, shuffle=True, generator=g,
        worker_init_fn=seed_worker, num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0)

    model = SIMPLE4D(cfg, H, W, series.num_positions, series.spacing, series.depth_crop)
    with open(os.path.join(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump({k: v for k, v in vars(cfg).items() if k != "config"}, f, sort_keys=True)

    t0, step = time.time(), 1
    while step <= cfg.iters:
        for batch in loader:
            log = model.training_step(batch, step)
            if step % cfg.log_every == 0:
                print(f"step {step:6d} | " + " ".join(f"{k} {v:.5f}" for k, v in log.items())
                      + f" | {time.time() - t0:.0f} s", flush=True)
            if cfg.save_every and step % cfg.save_every == 0:
                model.save(os.path.join(cfg.out_dir, f"model_{step:06d}.pth"))
            step += 1
            if step > cfg.iters:
                break
    model.save(os.path.join(cfg.out_dir, "model_final.pth"))
    print(f"training finished in {(time.time() - t0) / 60:.1f} min")

    res = heldout_metrics(model, series, held_idx)
    with open(os.path.join(cfg.out_dir, "heldout_metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    s = res["summary"]
    print(f"held-out self-consistency (V1): PSNR {s['v1_psnr']:.2f} dB, SSIM {s['v1_ssim']:.3f}; "
          f"prediction with target withheld (V2): PSNR {s['v2_psnr']:.2f} dB, SSIM {s['v2_ssim']:.3f}")


if __name__ == "__main__":
    main(parse_args())
