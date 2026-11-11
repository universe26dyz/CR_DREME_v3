"""Configuration. Defaults reproduce the configuration used for all results in the paper
(both the 1.5 T and the 0.5 T datasets); ablation variants are selected by single options."""
from __future__ import annotations

import argparse

import yaml


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"boolean expected, got {v!r}")


def build_parser():
    p = argparse.ArgumentParser(description="SIMPLE-4D", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=str, default=None, help="YAML file with option values (overridden by CLI)")
    g = p.add_argument_group("data")
    g.add_argument("--data", type=str, help="input scan (.npz, see simple4d/data.py)")
    g.add_argument("--out_dir", type=str, default="runs/simple4d")
    g.add_argument("--n_holdout", type=int, default=64,
                   help="number of final frames held out for evaluation (64 = one sweep at 1.5 T; 50 at 0.5 T)")
    g.add_argument("--depth_crop", type=int, nargs=2, default=None, metavar=("START", "END"),
                   help="slice positions excluded from the photometric loss (default: from the .npz)")
    g = p.add_argument_group("optimisation")
    g.add_argument("--iters", type=int, default=10000)
    g.add_argument("--batch_size", type=int, default=8, help="slices per iteration")
    g.add_argument("--num_points", type=int, default=10000, help="pixels sampled per slice")
    g.add_argument("--lr_hash", type=float, default=1e-2)
    g.add_argument("--lr_template", type=float, default=2e-5, help="template MLP (or SIREN template)")
    g.add_argument("--lr_encoder", type=float, default=5e-5)
    g.add_argument("--lr_deform", type=float, default=5e-5)
    g.add_argument("--lr_decay_steps", type=int, default=5000)
    g.add_argument("--lr_decay_factor", type=float, default=0.5)
    g.add_argument("--seed", type=int, default=6)
    g.add_argument("--loader_seed", type=int, default=12)
    g.add_argument("--num_workers", type=int, default=2)
    g.add_argument("--device", type=int, default=0)
    g.add_argument("--save_every", type=int, default=0, help="checkpoint interval (0: final only)")
    g.add_argument("--log_every", type=int, default=50)
    g = p.add_argument_group("canonical anatomy")
    g.add_argument("--template", choices=["ngp", "siren"], default="ngp")
    g.add_argument("--ngp_levels", type=int, default=16)
    g.add_argument("--ngp_features_per_level", type=int, default=2)
    g.add_argument("--ngp_log2_hashmap_size", type=int, default=19)
    g.add_argument("--ngp_base_resolution", type=int, default=16)
    g.add_argument("--ngp_max_resolution", type=int, default=512)
    g.add_argument("--ngp_mlp", type=int, nargs="+", default=[64, 64])
    g.add_argument("--siren_layers", type=int, nargs="+", default=[512, 512, 512, 512, 512])
    g.add_argument("--siren_omega", type=float, default=54)
    g = p.add_argument_group("motion encoder")
    g.add_argument("--encoder", choices=["vae", "deterministic"], default="vae")
    g.add_argument("--code_dim", type=int, default=32)
    g.add_argument("--encoder_channels", type=int, default=32)
    g.add_argument("--kl_weight", type=float, default=1e-3)
    g = p.add_argument_group("deformation")
    g.add_argument("--conditioning", choices=["adaptive", "concat", "film"], default="adaptive")
    g.add_argument("--deform_layers", type=int, nargs="+", default=[256, 256, 256, 256])
    g.add_argument("--deform_omega", type=float, default=7)
    g.add_argument("--deform_residual", type=str2bool, default=False)
    g.add_argument("--bidirectional", type=str2bool, default=True)
    g.add_argument("--lambda_cycle", type=float, default=0.1)
    g.add_argument("--lambda_jac", type=float, default=1e-4)
    g.add_argument("--lambda_mag", type=float, default=1e-2)
    g = p.add_argument_group("forward model")
    g.add_argument("--thick_slice", type=str2bool, default=True)
    g.add_argument("--slice_thickness", type=float, default=None,
                   help="slab width delta in mm (default: slice spacing, as in the paper)")
    return p


def parse_args(argv=None):
    p = build_parser()
    pre, _ = p.parse_known_args(argv)
    if pre.config:
        with open(pre.config) as f:
            values = yaml.safe_load(f) or {}
        unknown = set(values) - {a.dest for a in p._actions}
        if unknown:
            raise ValueError(f"unknown options in {pre.config}: {sorted(unknown)}")
        p.set_defaults(**values)
    return p.parse_args(argv)


def load_config(path, **overrides):
    """Configuration of a finished run (``config.yaml`` in its output directory)."""
    p = build_parser()
    with open(path) as f:
        p.set_defaults(**(yaml.safe_load(f) or {}))
    cfg = p.parse_args([])
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
