"""Patient-world-mm adapter for NeSVoR's pinned official ``INR`` class."""
from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager
import torch
from torch import nn
from ._upstream import add_upstream_to_path

add_upstream_to_path("NeSVoR")
import nesvor.inr.models as nesvor_models  # noqa: E402
from nesvor.inr.models import INR, NeSVoR  # noqa: E402


def detect_checkpoint_encoding_backend(state_dict: dict[str, torch.Tensor]) -> str:
    """Identify legacy schema-1 encoding layout without converting weights."""
    keys = set(state_dict)
    tinycudann = "canonical.inr.encoding.params" in keys
    torch_hash = "canonical.inr.encoding.box_offsets" in keys or any(key.startswith("canonical.inr.encoding.embeddings.") for key in keys)
    if tinycudann and torch_hash:
        raise ValueError("ambiguous checkpoint canonical encoding backend")
    if tinycudann:
        return "tinycudann"
    if torch_hash:
        return "torch_hash"
    raise ValueError("cannot determine checkpoint canonical encoding backend")


@contextmanager
def _upstream_encoding_backend(backend: str):
    """Temporarily select the pinned upstream constructor backend only."""
    if backend not in {"torch_hash", "tinycudann"}:
        raise ValueError("canonical encoding backend must be torch_hash or tinycudann")
    if backend == "tinycudann" and nesvor_models.USE_TORCH:
        raise RuntimeError("checkpoint requires tinycudann canonical encoding but tinycudann is unavailable")
    previous = nesvor_models.USE_TORCH
    nesvor_models.USE_TORCH = backend == "torch_hash"
    try:
        yield
    finally:
        nesvor_models.USE_TORCH = previous


class NeSVoRCanonicalAdapter(nn.Module):
    """Directly wraps upstream ``INR``; NeSVoR normalises world-mm bounds itself."""

    def __init__(self, bounds_world_mm: torch.Tensor, *, coarsest_resolution: float = 8., finest_resolution: float = 1., level_scale: float = 1.5, n_features_per_level: int = 2, log2_hashmap_size: int = 19, n_features_z: int = 16, width: int = 64, depth: int = 2, spatial_scaling: float = 1., encoding_backend: str | None = None) -> None:
        super().__init__()
        if bounds_world_mm.shape != (2, 3) or not torch.isfinite(bounds_world_mm).all() or torch.any(bounds_world_mm[1] <= bounds_world_mm[0]):
            raise ValueError("bounds_world_mm must be finite [2,3] lower/upper bounds")
        self.args = Namespace(coarsest_resolution=coarsest_resolution, finest_resolution=finest_resolution, level_scale=level_scale, n_features_per_level=n_features_per_level, log2_hashmap_size=log2_hashmap_size, n_features_z=n_features_z, width=width, depth=depth, dtype=torch.float32, img_reg_autodiff=False)
        self.encoding_backend = ("torch_hash" if nesvor_models.USE_TORCH else "tinycudann") if encoding_backend is None else encoding_backend
        # Source-first resume needs the checkpoint's exact upstream encoding
        # layout; this changes construction selection only, never INR maths.
        with _upstream_encoding_backend(self.encoding_backend):
            self.inr = INR(bounds_world_mm.detach().clone().to(dtype=torch.float32), self.args, spatial_scaling)
        self.spatial_scaling = float(spatial_scaling)

    def forward(self, points_world_mm: torch.Tensor, return_features: bool = False):
        if points_world_mm.shape[-1] != 3 or not torch.isfinite(points_world_mm).all():
            raise ValueError("points_world_mm must be finite and end in xyz millimetres")
        # The pinned INR uses ``view`` internally; DICOM PSF/motion broadcasting
        # can produce a valid non-contiguous tensor, so the adapter materializes
        # only layout (not coordinates or INR math) at this interface.
        result = self.inr(points_world_mm.contiguous())
        if self.inr.training:
            density, _encoding, latent_z = result
            # NeSVoR's sigma_net concatenates its slice embedding with z[...,1:],
            # reserving z[...,0] for density before the official softplus.
            # Upstream preserves density prefix dimensions but keeps z flattened
            # for its own batch loss; restore only the public adapter layout.
            features = latent_z[..., 1:].reshape(*density.shape, latent_z.shape[-1] - 1)
            return (density, features) if return_features else density
        if return_features:
            raise RuntimeError("NeSVoR INR exposes latent z only in training mode")
        return result

    def image_regularization(self, density: torch.Tensor, sample_points_world_mm: torch.Tensor, *, mode: str = "edge", delta: float = 1.) -> torch.Tensor:
        """Delegate directly to pinned ``NeSVoR.img_reg`` without reimplementing it."""
        if mode not in {"edge", "TV", "L2", "none"} or delta <= 0:
            raise ValueError("invalid official NeSVoR image regularization mode or delta")
        context = Namespace(args=Namespace(image_regularization=mode, img_reg_autodiff=False), inr=self.inr, spatial_scaling=self.spatial_scaling, delta=float(delta))
        return NeSVoR.img_reg(context, density, sample_points_world_mm)
