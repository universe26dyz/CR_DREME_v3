"""World-mm adapters around the vendored pinned SINR SIREN and FFD classes."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ._upstream import add_upstream_to_path

add_upstream_to_path("SINR")
from networks.networks import BSplineSiren  # noqa: E402
from models.transformation import CubicBSplineFFDTransform  # noqa: E402


def _triple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    return (value, value, value) if isinstance(value, int) else tuple(int(x) for x in value)


class SINRFFDBasis(nn.Module):
    """One vector-valued MBC using upstream SIREN and upstream cubic FFD.

    # 参考源码（SINR，commit 1a524ca7ae453b55310595fe957245088a108233）：
    # https://github.com/vasl12/SINR/blob/1a524ca7ae453b55310595fe957245088a108233/models/transformation.py
    # 本项目适配：DREME specifies logical control counts; SINR requires a dense
    # ``img_size`` and a ``cps``. ``logical + 2`` padded controls are selected
    # by dense_shape=logical*cps, preserving SINR's unmodified boundary crop.
    """
    def __init__(self, lower_world_mm: torch.Tensor, upper_world_mm: torch.Tensor, *, logical_control_shape: tuple[int, int, int] | None = None, grid_shape: tuple[int, int, int] | None = None, cps: int | tuple[int, int, int] = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        if logical_control_shape is None:
            if grid_shape is None:
                raise ValueError("logical_control_shape is required")
            logical_control_shape = grid_shape
        if grid_shape is not None and tuple(grid_shape) != tuple(logical_control_shape):
            raise ValueError("grid_shape alias conflicts with logical_control_shape")
        self.register_buffer("lower_world_mm", lower_world_mm.to(dtype=torch.float32))
        self.register_buffer("upper_world_mm", upper_world_mm.to(dtype=torch.float32))
        if self.lower_world_mm.shape != (3,) or self.upper_world_mm.shape != (3,) or torch.any(self.upper_world_mm <= self.lower_world_mm):
            raise ValueError("world bounds must be ordered [3] millimetres")
        self.logical_control_shape, self.cps = _triple(logical_control_shape), _triple(cps)
        if min(*self.logical_control_shape, *self.cps, hidden_dim) <= 0:
            raise ValueError("logical_control_shape, cps, and hidden_dim must be positive")
        self.dense_evaluation_shape = tuple(logical * spacing for logical, spacing in zip(self.logical_control_shape, self.cps))
        self.padded_control_shape = tuple(int(math.ceil(size / stride)) + 2 for size, stride in zip(self.dense_evaluation_shape, self.cps))
        self.register_buffer("grid_spacing_mm", (self.upper_world_mm - self.lower_world_mm) / torch.tensor([size - 1 for size in self.dense_evaluation_shape], dtype=torch.float32))
        # DREME/S2V contract: one MBC vector e_i(x)=[e_ix,e_iy,e_iz], not
        # three vectors mixed again by Cartesian scores.
        self.siren = BSplineSiren([3, hidden_dim, hidden_dim, 3])
        self.ffd = CubicBSplineFFDTransform(ndim=3, img_size=self.dense_evaluation_shape, cps=self.cps)

    @property
    def control_shape(self) -> tuple[int, int, int]:
        """Compatibility name; always means actual padded controls."""
        return self.padded_control_shape

    def control_parameters_grid_units(self, batch_size: int) -> torch.Tensor:
        axes = [torch.linspace(-1., 1., size, device=self.lower_world_mm.device, dtype=self.lower_world_mm.dtype) for size in self.padded_control_shape]
        coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        values = self.siren(coordinates).reshape(*self.padded_control_shape, 3).permute(3, 0, 1, 2).unsqueeze(0)
        return values.expand(batch_size, -1, -1, -1, -1)

    def dense_dvf_mm(self, controls_grid_units: torch.Tensor) -> torch.Tensor:
        if controls_grid_units.ndim != 5 or controls_grid_units.shape[1:] != (3, *self.padded_control_shape):
            raise ValueError("controls must be [B,3,*padded_control_shape] in upstream grid units")
        dense_grid_units = self.ffd(controls_grid_units)
        return dense_grid_units * self.grid_spacing_mm.view(1, 3, 1, 1, 1)

    def forward(self, points_world_mm: torch.Tensor) -> torch.Tensor:
        if points_world_mm.ndim != 3 or points_world_mm.shape[-1] != 3:
            raise ValueError("points_world_mm must be [B,N,3]")
        batch, count = points_world_mm.shape[:2]
        dense = self.dense_dvf_mm(self.control_parameters_grid_units(batch))
        normalized = 2. * (points_world_mm - self.lower_world_mm) / (self.upper_world_mm - self.lower_world_mm) - 1.
        grid = normalized[..., [2, 1, 0]].reshape(batch, count, 1, 1, 3)
        sampled = F.grid_sample(dense, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return sampled.reshape(batch, 3, count).permute(0, 2, 1)


class RespiratorySINRMBCAdapter(nn.Module):
    """DREME 8/12/16 logical respiratory MBC levels from source SINR."""
    def __init__(self, lower_world_mm: torch.Tensor, upper_world_mm: torch.Tensor, *, logical_control_shapes: tuple[tuple[int, int, int], ...] = ((8, 8, 8), (12, 12, 12), (16, 16, 16)), grid_shapes: tuple[tuple[int, int, int], ...] | None = None, cps: int | tuple[int, int, int] = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        if grid_shapes is not None:
            logical_control_shapes = grid_shapes
        if len(logical_control_shapes) != 3:
            raise ValueError("v3 respiratory branch requires exactly three levels")
        self.levels = nn.ModuleList([SINRFFDBasis(lower_world_mm, upper_world_mm, logical_control_shape=shape, cps=cps, hidden_dim=hidden_dim) for shape in logical_control_shapes])
        self.level_gates = nn.Parameter(torch.zeros(len(self.levels)))

    def activate_levels(self, active: int) -> None:
        """Start newly scheduled levels near zero without altering upstream networks."""
        if not 0 <= active <= len(self.levels):
            raise ValueError("invalid active respiratory level count")
        with torch.no_grad():
            gates = self.level_gates[:active]
            gates[gates == 0] = 1e-6

    def forward(self, points_world_mm: torch.Tensor) -> torch.Tensor:
        return torch.stack([gate * level(points_world_mm) for gate, level in zip(self.level_gates, self.levels)], dim=1)


class CardiacSINRMBCAdapter(nn.Module):
    """One local logical-16 cubic SINR MBC plus an adapter-only box taper."""
    def __init__(self, lower_world_mm: torch.Tensor, upper_world_mm: torch.Tensor, *, logical_control_shape: tuple[int, int, int] = (16, 16, 16), grid_shape: tuple[int, int, int] | None = None, cps: int | tuple[int, int, int] = 2, hidden_dim: int = 64, taper_mm: float = 4.) -> None:
        super().__init__()
        if grid_shape is not None:
            logical_control_shape = grid_shape
        if taper_mm <= 0:
            raise ValueError("taper_mm must be positive")
        self.basis = SINRFFDBasis(lower_world_mm, upper_world_mm, logical_control_shape=logical_control_shape, cps=cps, hidden_dim=hidden_dim)
        self.register_buffer("lower_world_mm", lower_world_mm.to(dtype=torch.float32)); self.register_buffer("upper_world_mm", upper_world_mm.to(dtype=torch.float32)); self.taper_mm = float(taper_mm)

    def forward(self, points_world_mm: torch.Tensor) -> torch.Tensor:
        field = self.basis(points_world_mm)
        distance = torch.minimum(points_world_mm - self.lower_world_mm, self.upper_world_mm - points_world_mm)
        taper = (distance / self.taper_mm).clamp(0., 1.).amin(dim=-1)
        return field * taper[..., None]
