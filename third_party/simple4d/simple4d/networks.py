"""Network modules of SIMPLE-4D.

* Canonical anatomy f_theta: multi-resolution hash-encoded INR (``InstantNGPINR``)
  or, for the ablation, a residual SIREN (``SirenRes``).
* Motion encoder E_phi: single-slice CNN with slice-position embedding, either
  variational (``MotionCodeVAE``, default) or deterministic (``MotionCodeEncoder``).
* Deformation network D_psi: SIREN conditioned on the motion code, with
  additive code injection at hidden layers 1-3 (``DeformCodeAdaptive``, default)
  or with input concatenation / FiLM conditioning (``DeformCode``).

All coordinate inputs are [B, N, 3] in [-1, 1], ordered (depth, width, height).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

try:
    import tinycudann as tcnn
    HAS_TCNN = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_TCNN = False


# ════════════════════════════════════════════════════════════════════════════
#  SIREN building blocks
# ════════════════════════════════════════════════════════════════════════════
class Siren(nn.Module):
    """Fully connected network with sine activations (Sitzmann et al., 2020).

    Args:
        layers: widths of all layers, e.g. [3, 256, 256, 1].
        omega: frequency factor omega_0.
    """

    def __init__(self, layers, omega=30, **kwargs):
        super().__init__()
        self.n_layers = len(layers) - 1
        self.omega = omega
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            with torch.no_grad():
                if i == 0:
                    self.layers[-1].weight.uniform_(-1 / layers[i], 1 / layers[i])
                else:
                    self.layers[-1].weight.uniform_(-np.sqrt(6 / layers[i]) / self.omega,
                                                    np.sqrt(6 / layers[i]) / self.omega)
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.sin(self.omega * layer(x))
        return self.layers[-1](self.omega * x)


class SirenRes(Siren):
    """SIREN template with one residual connection and sigmoid output (ablation A1)."""

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            if i == 4:
                x = x + torch.sin(self.omega * layer(x))
            else:
                x = torch.sin(self.omega * layer(x))
        return torch.sigmoid(self.layers[-1](self.omega * x))


# ════════════════════════════════════════════════════════════════════════════
#  Canonical anatomy: multi-resolution hash-encoded INR
# ════════════════════════════════════════════════════════════════════════════
class MultiResolutionHashEncoding(nn.Module):
    """Pure-PyTorch multi-resolution hash encoding (fallback without tiny-cuda-nn).

    Note: the fallback is functionally equivalent but numerically different from
    the tiny-cuda-nn implementation used for all results in the paper, and much
    slower. Checkpoints are not interchangeable between the two.
    """

    def __init__(self, n_levels=16, n_features_per_level=2, log2_hashmap_size=19,
                 base_resolution=16, max_resolution=512):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.hashmap_size = 2 ** log2_hashmap_size
        growth = math.exp(math.log(max_resolution / base_resolution) / (n_levels - 1)) if n_levels > 1 else 1.0
        self.hash_tables = nn.ModuleList()
        self.resolutions = []
        for level in range(n_levels):
            res = int(math.floor(base_resolution * growth ** level))
            self.resolutions.append(res)
            table = nn.Embedding(min((res + 1) ** 3, self.hashmap_size), n_features_per_level)
            nn.init.uniform_(table.weight, -1e-4, 1e-4)
            self.hash_tables.append(table)
        self.output_dim = n_levels * n_features_per_level
        self.register_buffer("primes", torch.tensor([1, 2654435761, 805459861], dtype=torch.long))
        offsets = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.long)
        self.register_buffer("offsets", offsets)
        self.register_buffer("offsets_f", offsets.float())

    def _hash(self, c, table_size):
        h = c[..., 0] * self.primes[0]
        h = torch.bitwise_xor(h, c[..., 1] * self.primes[1])
        h = torch.bitwise_xor(h, c[..., 2] * self.primes[2])
        return h % table_size

    def forward(self, x):
        prefix = x.shape[:-1]
        x01 = ((x.reshape(-1, 3) + 1.0) * 0.5).clamp(0.0, 1.0)
        feats = []
        for level in range(self.n_levels):
            res = self.resolutions[level]
            table_size = self.hash_tables[level].weight.shape[0]
            g = x01 * res
            lo = g.long().clamp(0, res - 1)
            hi = (lo + 1).clamp(0, res)
            w = g - lo.float()
            corners = lo.unsqueeze(1) * (1 - self.offsets).unsqueeze(0) + hi.unsqueeze(1) * self.offsets.unsqueeze(0)
            cf = self.hash_tables[level](self._hash(corners, table_size))
            we = w.unsqueeze(1)
            tw = (we * self.offsets_f.unsqueeze(0) + (1 - we) * (1 - self.offsets_f.unsqueeze(0))).prod(-1, keepdim=True)
            feats.append((tw * cf).sum(1))
        return torch.cat(feats, -1).reshape(*prefix, self.output_dim)


class InstantNGPINR(nn.Module):
    """Hash-encoded INR: coordinates -> hash grid -> small ReLU MLP -> sigmoid (Mueller et al., 2022)."""

    def __init__(self, n_levels=16, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16,
                 max_resolution=512, mlp_hidden_dims=(64, 64), use_tcnn=True):
        super().__init__()
        mlp_hidden_dims = list(mlp_hidden_dims)
        enc_dim = n_levels * n_features_per_level
        self.use_tcnn = bool(use_tcnn and HAS_TCNN)
        if self.use_tcnn:
            self.encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid", "n_levels": n_levels, "n_features_per_level": n_features_per_level,
                    "log2_hashmap_size": log2_hashmap_size, "base_resolution": base_resolution,
                    "per_level_scale": math.exp(math.log(max_resolution / base_resolution) / max(n_levels - 1, 1)),
                })
            self.mlp = tcnn.Network(
                n_input_dims=enc_dim, n_output_dims=1,
                network_config={"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "Sigmoid",
                                "n_neurons": mlp_hidden_dims[0], "n_hidden_layers": len(mlp_hidden_dims)})
        else:
            self.encoding = MultiResolutionHashEncoding(n_levels, n_features_per_level, log2_hashmap_size,
                                                        base_resolution, max_resolution)
            layers, d_in = [], enc_dim
            for d_out in mlp_hidden_dims:
                layers += [nn.Linear(d_in, d_out), nn.ReLU(inplace=True)]
                d_in = d_out
            layers.append(nn.Linear(d_in, 1))
            self.mlp = nn.Sequential(*layers)

    def encoding_parameters(self):
        return self.encoding.parameters()

    def mlp_parameters(self):
        return self.mlp.parameters()

    def forward(self, x):
        prefix = x.shape[:-1]
        if self.use_tcnn:
            x01 = ((x.reshape(-1, 3) + 1.0) * 0.5).clamp(0.0, 1.0)
            return self.mlp(self.encoding(x01)).reshape(*prefix, 1).float()
        return torch.sigmoid(self.mlp(self.encoding(x)))


# ════════════════════════════════════════════════════════════════════════════
#  Motion encoder
# ════════════════════════════════════════════════════════════════════════════
class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal embedding of the slice-position index."""

    def __init__(self, embed_dim, max_positions=128):
        super().__init__()
        self.max_positions = max_positions
        position = torch.arange(max_positions).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_positions, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, positions):
        """positions: [B] normalised to [-1, 1] (float) or integer indices."""
        if positions.dim() == 2:
            positions = positions.squeeze(-1)
        if positions.dtype == torch.float:
            positions = ((positions + 1) / 2 * (self.max_positions - 1)).long()
        return self.pe[positions.clamp(0, self.max_positions - 1)]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                          nn.InstanceNorm2d(out_channels))
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.activation(out)


def _image_backbone(c):
    return nn.Sequential(
        nn.Conv2d(1, c, 7, 2, 3, bias=False), nn.InstanceNorm2d(c), nn.LeakyReLU(0.2, inplace=True),
        nn.MaxPool2d(2, 2),
        ResidualBlock(c, c * 2, stride=2), ResidualBlock(c * 2, c * 2),
        ResidualBlock(c * 2, c * 4, stride=2), ResidualBlock(c * 4, c * 4),
        ResidualBlock(c * 4, c * 8, stride=2), ResidualBlock(c * 8, c * 8),
        nn.AdaptiveAvgPool2d(1), nn.Flatten())


class MotionCodeVAE(nn.Module):
    """Variational single-slice motion encoder (default).

    forward(image [B, H, W], slice_position [B] in [-1, 1]) -> (code, mu, logvar).
    During training the code is sampled with the reparameterisation trick;
    in evaluation mode code = mu.
    """

    def __init__(self, code_dim=32, base_channels=32, pos_embed_dim=32, num_positions=64, dropout=0.1):
        super().__init__()
        self.code_dim = code_dim
        self.image_encoder = _image_backbone(base_channels)
        self.pos_embedding = SinusoidalPositionEmbedding(pos_embed_dim, num_positions)
        fusion_dim = base_channels * 8 + pos_embed_dim
        self.fc_hidden = nn.Sequential(nn.Linear(fusion_dim, fusion_dim // 2), nn.LayerNorm(fusion_dim // 2),
                                       nn.LeakyReLU(0.2), nn.Dropout(dropout))
        self.fc_mu = nn.Linear(fusion_dim // 2, code_dim)
        self.fc_logvar = nn.Linear(fusion_dim // 2, code_dim)

    def encode(self, image, slice_position):
        if image.dim() == 3:
            image = image.unsqueeze(1)
        h = self.fc_hidden(torch.cat([self.image_encoder(image), self.pos_embedding(slice_position)], -1))
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, image, slice_position):
        mu, logvar = self.encode(image, slice_position)
        if self.training:
            code = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            code = mu
        return code, mu, logvar

    @staticmethod
    def kl_loss(mu, logvar):
        """KL divergence to N(0, I), averaged over batch and latent dimensions."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class MotionCodeEncoder(nn.Module):
    """Deterministic single-slice motion encoder (ablation A2)."""

    def __init__(self, code_dim=32, base_channels=32, pos_embed_dim=32, num_positions=64, dropout=0.1):
        super().__init__()
        self.code_dim = code_dim
        self.image_encoder = _image_backbone(base_channels)
        self.pos_embedding = SinusoidalPositionEmbedding(pos_embed_dim, num_positions)
        self.pos_proj = nn.Sequential(nn.Linear(pos_embed_dim, pos_embed_dim), nn.LeakyReLU(0.2))
        fusion_dim = base_channels * 8 + pos_embed_dim
        self.fusion_mlp = nn.Sequential(nn.Linear(fusion_dim, fusion_dim // 2), nn.LayerNorm(fusion_dim // 2),
                                        nn.LeakyReLU(0.2), nn.Dropout(dropout), nn.Linear(fusion_dim // 2, code_dim))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image, slice_position):
        if image.dim() == 3:
            image = image.unsqueeze(1)
        pos = self.pos_proj(self.pos_embedding(slice_position))
        code = self.fusion_mlp(torch.cat([self.image_encoder(image), pos], -1))
        return code, None, None


# ════════════════════════════════════════════════════════════════════════════
#  Deformation network
# ════════════════════════════════════════════════════════════════════════════
class DeformCodeAdaptive(Siren):
    """SIREN deformation network with additive code injection (default).

    h_l = sin(omega W_l h_{l-1}) + tanh(Lin_l(z)) after hidden layers 1, 2 and 3;
    Tanhshrink output. ``residual=True`` adds identity skips between hidden
    layers of equal width (architecture study only).
    """

    def __init__(self, layers, code_dim=32, omega=30, residual=False, **kwargs):
        self.code_dim = code_dim
        self.residual = residual
        layers_mod = list(layers)
        layers_mod[0] = 3
        super().__init__(layers_mod, omega, **kwargs)
        self.injection_layers = [1, 2, 3]
        self.code_projectors = nn.ModuleList(
            [nn.Sequential(nn.Linear(code_dim, layers[i]), nn.Tanh()) for i in self.injection_layers])
        self.shrink = nn.Tanhshrink()

    def forward(self, x, motion_code):
        N = x.shape[1]
        codes = [p(motion_code).unsqueeze(1).expand(-1, N, -1) for p in self.code_projectors]
        k = 0
        for i, layer in enumerate(self.layers[:-1]):
            h = torch.sin(self.omega * layer(x))
            x = x + h if (self.residual and i > 0 and h.shape[-1] == x.shape[-1]) else h
            if i in self.injection_layers and k < len(codes):
                x = x + codes[k]
                k += 1
        return self.shrink(self.layers[-1](self.omega * x))


class DeformCode(Siren):
    """SIREN deformation network with concatenation or FiLM conditioning (ablation A3)."""

    def __init__(self, layers, code_dim=32, omega=30, use_film=False, **kwargs):
        self.code_dim = code_dim
        self.use_film = use_film
        layers = list(layers)
        layers[0] = 3 if use_film else 3 + code_dim
        super().__init__(layers, omega, **kwargs)
        if use_film:
            self.film_generators = nn.ModuleList(
                [nn.Linear(code_dim, layers[i] * 2) for i in range(1, len(layers) - 1)])
        self.shrink = nn.Tanhshrink()

    def forward(self, x, motion_code):
        N = x.shape[1]
        if not self.use_film:
            x = torch.cat([x, motion_code.unsqueeze(1).expand(-1, N, -1)], -1)
            for layer in self.layers[:-1]:
                x = torch.sin(self.omega * layer(x))
            return self.shrink(self.layers[-1](self.omega * x))
        film = [g(motion_code).chunk(2, dim=-1) for g in self.film_generators]
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if i < len(film):
                gamma, beta = film[i]
                x = gamma.unsqueeze(1).expand(-1, N, -1) * x + beta.unsqueeze(1).expand(-1, N, -1)
            x = torch.sin(self.omega * x)
        return self.shrink(self.layers[-1](self.omega * x))


def make_deformation_network(conditioning, layers, code_dim, omega, residual=False):
    if conditioning == "adaptive":
        return DeformCodeAdaptive(layers=list(layers), code_dim=code_dim, omega=omega, residual=residual)
    if conditioning in ("concat", "film"):
        return DeformCode(layers=list(layers), code_dim=code_dim, omega=omega, use_film=conditioning == "film")
    raise ValueError(f"unknown conditioning: {conditioning}")
