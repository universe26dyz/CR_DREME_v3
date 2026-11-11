"""SIMPLE-4D model: canonical anatomy, motion encoder, bidirectional deformation,
thick-slice forward model, training step and inference."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import losses
from .coords import chunks, make_coords, make_slice_coords, sample_points, weight_ramp
from .networks import InstantNGPINR, MotionCodeEncoder, MotionCodeVAE, SirenRes, make_deformation_network

# five-point Gauss-Legendre quadrature on [-1, 1]
GL_NODES = (-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459)
GL_WEIGHTS = (0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851)


class SIMPLE4D:
    """Patient-specific model for one scan.

    Args:
        cfg: configuration (see ``simple4d.config``).
        h, w: in-plane size of the slices; d: number of slice positions.
        spacing: (slice spacing, row spacing, column spacing) in mm.
        depth_crop: slice positions excluded from the photometric loss at the start / end.
    """

    def __init__(self, cfg, h, w, d, spacing, depth_crop=(0, 0)):
        self.cfg = cfg
        self.h, self.w, self.d = int(h), int(w), int(d)
        self.spacing = tuple(float(s) for s in spacing)
        self.depth_crop = tuple(int(c) for c in depth_crop)
        self.device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        # Network outputs are scaled to normalised coordinates by 2 / (h, w, d), element-wise.
        # (The channels are ordered as the coordinates (d, w, h); the scaling is therefore a fixed
        # per-channel factor. Kept as in the published implementation.)
        self.img_size = torch.FloatTensor((self.h, self.w, self.d)).to(self.device)
        self._build_networks()
        self._build_optimizer()
        self.step = 0

    # ------------------------------------------------------------------ setup
    def _build_networks(self):
        c = self.cfg
        if c.template == "ngp":
            self.feature_net = InstantNGPINR(c.ngp_levels, c.ngp_features_per_level, c.ngp_log2_hashmap_size,
                                             c.ngp_base_resolution, c.ngp_max_resolution, c.ngp_mlp).to(self.device)
        elif c.template == "siren":
            self.feature_net = SirenRes([3] + list(c.siren_layers) + [1], omega=c.siren_omega).to(self.device)
        else:
            raise ValueError(c.template)
        self.thick_slice = bool(c.thick_slice)
        if self.thick_slice:
            thk = c.slice_thickness if c.slice_thickness is not None else self.spacing[0]
            self.thickness_norm = 2.0 * float(thk) / (self.spacing[0] * self.d)
            nodes = torch.tensor(GL_NODES, device=self.device)
            weights = torch.tensor(GL_WEIGHTS, device=self.device)
            sigma = self.thickness_norm / 4.0                         # Gaussian slice profile
            self._z_offsets = nodes * (self.thickness_norm / 2)
            w = torch.exp(-0.5 * (self._z_offsets / sigma) ** 2) * weights
            self._thick_weights = w / w.sum()
        enc = MotionCodeVAE if c.encoder == "vae" else MotionCodeEncoder
        self.motion_encoder = enc(code_dim=c.code_dim, base_channels=c.encoder_channels,
                                  num_positions=self.d).to(self.device)
        layers = [3] + list(c.deform_layers) + [3]
        self.deform_net = make_deformation_network(c.conditioning, layers, c.code_dim, c.deform_omega,
                                                   c.deform_residual).to(self.device)
        self.inverse_deform_net = (make_deformation_network(c.conditioning, layers, c.code_dim, c.deform_omega,
                                                            c.deform_residual).to(self.device)
                                   if c.bidirectional else None)

    def _build_optimizer(self):
        c = self.cfg
        if c.template == "ngp":
            groups = [{"params": self.feature_net.encoding_parameters(), "lr": c.lr_hash},
                      {"params": self.feature_net.mlp_parameters(), "lr": c.lr_template}]
        else:
            groups = [{"params": self.feature_net.parameters(), "lr": c.lr_template}]
        groups += [{"params": self.motion_encoder.parameters(), "lr": c.lr_encoder},
                   {"params": self.deform_net.parameters(), "lr": c.lr_deform}]
        if self.inverse_deform_net is not None:
            groups.append({"params": self.inverse_deform_net.parameters(), "lr": c.lr_deform})
        self.opt = torch.optim.Adam(groups)
        self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=c.lr_decay_steps, gamma=c.lr_decay_factor)

    def networks(self):
        nets = dict(feature_net=self.feature_net, motion_encoder=self.motion_encoder, deform_net=self.deform_net)
        if self.inverse_deform_net is not None:
            nets["inverse_deform_net"] = self.inverse_deform_net
        return nets

    def train_mode(self, encoder_and_deform=True):
        self.feature_net.train()
        for name in ("motion_encoder", "deform_net", "inverse_deform_net"):
            net = getattr(self, name)
            if net is not None:
                net.train(encoder_and_deform)

    def eval_mode(self):
        for net in self.networks().values():
            net.eval()

    # ------------------------------------------------------------------ forward model
    def norm_position(self, slice_position):
        return 2 * slice_position.float() / max(self.d - 1, 1) - 1

    def encode(self, images, slice_positions):
        """images [B, H, W], slice_positions [B] (indices) -> (code, mu, logvar)."""
        return self.motion_encoder(images.to(self.device), self.norm_position(slice_positions.to(self.device)))

    def dvf(self, coords, code, inverse=False):
        """Displacement in normalised coordinates; forward = frame -> canonical."""
        net = self.inverse_deform_net if inverse else self.deform_net
        return net(coords, code) / self.img_size * 2

    def predict(self, coords, code):
        """Forward model at frame coordinates [B, N, 3]: returns predicted intensity, centre DVF
        and warped (canonical) centre coordinates."""
        dvf = self.dvf(coords, code)
        canonical = coords + dvf
        if not self.thick_slice:
            return self.feature_net(canonical), dvf, canonical
        B, N, _ = coords.shape
        S = self._z_offsets.shape[0]
        z = coords[:, :, 0:1] + self._z_offsets.view(1, 1, S)
        xs = torch.cat([z.unsqueeze(-1), coords[:, :, 1:].unsqueeze(2).expand(B, N, S, 2)], dim=-1)
        dvf_s = self.dvf(xs.reshape(B, N * S, 3), code).reshape(B, N, S, 3)       # deformation at every node
        vals = self.feature_net((xs + dvf_s).reshape(B, N * S, 3)).reshape(B, N, S, 1)
        return (vals * self._thick_weights.view(1, 1, S, 1)).sum(dim=2), dvf, canonical

    # ------------------------------------------------------------------ training
    def training_step(self, batch, step):
        """One optimisation step; returns a dict of scalar losses."""
        c = self.cfg
        self.step = step
        # every 10th iteration only the template is in training mode (dropout of the encoder off)
        self.train_mode(encoder_and_deform=(step % 10 != 0))
        self.opt.zero_grad()

        positions = batch["slice_position"].to(self.device)
        valid = (positions >= self.depth_crop[0]) & (positions <= self.d - self.depth_crop[1])
        gt = batch["gt"].to(self.device)
        coords = make_slice_coords(gt.shape, positions, self.w, self.d).to(self.device)
        x, target = sample_points(coords, c.num_points, gt)
        x = x.requires_grad_(True)

        code, mu, logvar = self.encode(batch["gt"], batch["slice_position"])
        pred, dvf, canonical = self.predict(x, code)

        loss_photo = F.l1_loss(pred[valid], target[valid])
        reg = 0.0
        if c.lambda_jac > 0:
            reg = reg + losses.jacobian_loss(x, dvf) * c.lambda_jac
        if c.lambda_mag > 0:
            reg = reg + losses.dvf_magnitude_loss(dvf) * c.lambda_mag
        loss = loss_photo + weight_ramp(step) * reg
        out = dict(photo=loss_photo.item())
        if c.encoder == "vae":
            kl = self.motion_encoder.kl_loss(mu, logvar) * c.kl_weight
            loss = loss + kl
            out["kl"] = kl.item()
        if self.inverse_deform_net is not None:
            # the inverse field learns to invert the forward field; canonical coordinates are detached
            xc = canonical.detach().clone().requires_grad_(True)
            back = xc + self.dvf(xc, code, inverse=True)
            cyc = F.l1_loss(back[valid], x[valid]) * c.lambda_cycle
            loss = loss + cyc
            out["cycle"] = cyc.item()
        loss.backward()
        self.opt.step()
        self.sched.step()
        out["total"] = loss.item()
        return out

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def encode_frames(self, frames, slice_positions, batch_size=32):
        """Posterior means (codes) of acquired frames [N, H, W] -> [N, code_dim]."""
        self.eval_mode()
        codes = []
        for i in range(0, len(frames), batch_size):
            f = torch.as_tensor(frames[i:i + batch_size])
            p = torch.as_tensor(slice_positions[i:i + batch_size])
            codes.append(self.encode(f, p)[0].float().cpu())
        return torch.cat(codes).numpy()

    @torch.no_grad()
    def render_slices(self, codes, slice_positions, batch_size=4):
        """Forward-model prediction of coronal slices for codes [N, C] at positions [N] -> [N, H, W]."""
        self.eval_mode()
        out = []
        for i in range(0, len(codes), batch_size):
            z = torch.as_tensor(codes[i:i + batch_size]).float().to(self.device)
            p = torch.as_tensor(slice_positions[i:i + batch_size]).long()
            x = make_slice_coords((z.shape[0], self.h, self.w), p, self.w, self.d).to(self.device)
            out.append(self.predict(x, z)[0].view(z.shape[0], self.h, self.w).float().cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def render_volume(self, code, n_chunks=110):
        """Estimated volume [h, w, d] for one code [C] (same forward model as for training)."""
        self.eval_mode()
        z = torch.as_tensor(code).float().view(1, -1).to(self.device)
        x = make_coords((1, self.h, self.w, self.d)).to(self.device)
        vals = torch.cat([self.predict(xc, z)[0] for xc in chunks(x, n_chunks)], dim=1)
        return vals.view(self.d, self.w, self.h).permute(2, 1, 0).float().cpu().numpy()

    @torch.no_grad()
    def template_volume(self, n_chunks=110):
        """Canonical anatomy f_theta sampled on the grid [h, w, d] (point queries)."""
        self.eval_mode()
        x = make_coords((1, self.h, self.w, self.d)).to(self.device)
        vals = torch.cat([self.feature_net(xc) for xc in chunks(x, n_chunks)], dim=1)
        return vals.view(self.d, self.w, self.h).permute(2, 1, 0).float().cpu().numpy()

    @torch.no_grad()
    def dvf_volume_mm(self, code, inverse=False, n_chunks=110):
        """Displacement field in millimetres on the grid, [h, w, d, 3] with components
        (cranio-caudal = row, left-right = column, slice normal = depth).

        forward (inverse=False): frame -> canonical; inverse: canonical -> frame.
        """
        self.eval_mode()
        z = torch.as_tensor(code).float().view(1, -1).to(self.device)
        x = make_coords((1, self.h, self.w, self.d)).to(self.device)
        u = torch.cat([self.dvf(xc, z, inverse=inverse) for xc in chunks(x, n_chunks)], dim=1)[0]
        u = u.view(self.d, self.w, self.h, 3).permute(2, 1, 0, 3).float().cpu().numpy()
        half = (self.d / 2 * self.spacing[0], self.w / 2 * self.spacing[2], self.h / 2 * self.spacing[1])
        return u[..., [2, 1, 0]] * [half[2], half[1], half[0]]

    # ------------------------------------------------------------------ checkpoints
    def save(self, path):
        torch.save(dict(step=self.step, **{k: v.state_dict() for k, v in self.networks().items()}), path)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        for k, net in self.networks().items():
            net.load_state_dict(state[k])
        self.step = int(state.get("step", 0))
        return self


def load_run(run_dir, series, checkpoint="model_final.pth", device=None):
    """Rebuild the model of a finished run for ``series`` and load its weights."""
    import os
    from .config import load_config
    cfg = load_config(os.path.join(run_dir, "config.yaml"), **({} if device is None else {"device": device}))
    H, W = series.frames.shape[1:]
    model = SIMPLE4D(cfg, H, W, series.num_positions, series.spacing, series.depth_crop)
    return model.load(os.path.join(run_dir, checkpoint)), cfg
