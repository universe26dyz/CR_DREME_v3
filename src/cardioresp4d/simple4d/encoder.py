"""Upstream SIMPLE-4D: https://github.com/ttlmh/simple-4d
commit c466b276a438b3e052ad3b44d38f398872d8aeae; vendored at third_party/simple4d/.
Project adaptation: view conditioning of the upstream motion encoder.
"""
import torch
from torch import nn
from third_party.simple4d.simple4d.networks import MotionCodeVAE


class Simple4DViewConditionedMotionCodeVAE(MotionCodeVAE):
    def __init__(self):
        super().__init__(code_dim=32, base_channels=32, pos_embed_dim=32, dropout=0.1)
        self.view_embedding = nn.Embedding(3, 8)
        self.fc_hidden[0] = nn.Linear(296, 144)

    def encode_distribution(self, image, normalized_slice_position, view_id):
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError('image must be [B,1,H,W]')
        b = image.shape[0]
        position = normalized_slice_position.reshape(-1)
        if position.shape != (b,) or not torch.isfinite(position).all() or torch.any(position.abs() > 1):
            raise ValueError('position must be finite [B] in [-1,1]')
        if view_id.shape != (b,) or view_id.dtype != torch.long or torch.any((view_id < 0) | (view_id > 2)):
            raise ValueError('view_id must be [B] with values 0, 1, 2')
        features = torch.cat((self.image_encoder(image), self.pos_embedding(position.float()), self.view_embedding(view_id)), dim=-1)
        h = self.fc_hidden(features)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def encode_code(self, image, normalized_slice_position, view_id, *, sample):
        # Disable only encoder dropout for repeatable posterior-mean diagnostics.
        was_training = self.training
        if not sample:
            self.eval()
        try:
            mu, logvar = self.encode_distribution(image, normalized_slice_position, view_id)
        finally:
            self.train(was_training)
        code = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return {'code': code, 'mu': mu, 'logvar': logvar}

    def forward(self, image, normalized_slice_position, view_id):
        return self.encode_code(image, normalized_slice_position, view_id, sample=self.training)
