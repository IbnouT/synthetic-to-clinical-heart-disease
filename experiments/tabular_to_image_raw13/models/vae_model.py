"""VAE for 11x11 images (64-dim latent representation).

Encoder compresses the image to a 64-dim latent via mu/logvar.
Decoder reconstructs the image from the latent (used during pretraining).
Classifier head maps the latent mu to a binary prediction (used after pretraining).
"""

import torch
import torch.nn as nn

from ..config import IMG_SIZE

N_PIXELS = IMG_SIZE * IMG_SIZE


class VAEModel(nn.Module):

    REP_DIM = 64

    def __init__(self, latent_dim=64):
        super().__init__()
        # Encoder: conv layers -> global pool -> mu and logvar projections
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 2), nn.BatchNorm2d(128), nn.GELU(), nn.Flatten())
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_lv = nn.Linear(128, latent_dim)

        # Decoder: reconstruct the flattened image from latent
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.GELU(),
            nn.Linear(128, 256), nn.GELU(),
            nn.Linear(256, N_PIXELS))

        # Classification head on the latent mu
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1))

    def encode(self, x):
        """Returns (mu, logvar) from the encoder."""
        h = self.enc_conv(x)
        return self.fc_mu(h), self.fc_lv(h)

    def forward_vae(self, x):
        """Full VAE pass: encode -> sample -> decode. Used during pretraining."""
        mu, lv = self.encode(x)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return self.decoder(z).view(-1, 1, IMG_SIZE, IMG_SIZE), mu, lv

    def extract(self, x):
        """Backbone: returns 64-dim mu (no sampling, deterministic)."""
        mu, _ = self.encode(x)
        return mu

    def forward(self, x):
        """Classification: mu -> classifier head."""
        return self.classifier(self.extract(x))
