"""cGAN generator and discriminator for 11x11 images.

The generator produces fake images from noise + class label.
The discriminator classifies real/fake AND predicts the disease label.
After GAN training, the discriminator's conv features are reused
as a feature extractor for a separate classifier head.
"""

import torch
import torch.nn as nn

from ..config import IMG_SIZE

N_PIXELS = IMG_SIZE * IMG_SIZE


class Generator(nn.Module):
    """Conditional generator: noise + label -> 11x11 image."""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.GELU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.GELU(),
            nn.Linear(512, N_PIXELS), nn.Tanh())

    def forward(self, z, labels):
        """z: (batch, latent_dim), labels: (batch,) -> (batch, 1, 11, 11)"""
        x = torch.cat([z, labels.unsqueeze(1)], dim=1)
        return self.net(x).view(-1, 1, IMG_SIZE, IMG_SIZE)


class Discriminator(nn.Module):
    """Discriminator with dual heads: real/fake + class prediction.

    The conv feature extractor (self.feat) is reused after GAN training
    as a backbone for classification via a separate head.
    """

    REP_DIM = 64

    def __init__(self):
        super().__init__()
        # Feature extractor: shared between real/fake and class heads
        self.feat = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())

        # Real/fake head
        self.real_head = nn.Linear(64, 1)
        # Class prediction head
        self.class_head = nn.Linear(64, 1)

    def forward(self, x):
        """Returns (real_logit, class_logit, features)."""
        f = self.feat(x)
        return self.real_head(f), self.class_head(f), f

    def extract(self, x):
        """64-dim feature vector from the conv backbone."""
        return self.feat(x)
