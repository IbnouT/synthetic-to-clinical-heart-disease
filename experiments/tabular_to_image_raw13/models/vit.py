"""ViT for 11x11 feature images.

4-layer, 4-head transformer with 64-dim embeddings. Each pixel is one token.
Backbone outputs the CLS token as a 64-dim representation.
Split into extract/forward for transfer experiments (same pattern as the CNN).

Representation: embed_dim (default 64)
Parameters: ~208K with defaults
"""

import torch
import torch.nn as nn

from ..config import IMG_SIZE

N_PIXELS = IMG_SIZE * IMG_SIZE


class PatchEmbed(nn.Module):
    """Projects each pixel to embed_dim, adds CLS token and positional embeddings."""

    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Linear(1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, N_PIXELS + 1, embed_dim) * 0.02)

    def forward(self, x):
        B = x.shape[0]
        x = x.flatten(2).transpose(1, 2)       # (B, 121, 1) — one scalar per pixel
        x = self.proj(x)                        # (B, 121, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls, x], dim=1) + self.pos_embed  # (B, 122, embed_dim)


class VisionTransformer(nn.Module):

    def __init__(self, embed_dim=64, n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        self.REP_DIM = embed_dim
        self.patch_embed = PatchEmbed(embed_dim)

        # Pre-norm transformer: LayerNorm before attention, more stable training
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=256,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Same purpose as in the CNN: separate head so we can swap it
        # for downstream classifiers during transfer experiments
        self.head = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def extract(self, x):
        """Backbone pass: returns CLS token as embed_dim representation."""
        x = self.patch_embed(x)
        x = self.encoder(x)
        return self.norm(x[:, 0])     # CLS token at position 0

    def forward(self, x):
        return self.head(self.extract(x))
