"""Hybrid CNN-Transformer (64-dim representation, ~200K params).

CNN front-end extracts local features from the 11x11 image, then a
transformer processes the flattened feature map with a CLS token.
Combines CNN's local feature extraction with transformer's global attention.
"""

import torch
import torch.nn as nn

from ..config import IMG_SIZE

N_PIXELS = IMG_SIZE * IMG_SIZE


class HybridCNNTransformer(nn.Module):

    def __init__(self, cnn_ch=64, n_heads=4, n_layers=3, dropout=0.15):
        super().__init__()
        self.REP_DIM = cnn_ch

        # CNN front-end: 2 conv layers to extract local spatial features
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, cnn_ch, 3, padding=1), nn.BatchNorm2d(cnn_ch), nn.GELU())

        # CLS token and positional embeddings for the transformer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cnn_ch))
        self.pos_embed = nn.Parameter(torch.randn(1, N_PIXELS + 1, cnn_ch) * 0.02)

        # Transformer processes CNN features + CLS token
        layer = nn.TransformerEncoderLayer(
            d_model=cnn_ch, nhead=n_heads, dim_feedforward=256,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(cnn_ch)

        # Classification head
        self.head = nn.Sequential(nn.Linear(cnn_ch, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def extract(self, x):
        """Backbone: CNN -> flatten -> transformer -> CLS token representation."""
        B = x.shape[0]
        x = self.cnn(x).flatten(2).transpose(1, 2)  # (B, N_PIXELS, cnn_ch)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.transformer(x)
        return self.norm(x[:, 0])

    def forward(self, x):
        return self.head(self.extract(x))
