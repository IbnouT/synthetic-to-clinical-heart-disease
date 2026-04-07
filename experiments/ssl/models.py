"""Neural network architectures for self-supervised tabular learning.

Three methods built on the same encoder backbone:

SCARF (Self-supervised Contrastive Learning for Tabular Data):
    Corrupts a random subset of features and trains the encoder to
    distinguish original from corrupted views via contrastive loss.
    The encoder learns which feature patterns are "real" vs noise.

MAE (Masked Autoencoder for Tabular Data):
    Masks random features with learnable tokens and trains a decoder
    to reconstruct them. The encoder learns a compressed representation
    that captures feature dependencies.

SemiMAE (Semi-Supervised MAE):
    Same architecture as MAE but treats the label as a 14th input
    dimension. Two masking strategies:
      - "random": label masked at the same rate as features (30%)
      - "always": label always masked, features at 30%
    At inference the label column uses the learned mask token, so the
    encoder never sees the true label but has learned to predict it.
"""

import torch
import torch.nn as nn


class _ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, latent_dim=64, proj_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    """Shared encoder backbone: input_dim -> hidden -> hidden -> latent.

    Two hidden layers with batch normalization, GELU activations, and
    dropout. Used by all three SSL methods (SCARF, MAE, SemiMAE).
    """

    def __init__(self, input_dim, hidden_dim=256, latent_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class SCARF(nn.Module):
    """SCARF contrastive model with encoder + projection head.

    The projection head maps latent representations to a lower-dimensional
    space where the contrastive loss (NT-Xent) is computed. After
    pretraining, only the encoder is used for downstream tasks.
    """

    def __init__(self, input_dim=13, hidden_dim=256, latent_dim=64,
                 proj_dim=32, corruption_rate=0.3):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        # Projection head maps latent space to contrastive space.
        # Wrapped in a module with .net attribute for checkpoint compatibility
        # with the original training code.
        self.projector = _ProjectionHead(latent_dim, proj_dim)
        self.corruption_rate = corruption_rate

    def forward(self, x):
        """Forward pass for contrastive pretraining.

        Creates a corrupted view by replacing random features with values
        sampled from other rows in the batch. Both views are encoded and
        projected for the contrastive loss.
        """
        batch_size, n_features = x.shape

        # Build corrupted view: for each masked position, swap with a
        # random value from another row in the batch
        mask = torch.rand(batch_size, n_features, device=x.device) < self.corruption_rate
        random_indices = torch.randint(0, batch_size, (batch_size, n_features), device=x.device)
        random_values = x[random_indices, torch.arange(n_features, device=x.device)]
        x_corrupted = torch.where(mask, random_values, x)

        # Encode both views
        z_original = self.encoder(x)
        z_corrupted = self.encoder(x_corrupted)

        # Project to contrastive space
        p_original = self.projector(z_original)
        p_corrupted = self.projector(z_corrupted)

        return p_original, p_corrupted

    @torch.no_grad()
    def encode(self, x):
        """Extract encoder representations (no projection head)."""
        self.eval()
        return self.encoder(x)


class MAE(nn.Module):
    """Masked Autoencoder for tabular data.

    Replaces masked features with learnable tokens, encodes the partially
    visible input, then decodes to reconstruct all original values.
    Loss is only computed on masked positions.

    When include_label=True, the model operates on 14 dimensions (13
    features + 1 label). The label_mask_mode controls how the label
    column is masked during training:
      - None:     standard MAE, no label column
      - "random": label masked at the same rate as other features
      - "always": label is always masked (100%), other features at mask_rate
    """

    def __init__(self, input_dim=13, hidden_dim=256, latent_dim=64,
                 mask_rate=0.3, dropout=0.1, label_mask_mode=None):
        super().__init__()
        self.mask_rate = mask_rate
        self.input_dim = input_dim
        self.label_mask_mode = label_mask_mode

        # One learnable mask token per feature dimension
        self.mask_token = nn.Parameter(torch.zeros(input_dim))

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        """Forward pass: mask -> encode -> decode -> reconstruction loss.

        Returns (reconstruction, mask, latent) where mask indicates which
        positions were masked (True = masked, used for loss computation).
        """
        batch_size, n_dims = x.shape

        # Random mask: True means "this position is masked"
        mask = torch.rand(batch_size, n_dims, device=x.device) < self.mask_rate

        # Force label column masking if in semi-supervised mode
        if self.label_mask_mode == "always":
            mask[:, -1] = True

        # Replace masked positions with learnable tokens
        x_masked = torch.where(mask, self.mask_token.expand(batch_size, -1), x)

        z = self.encoder(x_masked)
        x_recon = self.decoder(z)

        return x_recon, mask, z

    @torch.no_grad()
    def encode(self, x):
        """Extract encoder representations from full input (all dims visible)."""
        self.eval()
        return self.encoder(x)

    @torch.no_grad()
    def encode_without_label(self, x_features):
        """Extract representations from 13-dim input by padding with mask token.

        Used at inference for SemiMAE models: the label column is replaced
        with the learned mask token so the encoder processes 14 dimensions
        but never sees the true label.
        """
        self.eval()
        batch_size = x_features.shape[0]
        label_pad = self.mask_token[-1:].expand(batch_size, 1)
        x_full = torch.cat([x_features, label_pad], dim=1)
        return self.encoder(x_full)
