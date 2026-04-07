"""Expansion layer and model wrapper.

The expansion layer projects 13 raw clinical features into 121 learned
features via a linear layer, then reshapes them as an 11x11 image.
This replaces the hand-crafted 117 mega-feature pipeline used in other
experiments — here the feature interactions are learned end-to-end.

ExpansionModel wraps expansion + any image backbone so the full pipeline
(13 features -> image -> prediction) trains as one model.
"""

import torch.nn as nn

from ..config import N_RAW_FEATURES, EXPANSION_DIM, IMG_SIZE


class ExpansionLayer(nn.Module):
    """Linear(13 -> 121) + BatchNorm + GELU, reshaped to 1x11x11.

    Each of the 121 outputs is a learned weighted combination of all 13
    input features. BatchNorm stabilizes the scale before the image
    backbone processes it.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_RAW_FEATURES, EXPANSION_DIM),
            nn.BatchNorm1d(EXPANSION_DIM),
            nn.GELU(),
        )

    def forward(self, x):
        """(batch, 13) -> (batch, 1, 11, 11)"""
        return self.net(x).view(-1, 1, IMG_SIZE, IMG_SIZE)


class ExpansionModel(nn.Module):
    """Combines expansion + backbone into one end-to-end model.

    Input is raw 13 features. The expansion layer converts them to an
    11x11 image, then the backbone (CNN, ViT, etc.) processes it.
    """

    def __init__(self, backbone):
        super().__init__()
        self.expansion = ExpansionLayer()
        self.backbone = backbone

    def forward(self, x):
        """13 features -> expansion -> backbone -> classification logit."""
        return self.backbone(self.expansion(x))

    def extract(self, x):
        """13 features -> expansion -> backbone representation (no head)."""
        return self.backbone.extract(self.expansion(x))
