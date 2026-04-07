"""4-layer residual CNN for classifying 11x11 feature images.

Two residual blocks (1->32->64 channels). Global average pooling produces
a 64-dim representation. Split into extract/forward for transfer experiments.

Representation: 64-dim (aligned across all models)
"""

import torch.nn as nn


class DeeperCNN(nn.Module):

    # Representation size, needed by downstream evaluation code
    REP_DIM = 64

    def __init__(self, dropout=0.3):
        super().__init__()

        # First residual block: grayscale input (1 channel) expanded to 32 feature maps.
        # Two 3x3 convolutions with same-padding keep spatial size at 11x11.
        # The skip connection adds the input directly since channels match.
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU())

        # Second residual block: 32 -> 64 channels.
        # A 1x1 convolution on the skip path handles the channel mismatch.
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.skip34 = nn.Conv2d(32, 64, 1)

        # Collapse spatial dims to a 64-dim representation vector
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classification head: 64-dim -> 1 logit.
        # Separate from backbone so we can swap it for downstream classifiers.
        self.head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def extract(self, x):
        """Backbone pass: returns the 64-dim representation without classification."""
        x = self.conv1(x)
        x = self.conv2(x) + x                   # block 1 residual
        h = self.conv3(x)
        h = self.conv4(h) + self.skip34(x)       # block 2 residual (channel-projected)
        return self.pool(h).flatten(1)

    def forward(self, x):
        return self.head(self.extract(x))
