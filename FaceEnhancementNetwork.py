from torch import nn
from ConvBlock import ConvBlock
import torch

class FaceEnhancementNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )
        
        # Context module
        self.context = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 256)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            ConvBlock(256, 128),
            ConvBlock(128, 64),
            ConvBlock(64, 3)
        )
        
        # Detail enhancement branch
        self.detail_branch = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 3)
        )
        
        # Blending module
        self.blend_conv = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        context = self.context(features)
        enhanced_base = self.decoder(features * context)
        enhanced_details = self.detail_branch(x)
        
        blend_weights = self.blend_conv(torch.cat([enhanced_base, enhanced_details], dim=1))
        return blend_weights * enhanced_base + (1 - blend_weights) * enhanced_details