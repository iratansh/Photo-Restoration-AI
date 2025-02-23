from torch import nn
from torchvision import models
import torch

class ColorizationNetwork(nn.Module):
    def __init__(self):
        super(ColorizationNetwork, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),  # 8x8 → 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 16x16 → 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 32x32 → 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),    # 64x64 → 128x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),     # 128x128 → 256x256
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 2, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        if x.shape[1] == 3:
            x = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
            x = x.unsqueeze(1)

        x = x.repeat(1, 3, 1, 1)
        features = self.feature_extractor(x)  # [B, 512, 8, 8]
        fused = self.fusion(features)         # [B, 128, 8, 8]
        ab = self.decoder(fused)              # [B, 2, 256, 256]
        out = torch.cat([x[:, :1], ab], dim=1)  # Now both are 256x256
        return out