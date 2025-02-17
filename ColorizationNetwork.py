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
        
        # Colorization decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        if x.shape[1] == 3:
            x = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
            x = x.unsqueeze(1)

        x = x.repeat(1, 3, 1, 1)
        features = self.feature_extractor(x)
        fused = self.fusion(features)
        ab = self.decoder(fused)
        
        out = torch.cat([x[:, :1], ab], dim=1) 
        return out