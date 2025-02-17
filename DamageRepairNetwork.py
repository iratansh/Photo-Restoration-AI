from torch import nn
from ConvBlock import ConvBlock
import torch
from torchvision import transforms

class DamageRepairNetwork(nn.Module):
    def __init__(self):
        super(DamageRepairNetwork, self).__init__()
        
        self.damage_detector = nn.Sequential(
            ConvBlock(3, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        self.reconstructor = nn.Sequential(
            ConvBlock(4, 64), 
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 128),
            ConvBlock(128, 64),
            nn.Conv2d(64, 3, 3, padding=1)
        )
        
    def forward(self, x):
        damage_mask = self.damage_detector(x)
        features = torch.cat([x, damage_mask], dim=1)
        reconstructed = self.reconstructor(features)
        
        return x * (1 - damage_mask) + reconstructed * damage_mask
    
    def repair_damage(self, image):
        img_tensor = transforms.ToTensor()(image).unsqueeze(0)
        repaired_tensor = self.damage_repair(img_tensor)
        return transforms.ToPILImage()(repaired_tensor.squeeze(0))