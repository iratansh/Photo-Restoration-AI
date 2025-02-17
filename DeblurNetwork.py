from ConvBlock import ConvBlock
import torch.nn as nn
import torch

class DeblurNetwork(nn.Module):
    def __init__(self):
        super(DeblurNetwork, self).__init__()
        
        self.enc1 = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        
        self.mp = nn.MaxPool2d(2)
        self.conv_mid = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 256)
        )
        
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = ConvBlock(256, 128) 
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)  
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1)
        )
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.mp(e1))
        e3 = self.enc3(self.mp(e2))
        
        middle = self.conv_mid(self.mp(e3))
        
        d1 = self.up1(middle)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        out = self.final(d2)
        return out + x 