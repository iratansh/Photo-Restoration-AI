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
        
        # Decoder with additional upsampling
        self.up1 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec1 = ConvBlock(512, 128)  # 256+256=512 channels after cat
        
        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 64)   # 128+128=256 channels after cat
        
        self.up3 = nn.ConvTranspose2d(64, 64, 2, stride=2)  
        self.dec3 = ConvBlock(128, 32)    # 64+64=128 channels after cat
        
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1)
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                 # [B, 64, H, W]
        e2 = self.enc2(self.mp(e1))       # [B, 128, H/2, W/2]
        e3 = self.enc3(self.mp(e2))       # [B, 256, H/4, W/4]
        
        # Middle
        middle = self.conv_mid(self.mp(e3))  # [B, 256, H/8, W/8]
        
        # Decoder
        d1 = self.up1(middle)             # [B, 256, H/4, W/4]
        d1 = torch.cat([d1, e3], dim=1)   # [B, 512, H/4, W/4]
        d1 = self.dec1(d1)                # [B, 128, H/4, W/4]
        
        d2 = self.up2(d1)                 # [B, 128, H/2, W/2]
        d2 = torch.cat([d2, e2], dim=1)   # [B, 256, H/2, W/2]
        d2 = self.dec2(d2)                # [B, 64, H/2, W/2]
        
        d3 = self.up3(d2)                 # [B, 64, H, W]    
        d3 = torch.cat([d3, e1], dim=1)    # [B, 128, H, W]   
        d3 = self.dec3(d3)                # [B, 32, H, W]
        
        out = self.final(d3)              # [B, 3, H, W]
        return out + x 