from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class PhotoRestorationLoss(nn.Module):
    def __init__(self):
        super(PhotoRestorationLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:23]
        self.vgg = nn.Sequential(*list(vgg.children())).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        for param in self.vgg.parameters():
            param.requires_grad = False
            
    def forward(self, pred, target):
        l1_loss = F.l1_loss(pred, target)
        
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        perceptual_loss = F.mse_loss(pred_features, target_features)
        
        tv_loss = torch.mean(torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])) + \
                 torch.mean(torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :]))
        
        total_loss = l1_loss + 0.1 * perceptual_loss + 0.0001 * tv_loss
        return total_loss