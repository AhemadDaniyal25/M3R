# src/models/unet3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            # Second conv for better feature extraction
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UpSampleBlock(nn.Module):
    """
    Replaces ConvTranspose3d to eliminate Checkerboard Artifacts.
    Uses Nearest Neighbor Upsampling + Convolution.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1. Upsample (Nearest neighbor is checkerboard-free)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # 2. Convolve
        return self.relu(self.bn(self.conv(x)))

class UNet3D(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch, 8)
        self.down1 = nn.Conv3d(8, 8, kernel_size=2, stride=2)
        
        self.enc2 = ConvBlock(8, 16)
        self.down2 = nn.Conv3d(16, 16, kernel_size=2, stride=2)
        
        self.enc3 = ConvBlock(16, 32)
        self.down3 = nn.Conv3d(32, 32, kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(32, 64)
        
        # Decoder (Using UpSampleBlock instead of ConvTranspose)
        self.up1 = UpSampleBlock(64, 32)
        self.dec1 = ConvBlock(32, 32)
        
        self.up2 = UpSampleBlock(32, 16)
        self.dec2 = ConvBlock(16, 16)
        
        self.up3 = UpSampleBlock(16, 8)
        self.dec3 = ConvBlock(8, 8)
        
        # Final
        self.outc = nn.Conv3d(8, 1, kernel_size=1)

    def forward(self, x):
        # Standard Encoder-Decoder path
        x = self.enc1(x); x = self.down1(x)
        x = self.enc2(x); x = self.down2(x)
        x = self.enc3(x); x = self.down3(x)
        
        x = self.bottleneck(x)
        
        x = self.up1(x); x = self.dec1(x)
        x = self.up2(x); x = self.dec2(x)
        x = self.up3(x); x = self.dec3(x)
        
        x = self.outc(x)
        return x