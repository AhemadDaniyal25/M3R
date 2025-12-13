# src/models/unet3d.py
"""
Stable symmetric 3D autoencoder kept under the name UNet3D so training script works unchanged.
Deterministic channel sizing and up/down sampling — no skip connections to avoid shape bugs.
"""
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet3D(nn.Module):
    """
    Simple symmetric 3D autoencoder.
    in_ch: 1
    base_chs: deterministic [8,16,32] for encoder; bottleneck 64; symmetric decoder.
    """
    def __init__(self, in_ch=1):
        super().__init__()
        # encoder
        self.enc1 = ConvBlock(in_ch, 8)
        self.down1 = nn.Conv3d(8, 8, kernel_size=2, stride=2)   # halves spatial, keeps channels
        self.enc2 = ConvBlock(8, 16)
        self.down2 = nn.Conv3d(16, 16, kernel_size=2, stride=2)
        self.enc3 = ConvBlock(16, 32)
        self.down3 = nn.Conv3d(32, 32, kernel_size=2, stride=2)
        # bottleneck
        self.bottleneck = ConvBlock(32, 64)
        # decoder
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 32)
        self.up2 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(16, 16)
        self.up3 = nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(8, 8)
        # final
        self.outc = nn.Conv3d(8, 1, kernel_size=1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.down1(x)
        x = self.enc2(x)
        x = self.down2(x)
        x = self.enc3(x)
        x = self.down3(x)
        x = self.bottleneck(x)
        x = self.up1(x); x = self.dec1(x)
        x = self.up2(x); x = self.dec2(x)
        x = self.up3(x); x = self.dec3(x)
        x = self.outc(x)
        return x
