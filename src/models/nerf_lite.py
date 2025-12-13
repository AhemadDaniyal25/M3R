# src/models/nerf_lite.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, in_dim, num_freqs=6):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        out = [x]
        for f in self.freqs:
            out.append(torch.sin(x * f))
            out.append(torch.cos(x * f))
        return torch.cat(out, dim=-1)

class NerfLite(nn.Module):
    def __init__(self, in_dim=3, hidden=128, n_layers=4, posenc=True, pe_freqs=6):
        super().__init__()
        self.posenc_on = posenc
        if posenc:
            self.pe = PositionalEncoding(in_dim, num_freqs=pe_freqs)
            in_dim = in_dim * (1 + 2 * pe_freqs)
        layers = []
        last = in_dim
        for i in range(n_layers):
            layers.append(nn.Linear(last, hidden))
            layers.append(nn.ReLU(inplace=True))
            last = hidden
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        # coords: (N, 3) float tensor in normalized coordinates [-1,1]
        if self.posenc_on:
            coords = self.pe(coords)
        out = self.net(coords)
        return out.squeeze(-1)
