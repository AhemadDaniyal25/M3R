import torch
import numpy as np
from src.models.unet3d import UNet3D
import nibabel as nib

vol = nib.load("data/sample_fastmri/sample.nii.gz").get_fdata()
sparse = np.zeros_like(vol)
sparse[::4] = vol[::4]

x = torch.from_numpy(sparse[None,None].astype("float32"))
model = UNet3D(in_ch=1)
model.load_state_dict(torch.load("models/unet_demo.pt", map_location="cpu"))
model.eval()

with torch.no_grad():
    out = model(x)

print("Output shape:", out.shape)
print("Output min/max:", out.min().item(), out.max().item())
