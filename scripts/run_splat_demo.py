# scripts/run_splat_demo.py
import os
import numpy as np
import nibabel as nib
from src.splatting.splatify import splatify_volume
from src.splatting.renderer import render_splats_image, save_image
from src.models.nerf_lite import NerfLite
import torch
import imageio

DATA_SAMPLE = os.path.join("data","sample_fastmri","sample.nii.gz")
OUT_DIR = os.path.join("data","sample_fastmri","splat_demo")
os.makedirs(OUT_DIR, exist_ok=True)

def load_vol(path):
    img = nib.load(path)
    vol = img.get_fdata().astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    return vol

def run_splat_demo():
    vol = load_vol(DATA_SAMPLE)
    print("Volume shape:", vol.shape)
    splats = splatify_volume(vol, mode="topk", topk=2000)
    print("Created splats:", splats.shape[0])
    for angle in [0, 45, 90, 135]:
        img = render_splats_image(splats, vol.shape, angle_deg=angle, out_size=(512,512))
        outp = os.path.join(OUT_DIR, f"splat_view_{angle}.png")
        save_image(img, outp)
        print("Wrote", outp)
    # small nerf-lite demo
    D,H,W = vol.shape
    rng = np.random.RandomState(0)
    pts = rng.rand(2000,3)
    pts[:,0] = (pts[:,0] * (W-1))
    pts[:,1] = (pts[:,1] * (H-1))
    pts[:,2] = (pts[:,2] * (D-1))
    coords_norm = (pts / np.array([W-1,H-1,D-1]) - 0.5) * 2.0
    coords_t = torch.from_numpy(coords_norm).float()
    model = NerfLite(in_dim=3, hidden=64, n_layers=3, posenc=True, pe_freqs=4)
    model.eval()
    with torch.no_grad():
        out = model(coords_t).cpu().numpy()
    dens_img = np.zeros((64,64), dtype=np.float32)
    for i in range(min(4096, pts.shape[0])):
        x = int((pts[i,0]/(W-1)) * 63)
        y = int((pts[i,1]/(H-1)) * 63)
        dens_img[y,x] = max(dens_img[y,x], out[i])
    dens_img = (dens_img - dens_img.min()) / (dens_img.max() - dens_img.min() + 1e-8)
    dens_path = os.path.join(OUT_DIR, "nerf_density.png")
    imageio.imwrite(dens_path, (dens_img*255).astype('uint8'))
    print("Wrote", dens_path)
    print("Splat demo complete. Check", OUT_DIR)

if __name__ == "__main__":
    run_splat_demo()
