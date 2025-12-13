# scripts/train_unet.py
import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.unet3d import UNet3D
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import imageio

DATA = os.path.join("data","sample_fastmri","sample.nii.gz")
MODEL_DIR = os.path.join("models")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_vol(path):
    img = nib.load(path)
    vol = img.get_fdata().astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    return vol  # shape (D,H,W)

def make_training_pair(vol, step=4):
    sparse = np.zeros_like(vol)
    sparse[::step] = vol[::step]
    x = sparse[None,None]  # (1,1,D,H,W)
    y = vol[None,None]
    return x.astype(np.float32), y.astype(np.float32)

def compute_metrics(gt, pred):
    D = gt.shape[0]
    s = []
    p = []
    for i in range(D):
        g = gt[i]; r = pred[i]
        g_n = (g - g.min())/(g.max()-g.min()+1e-8)
        r_n = (r - r.min())/(r.max()-r.min()+1e-8)
        try:
            s.append(ssim(g_n, r_n, data_range=1.0))
        except:
            s.append(0.0)
        p.append(psnr(g_n, r_n, data_range=1.0))
    return float(np.mean(s)), float(np.mean(p))

def main(epochs=6, step=4, lr=1e-3, device="cpu"):
    vol = load_vol(DATA)
    x_np, y_np = make_training_pair(vol, step=step)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    model = UNet3D(in_ch=1).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    print("Starting training on device:", device)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        print(f"Epoch {ep+1}/{epochs} loss: {loss.item():.6f}")
    model_path = os.path.join(MODEL_DIR, "unet_demo.pt")
    torch.save(model.state_dict(), model_path)
    print("Saved model to", model_path)
    model.eval()
    with torch.no_grad():
        pred = model(x).cpu().numpy()[0,0]
    mid = pred.shape[0]//2
    img = (pred[mid] - pred[mid].min())/(pred[mid].max()-pred[mid].min()+1e-8)
    out_png = os.path.join("data","sample_fastmri","recon_slice.png")
    imageio.imwrite(out_png, (img*255).astype("uint8"))
    print("Wrote reconstructed slice to", out_png)
    s,p = compute_metrics(vol, pred)
    print(f"Reconstruction metrics: SSIM={s:.4f}, PSNR={p:.4f}")

if __name__ == "__main__":
    main(epochs=6, step=4, lr=1e-3, device="cpu")
