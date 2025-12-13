# src/app/api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os
import threading

# inference imports
import torch
import nibabel as nib
import numpy as np
import imageio

# model import (uses your repo src/ package)
from src.models.unet3d import UNet3D

app = FastAPI(title="M3R Demo API")

BASE = os.path.join("data", "sample_fastmri")
SPLAT_DIR = os.path.join(BASE, "splat_demo")
RECON_PNG = os.path.join(BASE, "recon_slice.png")
MODEL_PATH = os.path.join("models", "unet_demo.pt")

_model_lock = threading.Lock()
_model = None

def load_model_once(device="cpu"):
    global _model
    with _model_lock:
        if _model is None:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError("Model not found. Run scripts/train_unet.py first.")
            m = UNet3D(in_ch=1)  # matches training
            state = torch.load(MODEL_PATH, map_location=device)
            m.load_state_dict(state)
            m.to(device)
            m.eval()
            _model = m
    return _model

def make_sparse_input(step=4):
    """Return a numpy volume where every `step` slice is kept and others zeroed (D,H,W)"""
    path = os.path.join(BASE, "sample.nii.gz")
    if not os.path.exists(path):
        raise FileNotFoundError("Synthetic sample not found. Run scripts/create_synthetic_data.py")
    img = nib.load(path)
    vol = img.get_fdata().astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    sparse = np.zeros_like(vol)
    sparse[::step] = vol[::step]
    # model expects (1,1,D,H,W) torch tensor
    return sparse

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/render/gauss")
def render_gauss(angle: int = 0):
    fn = f"splat_view_{angle}.png"
    path = os.path.join(SPLAT_DIR, fn)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {fn}. Run scripts/run_splat_demo.py first.")
    return FileResponse(path, media_type="image/png")

@app.get("/render/nerf_density")
def render_nerf_density():
    path = os.path.join(SPLAT_DIR, "nerf_density.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="nerf_density.png not found. Run scripts/run_splat_demo.py first.")
    return FileResponse(path, media_type="image/png")

@app.get("/reconstruct")
def reconstruct(step: int = 4):
    """
    Runs UNet inference on the sparse input (keeps every `step`-th slice).
    Returns the reconstructed mid-slice PNG (and saves it to data/.../recon_slice.png).
    Usage: /reconstruct?step=4
    """
    try:
        model = load_model_once(device="cpu")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    sparse = make_sparse_input(step=step)  # (D,H,W)
    x = torch.from_numpy(sparse[None, None].astype(np.float32))  # (1,1,D,H,W)
    with torch.no_grad():
        out = model(x).cpu().numpy()[0,0]  # (D,H,W)
    # save mid-slice png
    mid = out.shape[0] // 2
    img = out[mid]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    os.makedirs(BASE, exist_ok=True)
    imageio.imwrite(RECON_PNG, (img*255).astype("uint8"))
    return FileResponse(RECON_PNG, media_type="image/png")
