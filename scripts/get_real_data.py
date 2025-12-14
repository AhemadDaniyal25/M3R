# scripts/get_real_data.py
import os
import numpy as np
import nibabel as nib
from nilearn.datasets import fetch_haxby

# 1. Setup paths
OUT_DIR = "data/sample_fastmri"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "sample.nii.gz")

print("Fetching Single-Subject Haxby Anatomical Scan...")
# This fetches raw data from ONE person. It will be sharp/noisy.
dataset = fetch_haxby()
anat_path = dataset.anat[0]
img = nib.load(anat_path)
vol = img.get_fdata().astype(np.float32)

print(f"Original shape: {vol.shape}")

# 2. Robust Normalization (Clip high percentile to remove outlier noise)
# Real MRI has bright noise spikes. We clip top 1% to make contrast better.
v_min, v_max = np.percentile(vol, [0, 99.5])
vol = np.clip(vol, v_min, v_max)
vol = (vol - v_min) / (v_max - v_min)

# 3. Crop to relevant brain area (remove neck/air)
# Haxby anat is usually around (124, 256, 256). We crop the center.
cx, cy, cz = vol.shape[0]//2, vol.shape[1]//2, vol.shape[2]//2
# Crop size: 64 depth, 160 height, 160 width (Standardize for UNet)
D, H, W = 64, 160, 160

# Ensure we don't go out of bounds
z_start = max(0, cz - D//2)
z_end = z_start + D
y_start = max(0, cy - H//2)
y_end = y_start + H
x_start = max(0, cx - W//2)
x_end = x_start + W

crop_vol = vol[z_start:z_end, y_start:y_end, x_start:x_end]
print(f"Cropped shape: {crop_vol.shape}")

# 4. Save
nib.save(nib.Nifti1Image(crop_vol, affine=np.eye(4)), OUT_PATH)
print(f"✅ Saved SHARP single-subject MRI to {OUT_PATH}")