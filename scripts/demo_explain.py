import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import interpolate
from src.models.unet3d import UNet3D

def linear_interpolate_volume(sparse_vol, step):
    """
    Fills zeros in a sparse volume using 1D linear interpolation along the z-axis.
    """
    D, H, W = sparse_vol.shape
    dense_vol = np.zeros_like(sparse_vol)
    
    # Known indices (where we have data)
    known_z = np.arange(0, D, step)
    
    # We interpolate for all z (0 to D-1)
    target_z = np.arange(D)

    for r in range(H):
        for c in range(W):
            # Extract the sparse column (only the known values)
            values = sparse_vol[known_z, r, c]
            
            # Create interpolator
            f = interpolate.interp1d(known_z, values, kind='linear', fill_value="extrapolate")
            
            # Fill column
            dense_vol[:, r, c] = f(target_z)
            
    return dense_vol

# 1. Configuration (MUST match training)
STEP = 8  # Matches your train_unet.py default
MODEL_PATH = "models/unet_demo.pt"
DATA_PATH = "data/sample_fastmri/sample.nii.gz"

# 2. Load Data
print(f"Loading {DATA_PATH}...")
vol = nib.load(DATA_PATH).get_fdata()

# 3. Create Sparse Input
sparse = np.zeros_like(vol)
sparse[::STEP] = vol[::STEP]

# 4. Generate Baselines
print("Running Linear Interpolation Baseline (this might take a moment)...")
linear_recon = linear_interpolate_volume(sparse, STEP)

# 5. Run Neural Reconstruction
print("Running Neural Reconstruction...")
model = UNet3D(in_ch=1)
# Load model (handle map_location for CPU/GPU safety)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

inp_tensor = torch.from_numpy(sparse[None, None].astype("float32"))
with torch.no_grad():
    neural_recon = model(inp_tensor).numpy()[0, 0]

# 6. Select a slice that was MISSING
# If we pick a slice divisible by STEP, it was seen by the model. We want one in between.
z_middle = vol.shape[0] // 2
# Ensure z is NOT divisible by STEP (force it to be a missing slice)
if z_middle % STEP == 0:
    z_target = z_middle + (STEP // 2)
else:
    z_target = z_middle

print(f"Visualizing Slice Z={z_target} (This slice was missing in input)")

# 7. Visualization
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
titles = ["Original (Ground Truth)", "Sparse Input (What Model Saw)", "Linear Baseline", "Neural Recon"]
images = [vol[z_target], sparse[z_target], linear_recon[z_target], neural_recon[z_target]]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
out_file = "data/sample_fastmri/demo_comparison.png"
plt.savefig(out_file, dpi=150)
print(f"Saved comparison to {out_file}")
plt.show()