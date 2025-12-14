import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.models.unet3d import UNet3D

# 1. Load the data
DATA_PATH = "data/sample_fastmri/sample.nii.gz"
MODEL_PATH = "models/unet_demo.pt"
vol = nib.load(DATA_PATH).get_fdata()

# 2. Create the "Black" Sparse Input
# (We delete 7 slices, keep 1, delete 7...)
STEP = 8
sparse = np.zeros_like(vol)
sparse[::STEP] = vol[::STEP]  # This creates the "gaps"

# 3. Load your trained model
model = UNet3D(in_ch=1)
if torch.cuda.is_available():
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# 4. Run Reconstruction
print("Reconstructing...")
inp = torch.from_numpy(sparse[None, None].astype("float32"))
with torch.no_grad():
    recon = model(inp).numpy()[0, 0]

# 5. VISUALIZE THE SIDE (Sagittal View)
# Instead of looking at a "Gap" slice (z), we cut down the middle (x)
mid_x = vol.shape[2] // 2

plt.figure(figsize=(12, 5))

# Plot 1: The Sparse Input (Side View)
plt.subplot(1, 3, 1)
# We rotate it so it looks like a head standing up
plt.imshow(np.rot90(sparse[:, :, mid_x]), cmap='gray')
plt.title("Sparse Input (Side View)\nLOOK! LINES!") 
plt.axis('off')

# Plot 2: The Neural Recon
plt.subplot(1, 3, 2)
plt.imshow(np.rot90(recon[:, :, mid_x]), cmap='gray')
plt.title("Neural Reconstruction\n(Fills the gaps)")
plt.axis('off')

# Plot 3: Ground Truth
plt.subplot(1, 3, 3)
plt.imshow(np.rot90(vol[:, :, mid_x]), cmap='gray')
plt.title("Original Ground Truth")
plt.axis('off')

plt.tight_layout()
plt.show()