import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.models.unet3d import UNet3D
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import imageio

# --- CONFIGURATION ---
DATA_PATH = os.path.join("data", "sample_fastmri", "sample.nii.gz")
MODEL_DIR = os.path.join("models")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- HELPER CLASSES ---
class EdgeLoss(nn.Module):
    """
    Computes the Mean Absolute Error between the gradients of the prediction and target.
    This discourages 'blurry' outputs which are common with pure MSE Loss.
    """
    def __init__(self, device="cpu"):
        super().__init__()
        # Sobel Kernel for 2D gradients (3x3)
        # Defined as a simple 2D matrix first to avoid dimension confusion
        k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        
        # Reshape to (Out_channels=1, In_channels=1, H=3, W=3) for Conv2d
        self.k_x = k.unsqueeze(0).unsqueeze(0).to(device)
        
        # Transpose for Y-direction (swap rows/cols) and reshape
        self.k_y = k.transpose(0, 1).unsqueeze(0).unsqueeze(0).to(device)
        
        self.device = device
        
    def forward(self, pred, target):
        # Input shape: (Batch, Channel, Depth, Height, Width)
        # We reshape to (Batch*Depth, 1, Height, Width) to compute 2D gradients on every slice
        b, c, d, h, w = pred.shape
        pred_2d = pred.view(b*d, 1, h, w)
        target_2d = target.view(b*d, 1, h, w)
        
        # Compute Gradients
        pred_dx = F.conv2d(pred_2d, self.k_x, padding=1)
        pred_dy = F.conv2d(pred_2d, self.k_y, padding=1)
        target_dx = F.conv2d(target_2d, self.k_x, padding=1)
        target_dy = F.conv2d(target_2d, self.k_y, padding=1)
        
        # Loss is the difference in edge strength
        return torch.mean(torch.abs(pred_dx - target_dx) + torch.abs(pred_dy - target_dy))

# --- DATA FUNCTIONS ---

def load_vol(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}. Did you run scripts/get_real_data.py?")
    
    img = nib.load(path)
    vol = img.get_fdata().astype(np.float32)
    
    # 1. Normalize (0 to 1)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    
    # 2. Pad dimensions to be divisible by 8 (Standard UNet requirement)
    # We check Depth, Height, Width
    pad_dims = []
    for dim in vol.shape:
        if dim % 8 == 0:
            pad_dims.append((0, 0)) # No padding needed
        else:
            # Calculate how much to add to reach the next multiple of 8
            needed = 8 - (dim % 8)
            pad_dims.append((0, needed))
            
    # Apply padding: ((0, pad_d), (0, pad_h), (0, pad_w))
    if any(p[1] > 0 for p in pad_dims):
        print(f"⚠️ Padding volume shape {vol.shape} to be divisible by 8...")
        vol = np.pad(vol, pad_dims, mode='constant', constant_values=0)
        print(f"   New shape: {vol.shape}")

    return vol

def make_training_pair(vol, step):
    """
    Creates a sparse input by keeping only every nth slice.
    """
    sparse = np.zeros_like(vol)
    sparse[::step] = vol[::step]
    
    # Add Batch and Channel dimensions: (1, 1, D, H, W)
    x = sparse[None, None]
    y = vol[None, None]
    return x.astype(np.float32), y.astype(np.float32)

def compute_metrics(gt, pred):
    D = gt.shape[0]
    s_scores, p_scores = [], []
    for i in range(D):
        g = gt[i]
        r = pred[i]
        # Normalize per slice for metric stability
        g_n = (g - g.min()) / (g.max() - g.min() + 1e-8)
        r_n = (r - r.min()) / (r.max() - r.min() + 1e-8)
        try:
            s_scores.append(ssim(g_n, r_n, data_range=1.0))
        except:
            s_scores.append(0.0)
        p_scores.append(psnr(g_n, r_n, data_range=1.0))
    return float(np.mean(s_scores)), float(np.mean(p_scores))

# --- MAIN TRAINING LOOP ---

def main(epochs=300, step=8, lr=1e-3, device="cpu"):
    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    vol = load_vol(DATA_PATH)
    x_np, y_np = make_training_pair(vol, step=step)
    
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    
    # 2. Setup Model & Loss
    print(f"Initializing UNet3D on {device}...")
    model = UNet3D(in_ch=1).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    
    criterion_mse = nn.MSELoss()
    criterion_edge = EdgeLoss(device=device)
    
    # 3. Training
    print(f"Starting training for {epochs} epochs (Step={step})...")
    model.train()
    
    for ep in range(epochs):
        opt.zero_grad()
        out = model(x)
        
        # Calculate Losses
        loss_mse = criterion_mse(out, y)
        loss_edge = criterion_edge(out, y)
        
        # Composite Loss: We weight Edge Loss less (0.1) because gradients can be noisy
        loss = loss_mse + (0.1 * loss_edge)
        
        loss.backward()
        opt.step()
        
        if (ep + 1) % 20 == 0:
            print(f"Epoch {ep+1}/{epochs} | Total: {loss.item():.6f} | MSE: {loss_mse.item():.6f} | Edge: {loss_edge.item():.6f}")

    # 4. Save Model
    model_path = os.path.join(MODEL_DIR, "unet_demo.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    # 5. Quick Eval & Sanity Check
    model.eval()
    with torch.no_grad():
        pred = model(x).cpu().numpy()[0, 0]
        
    # Save a check image (Middle slice)
    mid = pred.shape[0] // 2
    img_slice = (pred[mid] - pred[mid].min()) / (pred[mid].max() - pred[mid].min() + 1e-8)
    out_png = os.path.join("data", "sample_fastmri", "recon_slice_check.png")
    imageio.imwrite(out_png, (img_slice * 255).astype("uint8"))
    
    # Compute Metrics
    s, p = compute_metrics(vol, pred)
    print(f"Final Metrics -> SSIM: {s:.4f}, PSNR: {p:.4f}")
    print(f"Wrote sanity check image to {out_png}")

if __name__ == "__main__":
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # RUN
    # Note: We use step=8 to ensure the task is hard enough to distinguish Neural vs Linear
    main(epochs=300, step=8, lr=1e-3, device=device)