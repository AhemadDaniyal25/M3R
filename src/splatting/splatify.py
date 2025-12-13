# src/splatting/splatify.py
import numpy as np
from scipy.ndimage import gaussian_filter

def splatify_volume(vol, mode="topk", topk=2000, downsample=(2,2,2), sigma_scale=0.8, kmeans_k=512):
    """
    vol: 3D numpy array (D,H,W) normalized 0..1
    mode: "topk" or "grid+kmeans"
    Returns: splats array (N,6): x,y,z,sigma,intensity,weight (voxel coords)
    """
    D,H,W = vol.shape
    # smooth slightly
    vol_s = gaussian_filter(vol, sigma=1.0)
    if mode == "topk":
        flat = vol_s.ravel()
        k = min(int(topk), flat.size)
        idx = np.argpartition(-flat, k-1)[:k]
        zs = idx // (H*W)
        rem = idx % (H*W)
        ys = rem // W
        xs = rem % W
        intensities = flat[idx]
        # sigma scaled to volume size and intensity
        sigmas = np.clip(np.sqrt(intensities) * sigma_scale * max(D,H,W)/100.0, 0.5, max(D,H,W)/5.0)
        splats = np.stack([xs, ys, zs, sigmas, intensities, np.ones_like(intensities)], axis=1)
        return splats
    else:
        # fallback simple sampling if grid+kmeans path not used (keeps code minimal here)
        pts = np.argwhere(vol_s > 1e-6)
        if pts.shape[0] == 0:
            return np.zeros((0,6))
        # downsample points if too many
        if pts.shape[0] > kmeans_k:
            rng = np.random.RandomState(0)
            sel = rng.choice(pts.shape[0], kmeans_k, replace=False)
            pts = pts[sel]
        intensities = vol_s[pts[:,0], pts[:,1], pts[:,2]]
        sigmas = np.clip(np.sqrt(intensities) * sigma_scale * max(D,H,W)/100.0, 0.5, max(D,H,W)/5.0)
        splats = np.concatenate([pts[:,[2,1,0]], sigmas[:,None], intensities[:,None], np.ones((pts.shape[0],1))], axis=1)
        return splats
