# src/baseline/interp.py
import numpy as np
import os
import nibabel as nib
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.interpolate import interp1d

def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata().astype(np.float32)

def sparse_slice_volume(vol, step):
    out = np.zeros_like(vol)
    out[::step] = vol[::step]
    return out

def cubic_interpolate_slices(sparse_vol):
    D,H,W = sparse_vol.shape
    xs = np.where(np.any(np.any(sparse_vol, axis=2), axis=1))[0]
    ys = sparse_vol[xs].reshape(len(xs), -1)
    f = interp1d(xs, ys, kind='cubic', axis=0, fill_value='extrapolate')
    all_idx = np.arange(D)
    interp_flat = f(all_idx)
    interp_vol = interp_flat.reshape(D,H,W)
    return interp_vol

def compute_metrics(gt, recon):
    D = gt.shape[0]
    s_vals = []
    p_vals = []
    for i in range(D):
        g = gt[i]
        r = recon[i]
        g_n = (g - g.min()) / (g.max() - g.min() + 1e-8)
        r_n = (r - r.min()) / (r.max() - r.min() + 1e-8)
        try:
            s = ssim(g_n, r_n, data_range=1.0)
        except Exception:
            s = 0.0
        p = psnr(g_n, r_n, data_range=1.0)
        s_vals.append(s); p_vals.append(p)
    return float(np.mean(s_vals)), float(np.mean(p_vals))

def demo(sample_path, step=4):
    vol = load_nifti(sample_path)
    sparse = sparse_slice_volume(vol, step)
    interp = cubic_interpolate_slices(sparse)
    s,p = compute_metrics(vol, interp)
    out_dir = os.path.dirname(sample_path)
    np.save(os.path.join(out_dir, "sparse.npy"), sparse)
    np.save(os.path.join(out_dir, "interp.npy"), interp)
    print(f"Baseline (step={step}): SSIM={s:.4f}, PSNR={p:.4f}")

if __name__ == "__main__":
    sample = os.path.join("data","sample_fastmri","sample.nii.gz")
    demo(sample, step=4)
