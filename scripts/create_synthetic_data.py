# scripts/create_synthetic_data.py
"""
Create a tiny synthetic 3D volume with a few gaussian blobs and save as NIfTI.
Saves to data/sample_fastmri/sample.nii.gz
"""
import numpy as np
import os
import nibabel as nib

def make_synthetic_volume(shape=(64,64,64), n_blobs=3, seed=0):
    vol = np.zeros(shape, dtype=np.float32)
    rng = np.random.RandomState(seed)
    D,H,W = shape
    coords = np.indices(shape).astype(np.float32)  # shape (3,D,H,W): 0->z,1->y,2->x if you index that way
    for _ in range(n_blobs):
        # pick center in (z,y,x) order to match vol indexing
        center = rng.randint(10, [D-10, H-10, W-10])
        sigma = float(rng.uniform(3.0, 6.0))
        dz = (coords[0] - center[0])**2
        dy = (coords[1] - center[1])**2
        dx = (coords[2] - center[2])**2
        d2 = dx + dy + dz
        vol += np.exp(-d2 / (2.0 * sigma * sigma))
    maxv = vol.max()
    if maxv <= 0:
        return vol.astype(np.float32)
    vol = vol / maxv
    return vol.astype(np.float32)

def save_nifti(vol, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = nib.Nifti1Image(vol, affine=np.eye(4))
    nib.save(img, out_path)

if __name__ == "__main__":
    out_dir = os.path.join("data", "sample_fastmri")
    out_file = os.path.join(out_dir, "sample.nii.gz")
    vol = make_synthetic_volume((64,64,64), n_blobs=3, seed=0)
    save_nifti(vol, out_file)
    print("Wrote synthetic sample to", out_file,    "shape:", vol.shape, " min/max:", vol.min(), vol.max())

