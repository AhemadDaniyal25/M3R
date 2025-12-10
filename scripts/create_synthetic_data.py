# scripts/create_synthetic_data.py
import numpy as np
import os
import nibabel as nib

def make_synthetic_volume(shape=(64,64,64)):
    vol = np.zeros(shape, dtype=np.float32)
    rng = np.random.RandomState(0)
    for _ in range(3):
        center = rng.randint(10, shape[0]-10, size=3)
        sigma = rng.uniform(3.0, 6.0)
        coords = np.indices(shape).astype(np.float32)
        d2 = ((coords[0]-center[0])**2 + (coords[1]-center[1])**2 + (coords[2]-center[2])**2)
        vol += np.exp(-d2/(2*sigma*sigma))
    vol = vol / vol.max()
    return vol

def save_nifti(vol, out_path):
    img = nib.Nifti1Image(vol, affine=np.eye(4))
    nib.save(img, out_path)

if __name__ == "__main__":
    out_dir = os.path.join("data", "sample_fastmri")
    os.makedirs(out_dir, exist_ok=True)
    vol = make_synthetic_volume((64,64,64))
    out_file = os.path.join(out_dir, "sample.nii.gz")
    save_nifti(vol, out_file)
    print("Wrote synthetic sample to", out_file)
