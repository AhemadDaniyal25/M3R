import numpy as np
import os
import nibabel as nib

def make_synthetic_volume(shape=(64,64,64)):
    D,H,W = shape
    vol = np.zeros(shape, dtype=np.float32)
    xs, ys = np.meshgrid(np.linspace(-1,1,W), np.linspace(-1,1,H))

    for z in range(D):
        # moving center along depth (NON-LINEAR)
        cx = 0.5 * np.sin(2 * np.pi * z / D)
        cy = 0.5 * np.cos(2 * np.pi * z / D)
        r = 0.3 + 0.1 * np.sin(4 * np.pi * z / D)

        blob = np.exp(-((xs-cx)**2 + (ys-cy)**2) / (2*r*r))
        vol[z] = blob

    vol = vol / vol.max()
    return vol

if __name__ == "__main__":
    out_dir = "data/sample_fastmri"
    os.makedirs(out_dir, exist_ok=True)
    vol = make_synthetic_volume()
    nib.save(nib.Nifti1Image(vol, affine=np.eye(4)),
             os.path.join(out_dir, "sample.nii.gz"))
    print("Wrote improved synthetic volume")
