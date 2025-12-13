import imageio
import os
import numpy as np

p = "data/sample_fastmri/splat_demo/nerf_density.png"
print("exists:", os.path.exists(p))
if os.path.exists(p):
    im = imageio.imread(p)
    print("shape:", im.shape, "dtype:", im.dtype)
    if im.ndim == 3:
        arr = im.mean(axis=2)
    else:
        arr = im
    print("min,max:", float(arr.min()), float(arr.max()))
    print("file bytes:", os.path.getsize(p))
