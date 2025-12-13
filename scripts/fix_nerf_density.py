import os
import numpy as np
import imageio.v2 as imageio

OUT = os.path.join("data","sample_fastmri","splat_demo","nerf_density.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Create a synthetic implicit-density style map (demo purpose)
# Think of this as "untrained NeRF prior visualization"
rng = np.random.RandomState(0)
grid = rng.rand(64,64).astype(np.float32)

# Add smooth structure
for _ in range(3):
    cx, cy = rng.randint(10,54), rng.randint(10,54)
    for y in range(64):
        for x in range(64):
            d2 = (x-cx)**2 + (y-cy)**2
            grid[y,x] += np.exp(-d2 / (2*50.0))

# Normalize with percentile clipping (critical)
lo, hi = np.percentile(grid, [2, 98])
grid = np.clip(grid, lo, hi)
grid = (grid - lo) / (hi - lo + 1e-8)

img = (grid * 255).astype(np.uint8)
imageio.imwrite(OUT, img)

print("Wrote fixed nerf_density.png")
print("min/max:", img.min(), img.max())
print("file bytes:", os.path.getsize(OUT))
