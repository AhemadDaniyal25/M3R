# src/app/demo_smoke.py
import numpy as np
from pathlib import Path

def main():
    print("M3R demo smoke test")
    x = np.zeros((1,64,64,64), dtype=np.float32)
    x[0,32,32,32] = 1.0
    print("sample volume shape:", x.shape)
    y = x + 0.1
    print("min/max:", y.min(), y.max())
    out = Path("data/sample_fastmri/demo_out.npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, y)
    print("wrote demo output to", out)

if __name__ == "__main__":
    main()
