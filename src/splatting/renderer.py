# src/splatting/renderer.py
import numpy as np
from math import cos, sin, pi
import imageio

def rotate_points(points, angle_deg):
    a = angle_deg * pi/180.0
    R = np.array([[cos(a), -sin(a), 0],
                  [sin(a),  cos(a), 0],
                  [0,       0,      1]])
    return points @ R.T

def render_splats_image(splats, volume_shape, angle_deg=0, out_size=(512,512), sigma_pixels_scale=1.0):
    D,H,W = volume_shape
    if splats is None or splats.shape[0] == 0:
        return np.zeros(out_size, dtype=np.uint8)
    pts = splats[:, :3].astype(np.float32).copy()
    center = np.array([W/2.0, H/2.0, D/2.0], dtype=np.float32)
    pts_c = pts - center
    pts_r = rotate_points(pts_c, angle_deg)
    pts = pts_r + center
    xs = pts[:,0]; ys = pts[:,1]; sigmas = splats[:,3]; intens = splats[:,4]
    outH, outW = out_size
    scale_x = outW / float(W); scale_y = outH / float(H)
    img = np.zeros((outH, outW), dtype=np.float32)
    for x,y,s,i in zip(xs, ys, sigmas, intens):
        px = int(x * scale_x)
        py = int(y * scale_y)
        rad = max(1, int(s * max(scale_x, scale_y) * sigma_pixels_scale))
        x0 = max(0, px - 3*rad); x1 = min(outW-1, px + 3*rad)
        y0 = max(0, py - 3*rad); y1 = min(outH-1, py + 3*rad)
        if x1 < x0 or y1 < y0:
            continue
        xs_grid = np.arange(x0, x1+1)
        ys_grid = np.arange(y0, y1+1)
        gx = ((xs_grid + 0.5)/scale_x - x)**2
        gy = ((ys_grid + 0.5)/scale_y - y)**2
        g = np.exp(- (gx[None,:] + gy[:,None]) / (2*(s**2 + 1e-6)))
        img[y0:y1+1, x0:x1+1] += (g * i)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    img_uint8 = (img * 255).astype(np.uint8)
    return img_uint8

def save_image(img, path):
    imageio.imwrite(path, img)
