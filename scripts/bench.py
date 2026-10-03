"""Times lumen against Pillow (zlib, libpng and libjpeg-turbo underneath) on a 1920x1080
synthetic photo: smooth gradients, edges and sensor-like noise. Single thread on both sides.

    python scripts/bench.py

Needs Pillow and NumPy. Pillow's PNG encoder is run at the same zlib level (6) and with the same
adaptive filtering idea; its JPEG encoder at quality 90, 4:2:0, optimize=True, like lumen's.
"""
import io
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent.parent
out = root / "target" / "bench"
out.mkdir(parents=True, exist_ok=True)

w, h = 1920, 1080
rng = np.random.default_rng(7)
y, x = np.mgrid[0:h, 0:w].astype(np.float64)
img = np.stack([
    128 + 90 * np.sin(x / 90.0) * np.cos(y / 70.0),
    255 * x / w,
    120 + 100 * np.sin((x + y) / 200.0),
], axis=-1)
disc = (x - 1200) ** 2 + (y - 400) ** 2 < 250 ** 2
img[disc] = [230, 40, 50]
img += rng.normal(0, 3, img.shape)
src = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")
src.save(out / "photo.png", compress_level=6)
src.save(out / "photo.jpg", quality=90, subsampling=2, optimize=True)


def median_ms(runs, f):
    f()
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        f()
        t.append((time.perf_counter() - s) * 1e3)
    return statistics.median(t)


png_bytes = (out / "photo.png").read_bytes()
jpg_bytes = (out / "photo.jpg").read_bytes()


def dec(b):
    im = Image.open(io.BytesIO(b))
    im.load()


pillow = {
    "png_decode": median_ms(15, lambda: dec(png_bytes)),
    "png_encode": median_ms(5, lambda: src.save(io.BytesIO(), "PNG", compress_level=6)),
    "jpeg_decode": median_ms(15, lambda: dec(jpg_bytes)),
    "jpeg_encode": median_ms(15, lambda: src.save(io.BytesIO(), "JPEG", quality=90, subsampling=2, optimize=True)),
    "resize_lanczos3": median_ms(15, lambda: src.resize((960, 540), Image.LANCZOS)),
}
r = subprocess.run(["cargo", "run", "--release", "--quiet", "--example", "bench", "--", str(out)],
                   cwd=root, capture_output=True, text=True, check=True)
lumen = json.loads(r.stdout)
b = io.BytesIO()
src.save(b, "PNG", compress_level=6)
print(f"sizes: Pillow png {len(b.getvalue())} bytes, jpeg {len(jpg_bytes)} bytes; lumen {r.stderr.strip()}")
print(f"| Operation (1920x1080 RGB) | lumen | Pillow | lumen / Pillow |")
print(f"|---|---:|---:|---:|")
for k in pillow:
    print(f"| {k} | {lumen[k]:.1f} ms | {pillow[k]:.1f} ms | {lumen[k] / pillow[k]:.2f}x |")
