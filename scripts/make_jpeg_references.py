"""Writes tests/jpeg/: JPEG files made by Pillow (libjpeg-turbo) and the pixels Pillow decodes
from them, plus Pillow's resizes of the source image, for tests/jpeg.rs and tests/resize.rs.

Every `.jpg` gets a `.rgb` (or `.gray`) file next to it with Pillow's decoded samples. The
source image is synthetic (gradients, hard edges, noise) so the fixtures stay small and the
script needs nothing but Pillow and NumPy.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

out = Path(__file__).resolve().parent.parent / "tests" / "jpeg"
out.mkdir(parents=True, exist_ok=True)


def source(w, h, seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 128 + 100 * np.sin(x / 7.0) * np.cos(y / 11.0)
    g = 255 * x / max(w - 1, 1)
    b = 255 * y / max(h - 1, 1)
    img = np.stack([r, g, b], axis=-1)
    # A hard-edged disc and a bar: the content blocky codecs and ringing filters get wrong.
    disc = (x - w * 0.6) ** 2 + (y - h * 0.4) ** 2 < (min(w, h) * 0.25) ** 2
    img[disc] = [250, 30, 40]
    img[(x > w * 0.1) & (x < w * 0.2)] = [10, 10, 10]
    img += rng.normal(0, 6, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


cases = {}
src = source(67, 45, 1)
src.save(out / "source.png")
big = source(256, 192, 2)
big.save(out / "source_big.png")

variants = [
    ("q90_420", src, dict(quality=90, subsampling=2)),
    ("q90_422", src, dict(quality=90, subsampling=1)),
    ("q90_444", src, dict(quality=90, subsampling=0)),
    ("q50_420", src, dict(quality=50, subsampling=2)),
    ("q100_444", src, dict(quality=100, subsampling=0)),
    ("q75_optimized", src, dict(quality=75, optimize=True)),
    ("q75_restart", src, dict(quality=75, restart_marker_blocks=3)),
    ("gray", src.convert("L"), dict(quality=85)),
    ("tiny_1x1", src.crop((5, 5, 6, 6)), dict(quality=85)),
    ("odd_17x9", src.crop((3, 2, 20, 11)), dict(quality=85, subsampling=2)),
    ("big_420", big, dict(quality=85, subsampling=2)),
]
for name, im, kw in variants:
    p = out / f"{name}.jpg"
    im.save(p, "JPEG", **kw)
    dec = Image.open(p)
    dec.load()
    ext = "gray" if dec.mode == "L" else "rgb"
    (out / f"{name}.{ext}").write_bytes(dec.tobytes())
    cases[name] = {"w": dec.width, "h": dec.height, "mode": dec.mode}

Image.open(out / "q90_420.jpg").save(out / "progressive.jpg", "JPEG", quality=80, progressive=True)

# Pillow's resizes (RGB, no alpha): shrinking source_big.png, stretching, and enlarging source.png.
resizes = []
for filt, pf in [("bilinear", Image.BILINEAR), ("bicubic", Image.BICUBIC), ("lanczos3", Image.LANCZOS)]:
    for (name, im, w, h) in [("source_big", big, 100, 75), ("source_big", big, 37, 211), ("source", src, 150, 101)]:
        r = im.resize((w, h), pf)
        (out / f"resize_{filt}_{w}x{h}.rgb").write_bytes(r.tobytes())
        resizes.append({"src": name + ".png", "filter": filt, "w": w, "h": h})
(out / "cases.json").write_text(json.dumps({"jpeg": cases, "resize": resizes}, indent=1))
print(len(cases), "JPEG cases,", len(resizes), "resizes")
