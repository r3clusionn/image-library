"""Writes tests/pngsuite/reference.json: for every PngSuite file, what two other decoders make of it.

* Pillow: the CRC-32 of the pixels as 8-bit RGBA (`crc`) and as 8-bit RGB (`rgb_crc`); for
  16-bit gray, which Pillow keeps at 16 bits, the CRC-32 of the big-endian samples instead.
* pypng (pure Python), for files with a tRNS chunk: the CRC-32 of the 8-bit alpha channel
  (`trns_alpha_crc`). Pillow gets transparency wrong for gray images with tRNS at 4 and 16 bits
  (tbbn0g04, tbwn0g16), so the test takes their alpha from pypng.

Files Pillow refuses are marked. Needs Pillow and pypng.
"""
import json
import struct
import zlib
from pathlib import Path

import png as pypng
from PIL import Image

root = Path(__file__).resolve().parent.parent / "tests" / "pngsuite"


def has_trns(path):
    d = path.read_bytes()
    i = 8
    while i + 8 <= len(d):
        n = struct.unpack(">I", d[i:i + 4])[0]
        if d[i + 4:i + 8] == b"tRNS":
            return True
        i += 12 + n
    return False


out = {}
for p in sorted(root.glob("*.png")):
    try:
        im = Image.open(p)
        im.load()
    except Exception as e:  # noqa: BLE001
        out[p.name] = {"error": type(e).__name__}
        continue
    r = {"w": im.width, "h": im.height}
    if im.mode in ("I;16", "I;16B") and "transparency" not in im.info:
        r["kind"] = "gray16"
        r["crc"] = zlib.crc32(im.tobytes("raw", "I;16B"))
    else:
        if im.mode in ("I;16", "I;16B"):
            # Pillow's conversion of 16-bit gray to RGB clamps at 255 instead of scaling; its raw
            # samples are right, so keep those too.
            r["gray16_crc"] = zlib.crc32(im.tobytes("raw", "I;16B"))
        r["kind"] = "rgba8"
        r["crc"] = zlib.crc32(im.convert("RGBA").tobytes())
        rgb = im.convert("RGBA").convert("RGB").tobytes()
        r["rgb_crc"] = zlib.crc32(rgb)
    if has_trns(p):
        _, _, rows, _ = pypng.Reader(filename=str(p)).asRGBA8()
        alpha = bytes(v for row in rows for v in list(row)[3::4])
        r["trns_alpha_crc"] = zlib.crc32(alpha)
    out[p.name] = r
(root / "reference.json").write_text(json.dumps(out, indent=1, sort_keys=True))
print(len(out), "files,", sum(1 for v in out.values() if "error" in v), "refused by Pillow,",
      sum(1 for v in out.values() if "trns_alpha_crc" in v), "with tRNS checked by pypng")
