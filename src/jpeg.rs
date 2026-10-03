//! Baseline JPEG: decoding (SOF0 and SOF1: Huffman, 8-bit, any sampling factors, restart
//! intervals, grayscale and YCbCr) and encoding (4:2:0 or 4:4:4, libjpeg's quality scaling,
//! standard or per-image optimized Huffman tables).
//!
//! The decoder follows libjpeg's defaults where they show in the pixels: an accurate IDCT,
//! "fancy" (triangle) upsampling of subsampled chroma, and BT.601 full-range YCbCr. Progressive
//! and arithmetic-coded files and CMYK are reported as unsupported.

use crate::deflate::code_lengths;
use crate::image::{Color, Image, Pixels};
use crate::Error;

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42,
    49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

// ---- IDCT and FDCT ---------------------------------------------------------------------------

fn dct_matrix() -> &'static [[f32; 8]; 8] {
    static M: std::sync::OnceLock<[[f32; 8]; 8]> = std::sync::OnceLock::new();
    M.get_or_init(|| {
        let mut m = [[0f32; 8]; 8];
        for (u, row) in m.iter_mut().enumerate() {
            let c = if u == 0 { (0.125f64).sqrt() } else { 0.5 };
            for (x, v) in row.iter_mut().enumerate() {
                *v = (c * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0).cos()) as f32;
            }
        }
        m
    })
}

/// Rounds to the nearest sample. Clamping first lets truncation round (`f32::round` is a libm
/// call on baseline x86-64).
#[inline]
fn to_u8(v: f32) -> u8 {
    (v + 0.5).clamp(0.0, 255.0) as u8
}

/// Inverse DCT of one block of dequantized coefficients (natural order) to samples 0..255.
fn idct(coef: &[i32; 64], out: &mut [u8; 64]) {
    let m = dct_matrix();
    let mut tmp = [0f32; 64];
    // Rows (over u), then columns (over v): f(x, y) = sum_u sum_v C(u) C(v) F(v, u) ...
    for v in 0..8 {
        if coef[v * 8 + 1..v * 8 + 8].iter().all(|c| *c == 0) {
            let dc = coef[v * 8] as f32 * m[0][0];
            tmp[v * 8..v * 8 + 8].fill(dc);
            continue;
        }
        for x in 0..8 {
            let mut s = 0f32;
            for u in 0..8 {
                s += m[u][x] * coef[v * 8 + u] as f32;
            }
            tmp[v * 8 + x] = s;
        }
    }
    for x in 0..8 {
        for y in 0..8 {
            let mut s = 0f32;
            for v in 0..8 {
                s += m[v][y] * tmp[v * 8 + x];
            }
            out[y * 8 + x] = to_u8(s + 128.0);
        }
    }
}

/// Forward DCT of one block of samples (already shifted by -128).
fn fdct(block: &[f32; 64], out: &mut [f32; 64]) {
    let m = dct_matrix();
    let mut tmp = [0f32; 64];
    for y in 0..8 {
        for u in 0..8 {
            let mut s = 0f32;
            for x in 0..8 {
                s += m[u][x] * block[y * 8 + x];
            }
            tmp[y * 8 + u] = s;
        }
    }
    for u in 0..8 {
        for v in 0..8 {
            let mut s = 0f32;
            for y in 0..8 {
                s += m[v][y] * tmp[y * 8 + u];
            }
            out[v * 8 + u] = s;
        }
    }
}

// ---- decoding ---------------------------------------------------------------------------------

#[derive(Clone)]
struct HuffTable {
    /// Fast path: next 9 bits -> (length << 8 | symbol), 0 if longer.
    fast: Vec<u16>,
    maxcode: [i32; 18],
    valptr: [i32; 17],
    mincode: [i32; 17],
    values: Vec<u8>,
}

impl HuffTable {
    fn new(counts: &[u8; 16], values: &[u8]) -> Result<HuffTable, Error> {
        let mut t =
            HuffTable { fast: vec![0; 512], maxcode: [-1; 18], valptr: [0; 17], mincode: [0; 17], values: values.to_vec() };
        let mut code = 0i32;
        let mut k = 0usize;
        for l in 1..=16 {
            let n = counts[l - 1] as usize;
            t.valptr[l] = k as i32;
            t.mincode[l] = code;
            for _ in 0..n {
                if l <= 9 {
                    let shift = 9 - l;
                    for f in 0..(1 << shift) {
                        let idx = ((code << shift) | f) as usize;
                        if idx < 512 {
                            t.fast[idx] = (l as u16) << 8 | values[k] as u16;
                        }
                    }
                }
                code += 1;
                k += 1;
            }
            t.maxcode[l] = if n > 0 { code - 1 } else { -1 };
            if code > (1 << l) {
                return Err(Error::Corrupt("bad Huffman table"));
            }
            code <<= 1;
        }
        t.maxcode[17] = i32::MAX;
        Ok(t)
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    n: u32,
    /// A marker was reached; only zeros are fed from here.
    marker: Option<u8>,
}

impl<'a> Reader<'a> {
    fn fill(&mut self) {
        while self.n <= 56 {
            let mut b = 0u8;
            if self.marker.is_none() && self.pos < self.data.len() {
                b = self.data[self.pos];
                if b == 0xff {
                    let next = self.data.get(self.pos + 1).copied().unwrap_or(0);
                    if next == 0 {
                        self.pos += 2;
                    } else {
                        self.marker = Some(next);
                        b = 0;
                    }
                } else {
                    self.pos += 1;
                }
            }
            self.buf |= (b as u64) << (56 - self.n);
            self.n += 8;
        }
    }

    fn bits(&mut self, k: u32) -> u32 {
        if k == 0 {
            return 0;
        }
        if self.n < k {
            self.fill();
        }
        let v = (self.buf >> (64 - k)) as u32;
        self.buf <<= k;
        self.n -= k;
        v
    }

    fn decode(&mut self, t: &HuffTable) -> Result<u8, Error> {
        if self.n < 16 {
            self.fill();
        }
        let f = t.fast[(self.buf >> 55) as usize];
        if f != 0 {
            let l = (f >> 8) as u32;
            self.buf <<= l;
            self.n -= l;
            return Ok(f as u8);
        }
        let mut code = 0i32;
        for l in 1..=16 {
            code = (code << 1) | self.bits(1) as i32;
            if code <= t.maxcode[l] {
                let i = t.valptr[l] + code - t.mincode[l];
                return t.values.get(i as usize).copied().ok_or(Error::Corrupt("bad Huffman code"));
            }
        }
        Err(Error::Corrupt("bad Huffman code"))
    }

    /// The signed value of a `size`-bit magnitude (JPEG's EXTEND).
    fn receive(&mut self, size: u32) -> i32 {
        if size == 0 {
            return 0;
        }
        let v = self.bits(size) as i32;
        if v < 1 << (size - 1) {
            v - (1 << size) + 1
        } else {
            v
        }
    }

    /// Skips to just after the next RSTn marker.
    fn restart(&mut self) -> Result<(), Error> {
        self.buf = 0;
        self.n = 0;
        if self.marker.is_none() {
            // Find the marker in the data.
            while self.pos + 1 < self.data.len()
                && !(self.data[self.pos] == 0xff && self.data[self.pos + 1] != 0 && self.data[self.pos + 1] != 0xff)
            {
                self.pos += 1;
            }
            self.marker = self.data.get(self.pos + 1).copied();
        }
        match self.marker {
            Some(m) if (0xd0..=0xd7).contains(&m) => {
                self.pos += 2;
                self.marker = None;
                Ok(())
            }
            _ => Err(Error::Corrupt("missing restart marker")),
        }
    }
}

#[derive(Clone)]
struct Component {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    td: usize,
    ta: usize,
    /// Decoded samples of the padded plane.
    plane: Vec<u8>,
    bw: usize,
    bh: usize,
}

fn be16(d: &[u8], at: usize) -> Result<usize, Error> {
    Ok(u16::from_be_bytes(d.get(at..at + 2).ok_or(Error::Corrupt("truncated segment"))?.try_into().unwrap()) as usize)
}

/// libjpeg's fancy upsampling by 2 along a row: 3/4 of the nearer sample plus 1/4 of the other.
fn upsample_h2(src: &[u8], w: usize, out: &mut [u16]) {
    // Output values are kept times 4 for the vertical step.
    let n = src.len();
    for i in 0..n {
        let c = src[i] as u16 * 3;
        let l = src[i.saturating_sub(1)] as u16;
        let r = src[(i + 1).min(n - 1)] as u16;
        if 2 * i < w {
            out[2 * i] = c + l;
        }
        if 2 * i + 1 < w {
            out[2 * i + 1] = c + r;
        }
    }
}

pub fn decode(data: &[u8]) -> Result<Image, Error> {
    if data.get(..2) != Some(&[0xff, 0xd8]) {
        return Err(Error::Corrupt("not a JPEG file"));
    }
    let mut qt = [[0u16; 64]; 4];
    let mut dc: Vec<Option<HuffTable>> = vec![None; 4];
    let mut ac: Vec<Option<HuffTable>> = vec![None; 4];
    let mut comps: Vec<Component> = Vec::new();
    let (mut width, mut height) = (0usize, 0usize);
    let mut restart = 0usize;
    let mut adobe_transform: Option<u8> = None;
    let mut pos = 2;
    loop {
        // Skip fill bytes before a marker.
        while data.get(pos) == Some(&0xff) && data.get(pos + 1) == Some(&0xff) {
            pos += 1;
        }
        if data.get(pos) != Some(&0xff) {
            return Err(Error::Corrupt("expected a marker"));
        }
        let m = *data.get(pos + 1).ok_or(Error::Corrupt("truncated file"))?;
        pos += 2;
        if m == 0xd9 {
            return Err(Error::Corrupt("no scan before the end of the image"));
        }
        if (0xd0..=0xd7).contains(&m) || m == 0x01 {
            continue;
        }
        let len = be16(data, pos)?;
        let seg = data.get(pos + 2..pos + len).ok_or(Error::Corrupt("truncated segment"))?;
        pos += len;
        match m {
            0xdb => {
                let mut i = 0;
                while i < seg.len() {
                    let (pq, tq) = (seg[i] >> 4, (seg[i] & 15) as usize);
                    if tq > 3 {
                        return Err(Error::Corrupt("bad quantization table id"));
                    }
                    i += 1;
                    for k in 0..64 {
                        let v = if pq == 0 {
                            *seg.get(i + k).ok_or(Error::Corrupt("truncated DQT"))? as u16
                        } else {
                            be16(seg, i + 2 * k)? as u16
                        };
                        qt[tq][ZIGZAG[k]] = v;
                    }
                    i += if pq == 0 { 64 } else { 128 };
                }
            }
            0xc4 => {
                let mut i = 0;
                while i < seg.len() {
                    let (tc, th) = (seg[i] >> 4, (seg[i] & 15) as usize);
                    if th > 3 || tc > 1 {
                        return Err(Error::Corrupt("bad Huffman table id"));
                    }
                    let counts: [u8; 16] = seg.get(i + 1..i + 17).ok_or(Error::Corrupt("truncated DHT"))?.try_into().unwrap();
                    let n: usize = counts.iter().map(|c| *c as usize).sum();
                    let values = seg.get(i + 17..i + 17 + n).ok_or(Error::Corrupt("truncated DHT"))?;
                    let t = HuffTable::new(&counts, values)?;
                    if tc == 0 {
                        dc[th] = Some(t);
                    } else {
                        ac[th] = Some(t);
                    }
                    i += 17 + n;
                }
            }
            0xc0 | 0xc1 => {
                if seg.len() < 6 || seg[0] != 8 {
                    return Err(Error::Unsupported("JPEG sample precision other than 8 bits"));
                }
                height = be16(seg, 1)?;
                width = be16(seg, 3)?;
                let n = seg[5] as usize;
                if width == 0 || height == 0 {
                    return Err(Error::Unsupported("JPEG with height set by a DNL marker"));
                }
                if n != 1 && n != 3 {
                    return Err(Error::Unsupported("JPEG with other than 1 or 3 components (CMYK)"));
                }
                for c in 0..n {
                    let b = seg.get(6 + c * 3..9 + c * 3).ok_or(Error::Corrupt("truncated SOF"))?;
                    let (h, v) = ((b[1] >> 4) as usize, (b[1] & 15) as usize);
                    if !(1..=4).contains(&h) || !(1..=4).contains(&v) || b[2] > 3 {
                        return Err(Error::Corrupt("bad sampling factors"));
                    }
                    comps.push(Component { id: b[0], h, v, tq: b[2] as usize, td: 0, ta: 0, plane: Vec::new(), bw: 0, bh: 0 });
                }
                if width as u64 * height as u64 > 1 << 28 {
                    return Err(Error::TooLarge);
                }
            }
            0xc2 | 0xc6 | 0xca | 0xce => return Err(Error::Unsupported("progressive JPEG")),
            0xc3 | 0xc5 | 0xc7 | 0xcb | 0xcd | 0xcf | 0xc9 => {
                return Err(Error::Unsupported("lossless, hierarchical or arithmetic-coded JPEG"))
            }
            0xdd => restart = be16(seg, 0)?,
            0xee => {
                if seg.len() >= 12 && &seg[..5] == b"Adobe" {
                    adobe_transform = Some(seg[11]);
                }
            }
            0xda => {
                if comps.is_empty() {
                    return Err(Error::Corrupt("scan before frame header"));
                }
                let ns = *seg.first().ok_or(Error::Corrupt("empty scan header"))? as usize;
                if ns != comps.len() {
                    return Err(Error::Unsupported("JPEG with several scans (non-interleaved)"));
                }
                for s in 0..ns {
                    let id = *seg.get(1 + s * 2).ok_or(Error::Corrupt("truncated scan header"))?;
                    let t = *seg.get(2 + s * 2).ok_or(Error::Corrupt("truncated scan header"))?;
                    let c = comps.iter_mut().find(|c| c.id == id).ok_or(Error::Corrupt("scan names an unknown component"))?;
                    c.td = (t >> 4) as usize;
                    c.ta = (t & 15) as usize;
                    if c.td > 3 || c.ta > 3 {
                        return Err(Error::Corrupt("bad Huffman table id in scan"));
                    }
                }
                decode_scan(&data[pos..], &mut comps, width, height, &qt, &dc, &ac, restart)?;
                return finish(width, height, &comps, adobe_transform);
            }
            _ => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_scan(
    data: &[u8],
    comps: &mut [Component],
    width: usize,
    height: usize,
    qt: &[[u16; 64]; 4],
    dc: &[Option<HuffTable>],
    ac: &[Option<HuffTable>],
    restart: usize,
) -> Result<(), Error> {
    let hmax = comps.iter().map(|c| c.h).max().unwrap();
    let vmax = comps.iter().map(|c| c.v).max().unwrap();
    let mcux = width.div_ceil(8 * hmax);
    let mcuy = height.div_ceil(8 * vmax);
    for c in comps.iter_mut() {
        c.bw = mcux * c.h;
        c.bh = mcuy * c.v;
        c.plane = vec![0; c.bw * 8 * c.bh * 8];
    }
    let tables: Vec<(&HuffTable, &HuffTable, &[u16; 64])> = comps
        .iter()
        .map(|c| -> Result<_, Error> {
            Ok((
                dc[c.td].as_ref().ok_or(Error::Corrupt("missing DC table"))?,
                ac[c.ta].as_ref().ok_or(Error::Corrupt("missing AC table"))?,
                &qt[c.tq],
            ))
        })
        .collect::<Result<_, _>>()?;
    let mut r = Reader { data, pos: 0, buf: 0, n: 0, marker: None };
    let mut pred = vec![0i32; comps.len()];
    let mut coef = [0i32; 64];
    let mut px = [0u8; 64];
    let mut todo = restart;
    for my in 0..mcuy {
        for mx in 0..mcux {
            if restart > 0 {
                if todo == 0 {
                    r.restart()?;
                    pred.iter_mut().for_each(|p| *p = 0);
                    todo = restart;
                }
                todo -= 1;
            }
            for (ci, c) in comps.iter_mut().enumerate() {
                let (dct, act, q) = tables[ci];
                for by in 0..c.v {
                    for bx in 0..c.h {
                        coef.fill(0);
                        let t = r.decode(dct)? as u32;
                        if t > 11 {
                            return Err(Error::Corrupt("bad DC size"));
                        }
                        pred[ci] += r.receive(t);
                        coef[0] = pred[ci] * q[0] as i32;
                        let mut k = 1;
                        while k < 64 {
                            let rs = r.decode(act)?;
                            let (run, size) = ((rs >> 4) as usize, (rs & 15) as u32);
                            if size == 0 {
                                if run == 15 {
                                    k += 16;
                                    continue;
                                }
                                break;
                            }
                            k += run;
                            if k > 63 {
                                return Err(Error::Corrupt("coefficient index out of range"));
                            }
                            let z = ZIGZAG[k];
                            coef[z] = r.receive(size) * q[z] as i32;
                            k += 1;
                        }
                        idct(&coef, &mut px);
                        let stride = c.bw * 8;
                        let (x0, y0) = ((mx * c.h + bx) * 8, (my * c.v + by) * 8);
                        for y in 0..8 {
                            c.plane[(y0 + y) * stride + x0..(y0 + y) * stride + x0 + 8].copy_from_slice(&px[y * 8..y * 8 + 8]);
                        }
                    }
                }
            }
        }
        if r.pos > data.len() + 8 {
            return Err(Error::Corrupt("unexpected end of data"));
        }
    }
    Ok(())
}

/// Upsamples chroma (fancy upsampling for 2x, nearest otherwise), converts to RGB, crops.
fn finish(width: usize, height: usize, comps: &[Component], adobe: Option<u8>) -> Result<Image, Error> {
    if comps.len() == 1 {
        let c = &comps[0];
        let stride = c.bw * 8;
        let mut out = Vec::with_capacity(width * height);
        for y in 0..height {
            out.extend_from_slice(&c.plane[y * stride..y * stride + width]);
        }
        return Ok(Image { width: width as u32, height: height as u32, color: Color::Gray, pixels: Pixels::U8(out) });
    }
    let hmax = comps.iter().map(|c| c.h).max().unwrap();
    let vmax = comps.iter().map(|c| c.v).max().unwrap();
    // Each component at full resolution.
    let full: Vec<Vec<u8>> = comps
        .iter()
        .map(|c| {
            let (sx, sy) = (hmax / c.h, vmax / c.v);
            let stride = c.bw * 8;
            // Rows and columns of this component that carry image data.
            let cw = width.div_ceil(sx);
            let ch = height.div_ceil(sy);
            let mut out = vec![0u8; width * height];
            if sx == 1 && sy == 1 {
                for y in 0..height {
                    out[y * width..(y + 1) * width].copy_from_slice(&c.plane[y * stride..y * stride + width]);
                }
            } else if sx == 2 && (sy == 1 || sy == 2) {
                // libjpeg's h2v1 and h2v2 fancy upsampling.
                let mut rows: Vec<Vec<u16>> = Vec::with_capacity(ch);
                for y in 0..ch {
                    let mut r = vec![0u16; width];
                    upsample_h2(&c.plane[y * stride..y * stride + cw], width, &mut r);
                    rows.push(r);
                }
                for y in 0..height {
                    let row = &mut out[y * width..(y + 1) * width];
                    if sy == 1 {
                        for (x, o) in row.iter_mut().enumerate() {
                            *o = ((rows[y][x] + 2) >> 2) as u8;
                        }
                    } else {
                        let cy = y / 2;
                        let other = if y % 2 == 0 { cy.saturating_sub(1) } else { (cy + 1).min(ch - 1) };
                        for (x, o) in row.iter_mut().enumerate() {
                            let v = 3 * rows[cy][x] as u32 + rows[other][x] as u32;
                            // libjpeg rounds alternately (+8 or +7) to avoid a bias.
                            *o = ((v + if x % 2 == 0 { 8 } else { 7 }) >> 4) as u8;
                        }
                    }
                }
            } else {
                for y in 0..height {
                    for x in 0..width {
                        out[y * width + x] = c.plane[(y / sy) * stride + x / sx];
                    }
                }
            }
            out
        })
        .collect();
    // Adobe transform 0 means the components are RGB already.
    let rgb_already = adobe == Some(0);
    let mut out = Vec::with_capacity(width * height * 3);
    for ((&y, &cb), &cr) in full[0].iter().zip(&full[1]).zip(&full[2]) {
        if rgb_already {
            out.extend([y, cb, cr]);
            continue;
        }
        let (y, cb, cr) = (y as f32, cb as f32 - 128.0, cr as f32 - 128.0);
        let r = y + 1.402 * cr;
        let g = y - 0.344_136 * cb - 0.714_136 * cr;
        let b = y + 1.772 * cb;
        out.extend([to_u8(r), to_u8(g), to_u8(b)]);
    }
    Ok(Image { width: width as u32, height: height as u32, color: Color::Rgb, pixels: Pixels::U8(out) })
}

// ---- encoding ---------------------------------------------------------------------------------

const STD_LUMA_Q: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80,
    62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98,
    112, 100, 103, 99,
];
const STD_CHROMA_Q: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99,
];

// Annex K standard Huffman tables: (counts, values).
const DC_LUMA: ([u8; 16], &[u8]) = ([0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
const DC_CHROMA: ([u8; 16], &[u8]) = ([0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
const AC_LUMA: ([u8; 16], &[u8]) = (
    [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d],
    &[
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32,
        0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
        0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45,
        0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94,
        0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
        0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8,
        0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa,
    ],
);
const AC_CHROMA: ([u8; 16], &[u8]) = (
    [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77],
    &[
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81,
        0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
        0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44,
        0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92,
        0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
        0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6,
        0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa,
    ],
);

/// libjpeg's quality scaling of a base table.
fn scaled(base: &[u16; 64], quality: u8) -> [u16; 64] {
    let q = quality.clamp(1, 100) as u32;
    let s = if q < 50 { 5000 / q } else { 200 - 2 * q };
    let mut out = [0u16; 64];
    for i in 0..64 {
        out[i] = ((base[i] as u32 * s + 50) / 100).clamp(1, 255) as u16;
    }
    out
}

/// Code (value, length) per symbol from (counts, values).
fn codes(spec: &([u8; 16], Vec<u8>)) -> Vec<(u16, u8)> {
    let mut table = vec![(0u16, 0u8); 256];
    let mut code = 0u16;
    let mut k = 0;
    for l in 1..=16 {
        for _ in 0..spec.0[l - 1] {
            table[spec.1[k] as usize] = (code, l as u8);
            code += 1;
            k += 1;
        }
        code <<= 1;
    }
    table
}

/// An optimized table for symbol frequencies: lengths limited to 16, and one spare code so no
/// real code is all ones (as JPEG requires).
fn optimized(freq: &[u32; 256]) -> ([u8; 16], Vec<u8>) {
    let mut f: Vec<u32> = freq.to_vec();
    f.push(1); // reserved
    let lens = code_lengths(&f, 16);
    let mut counts = [0u8; 16];
    let mut by_len: Vec<(u8, usize)> =
        lens.iter().enumerate().filter(|(s, l)| **l > 0 && *s < 256).map(|(s, l)| (*l, s)).collect();
    by_len.sort();
    for (l, _) in &by_len {
        counts[*l as usize - 1] += 1;
    }
    (counts, by_len.into_iter().map(|(_, s)| s as u8).collect())
}

#[derive(Clone, Copy, Debug)]
pub struct JpegOptions {
    pub quality: u8,
    /// 4:2:0 chroma subsampling (default) or 4:4:4.
    pub subsample: bool,
    /// Per-image Huffman tables (two passes; smaller files).
    pub optimize: bool,
}

impl Default for JpegOptions {
    fn default() -> Self {
        JpegOptions { quality: 90, subsample: true, optimize: true }
    }
}

struct Writer {
    out: Vec<u8>,
    buf: u32,
    n: u32,
}

impl Writer {
    fn put(&mut self, v: u32, k: u32) {
        if k == 0 {
            return;
        }
        self.buf = (self.buf << k) | (v & ((1 << k) - 1));
        self.n += k;
        while self.n >= 8 {
            let b = (self.buf >> (self.n - 8)) as u8;
            self.out.push(b);
            if b == 0xff {
                self.out.push(0);
            }
            self.n -= 8;
        }
    }

    fn flush(&mut self) {
        if self.n > 0 {
            self.put(0x7f, 8 - self.n);
        }
    }
}

fn magnitude(v: i32) -> (u32, u32) {
    if v == 0 {
        return (0, 0);
    }
    let size = 32 - v.unsigned_abs().leading_zeros();
    let bits = if v < 0 { (v - 1) as u32 } else { v as u32 };
    (size, bits & ((1 << size) - 1))
}

pub fn encode(img: &Image, opt: JpegOptions) -> Vec<u8> {
    let gray = matches!(img.color, Color::Gray | Color::GrayAlpha);
    let src = img.convert(if gray { Color::Gray } else { Color::Rgb });
    let (w, h) = (src.width as usize, src.height as usize);
    let d = src.data8();
    let ncomp = if gray { 1 } else { 3 };
    // Planes: Y, Cb, Cr (BT.601 full range).
    let mut planes = vec![vec![0f32; w * h]; ncomp];
    for i in 0..w * h {
        if gray {
            planes[0][i] = d[i] as f32;
        } else {
            let (r, g, b) = (d[i * 3] as f32, d[i * 3 + 1] as f32, d[i * 3 + 2] as f32);
            planes[0][i] = 0.299 * r + 0.587 * g + 0.114 * b;
            planes[1][i] = -0.168_736 * r - 0.331_264 * g + 0.5 * b + 128.0;
            planes[2][i] = 0.5 * r - 0.418_688 * g - 0.081_312 * b + 128.0;
        }
    }
    let sub = !gray && opt.subsample;
    let (hmax, vmax) = if sub { (2, 2) } else { (1, 1) };
    let mcux = w.div_ceil(8 * hmax);
    let mcuy = h.div_ceil(8 * vmax);
    let lq = scaled(&STD_LUMA_Q, opt.quality);
    let cq = scaled(&STD_CHROMA_Q, opt.quality);
    // Quantized coefficient blocks in scan order: (component, zigzag coefficients).
    let mut blocks: Vec<(usize, [i32; 64])> = Vec::with_capacity(mcux * mcuy * (hmax * vmax + 2));
    let sample = |c: usize, x: usize, y: usize, scale: usize| -> f32 {
        // Chroma at half resolution is the average of a 2x2 block; edges repeat the last pixel.
        if scale == 1 {
            planes[c][y.min(h - 1) * w + x.min(w - 1)]
        } else {
            let mut s = 0.0;
            for dy in 0..2 {
                for dx in 0..2 {
                    s += planes[c][(y * 2 + dy).min(h - 1) * w + (x * 2 + dx).min(w - 1)];
                }
            }
            s / 4.0
        }
    };
    let mut blk = [0f32; 64];
    let mut out = [0f32; 64];
    for my in 0..mcuy {
        for mx in 0..mcux {
            for c in 0..ncomp {
                let (nh, nv, scale) = if c == 0 { (hmax, vmax, 1) } else { (1, 1, hmax) };
                let q = if c == 0 { &lq } else { &cq };
                for by in 0..nv {
                    for bx in 0..nh {
                        let (x0, y0) = ((mx * nh + bx) * 8, (my * nv + by) * 8);
                        for y in 0..8 {
                            for x in 0..8 {
                                blk[y * 8 + x] = sample(c, x0 + x, y0 + y, scale) - 128.0;
                            }
                        }
                        fdct(&blk, &mut out);
                        let mut z = [0i32; 64];
                        for k in 0..64 {
                            let n = ZIGZAG[k];
                            let v = out[n] / q[n] as f32;
                            z[k] = (v + 0.5f32.copysign(v)) as i32;
                        }
                        blocks.push((c, z));
                    }
                }
            }
        }
    }
    // Huffman tables: standard, or built from this image's symbol counts.
    let (dcl, acl, dcc, acc) = if opt.optimize {
        let mut f = [[0u32; 256]; 4];
        let mut pred = [0i32; 3];
        for (c, z) in &blocks {
            let t = if *c == 0 { 0 } else { 2 };
            let (s, _) = magnitude(z[0] - pred[*c]);
            pred[*c] = z[0];
            f[t][s as usize] += 1;
            let mut run = 0;
            for &v in &z[1..64] {
                if v == 0 {
                    run += 1;
                    continue;
                }
                while run > 15 {
                    f[t + 1][0xf0] += 1;
                    run -= 16;
                }
                let (s, _) = magnitude(v);
                f[t + 1][(run << 4 | s) as usize] += 1;
                run = 0;
            }
            if run > 0 {
                f[t + 1][0] += 1;
            }
        }
        (optimized(&f[0]), optimized(&f[1]), optimized(&f[2]), optimized(&f[3]))
    } else {
        let s = |t: &([u8; 16], &[u8])| (t.0, t.1.to_vec());
        (s(&DC_LUMA), s(&AC_LUMA), s(&DC_CHROMA), s(&AC_CHROMA))
    };

    let mut o = vec![0xff, 0xd8];
    let seg = |o: &mut Vec<u8>, m: u8, body: &[u8]| {
        o.extend([0xff, m]);
        o.extend(((body.len() + 2) as u16).to_be_bytes());
        o.extend_from_slice(body);
    };
    seg(&mut o, 0xe0, b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0");
    let mut dqt = vec![0u8];
    dqt.extend(ZIGZAG.iter().map(|n| lq[*n] as u8));
    if !gray {
        dqt.push(1);
        dqt.extend(ZIGZAG.iter().map(|n| cq[*n] as u8));
    }
    seg(&mut o, 0xdb, &dqt);
    let mut sof = vec![8];
    sof.extend((h as u16).to_be_bytes());
    sof.extend((w as u16).to_be_bytes());
    sof.push(ncomp as u8);
    sof.extend([1, ((hmax << 4) | vmax) as u8, 0]);
    if !gray {
        sof.extend([2, 0x11, 1, 3, 0x11, 1]);
    }
    seg(&mut o, 0xc0, &sof);
    let mut dht = Vec::new();
    let mut add = |class_id: u8, t: &([u8; 16], Vec<u8>)| {
        dht.push(class_id);
        dht.extend(t.0);
        dht.extend(&t.1);
    };
    add(0x00, &dcl);
    add(0x10, &acl);
    if !gray {
        add(0x01, &dcc);
        add(0x11, &acc);
    }
    seg(&mut o, 0xc4, &dht);
    let mut sos = vec![ncomp as u8, 1, 0x00];
    if !gray {
        sos.extend([2, 0x11, 3, 0x11]);
    }
    sos.extend([0, 63, 0]);
    seg(&mut o, 0xda, &sos);

    let tabs = [(codes(&dcl), codes(&acl)), (codes(&dcc), codes(&acc))];
    let mut wr = Writer { out: o, buf: 0, n: 0 };
    let mut pred = [0i32; 3];
    for (c, z) in &blocks {
        let (dct, act) = &tabs[(*c != 0) as usize];
        let (s, bits) = magnitude(z[0] - pred[*c]);
        pred[*c] = z[0];
        let (code, len) = dct[s as usize];
        wr.put(code as u32, len as u32);
        wr.put(bits, s);
        let mut run = 0u32;
        for &v in &z[1..64] {
            if v == 0 {
                run += 1;
                continue;
            }
            while run > 15 {
                let (code, len) = act[0xf0];
                wr.put(code as u32, len as u32);
                run -= 16;
            }
            let (s, bits) = magnitude(v);
            let (code, len) = act[(run << 4 | s) as usize];
            wr.put(code as u32, len as u32);
            wr.put(bits, s);
            run = 0;
        }
        if run > 0 {
            let (code, len) = act[0];
            wr.put(code as u32, len as u32);
        }
    }
    wr.flush();
    let mut o = wr.out;
    o.extend([0xff, 0xd9]);
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: u32, h: u32) -> Image {
        let mut d = Vec::new();
        for y in 0..h {
            for x in 0..w {
                d.extend([(x * 255 / w.max(1)) as u8, (y * 255 / h.max(1)) as u8, ((x + y) * 3 % 256) as u8]);
            }
        }
        Image::new8(w, h, Color::Rgb, d).unwrap()
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let mse: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>() / a.len() as f64;
        10.0 * (255.0 * 255.0 / mse.max(1e-9)).log10()
    }

    #[test]
    fn dct_round_trip_is_exact_enough() {
        let mut b = [0f32; 64];
        for (i, v) in b.iter_mut().enumerate() {
            *v = ((i * 37) % 255) as f32 - 128.0;
        }
        let mut f = [0f32; 64];
        fdct(&b, &mut f);
        let coef: [i32; 64] = std::array::from_fn(|i| f[i].round() as i32);
        let mut px = [0u8; 64];
        idct(&coef, &mut px);
        for i in 0..64 {
            assert!((px[i] as f32 - (b[i] + 128.0)).abs() <= 1.0, "{i}");
        }
    }

    #[test]
    fn encode_then_decode_at_odd_sizes_and_both_samplings() {
        for (w, h) in [(1, 1), (7, 9), (16, 16), (33, 17), (100, 75)] {
            for subsample in [false, true] {
                for optimize in [false, true] {
                    let img = gradient(w, h);
                    let j = encode(&img, JpegOptions { quality: 95, subsample, optimize });
                    let back = decode(&j).unwrap();
                    assert_eq!((back.width, back.height, back.color), (w, h, Color::Rgb));
                    let p = psnr(img.data8(), back.data8());
                    assert!(p > 30.0, "{w}x{h} subsample {subsample}: PSNR {p:.1}");
                }
            }
        }
        let g = gradient(40, 30).convert(Color::Gray);
        let back = decode(&encode(&g, JpegOptions::default())).unwrap();
        assert_eq!(back.color, Color::Gray);
        assert!(psnr(g.data8(), back.data8()) > 38.0);
    }

    #[test]
    fn quality_trades_size_for_fidelity_and_optimizing_shrinks_files() {
        let img = gradient(128, 96);
        let lo = encode(&img, JpegOptions { quality: 30, ..Default::default() });
        let hi = encode(&img, JpegOptions { quality: 95, ..Default::default() });
        assert!(lo.len() < hi.len());
        let p_lo = psnr(img.data8(), decode(&lo).unwrap().data8());
        let p_hi = psnr(img.data8(), decode(&hi).unwrap().data8());
        assert!(p_hi > p_lo + 3.0, "{p_lo} {p_hi}");
        let std = encode(&img, JpegOptions { optimize: false, ..Default::default() });
        let opt = encode(&img, JpegOptions { optimize: true, ..Default::default() });
        assert!(opt.len() < std.len(), "{} {}", opt.len(), std.len());
    }

    #[test]
    fn bad_input_is_an_error() {
        assert!(decode(b"not a jpeg").is_err());
        let good = encode(&gradient(20, 20), JpegOptions::default());
        for n in [2, 10, 50, 200, good.len() - 3] {
            let _ = decode(&good[..n]); // must not panic
        }
        for i in 2..good.len() {
            let mut b = good.clone();
            b[i] ^= 0x55;
            let _ = decode(&b);
        }
    }
}
