//! PNG decoding and encoding.
//!
//! Decoding handles every color type and bit depth the specification allows (1, 2, 4, 8 and 16
//! bits; gray, RGB, palette, gray with alpha, RGBA), Adam7 interlacing, all five row filters,
//! `tRNS` transparency (which adds an alpha channel), and checks every chunk's CRC. Sub-byte
//! gray is scaled to 8 bits (a 2-bit 3 becomes 255); palettes are expanded to RGB or RGBA.
//! Ancillary chunks other than `tRNS` are skipped (gamma and color profiles are not applied).

use crate::deflate::zlib_compress;
use crate::image::{Color, Image, Pixels};
use crate::inflate::zlib_decompress;
use crate::Error;

const SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

pub fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let t = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *e = c;
        }
        t
    });
    let mut c = !0u32;
    for b in data {
        c = t[((c ^ *b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

/// Limits for untrusted files.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_pixels: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { max_pixels: 1 << 28 }
    }
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i16 + b as i16 - c as i16;
    let (pa, pb, pc) = ((p - a as i16).abs(), (p - b as i16).abs(), (p - c as i16).abs());
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Reverses the filter of one row in place. `bpp` is bytes per pixel (at least 1).
fn unfilter(kind: u8, row: &mut [u8], prev: &[u8], bpp: usize) -> Result<(), Error> {
    match kind {
        0 => {}
        1 => {
            for i in bpp..row.len() {
                row[i] = row[i].wrapping_add(row[i - bpp]);
            }
        }
        2 => {
            for i in 0..row.len() {
                row[i] = row[i].wrapping_add(prev[i]);
            }
        }
        3 => {
            for i in 0..row.len() {
                let left = if i >= bpp { row[i - bpp] } else { 0 };
                row[i] = row[i].wrapping_add(((left as u16 + prev[i] as u16) / 2) as u8);
            }
        }
        4 => {
            for i in 0..row.len() {
                let (a, c) = if i >= bpp { (row[i - bpp], prev[i - bpp]) } else { (0, 0) };
                row[i] = row[i].wrapping_add(paeth(a, prev[i], c));
            }
        }
        _ => return Err(Error::Corrupt("unknown row filter")),
    }
    Ok(())
}

struct Header {
    width: u32,
    height: u32,
    depth: u8,
    ctype: u8,
    interlace: bool,
}

impl Header {
    fn samples(&self) -> usize {
        match self.ctype {
            0 | 3 => 1,
            4 => 2,
            2 => 3,
            _ => 4,
        }
    }

    fn row_bytes(&self, w: u32) -> usize {
        (w as usize * self.samples() * self.depth as usize).div_ceil(8)
    }
}

/// Everything about a PNG file without decoding its pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct Info {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub color_type: u8,
    pub interlaced: bool,
    pub chunks: Vec<String>,
}

pub fn info(data: &[u8]) -> Result<Info, Error> {
    let (h, _, _, _, chunks) = parse(data, false)?;
    Ok(Info { width: h.width, height: h.height, bit_depth: h.depth, color_type: h.ctype, interlaced: h.interlace, chunks })
}

type Parsed = (Header, Vec<u8>, Option<Vec<u8>>, Option<Vec<u8>>, Vec<String>);

fn parse(data: &[u8], need_idat: bool) -> Result<Parsed, Error> {
    if data.get(..8) != Some(&SIGNATURE) {
        return Err(Error::Corrupt("not a PNG file"));
    }
    let mut at = 8;
    let mut header: Option<Header> = None;
    let (mut idat, mut plte, mut trns) = (Vec::new(), None, None);
    let mut names = Vec::new();
    let mut seen_idat = false;
    loop {
        let len = u32::from_be_bytes(data.get(at..at + 4).ok_or(Error::Corrupt("truncated chunk"))?.try_into().unwrap()) as usize;
        if len > 0x7fff_ffff {
            return Err(Error::Corrupt("chunk length too large"));
        }
        let kind: [u8; 4] = data.get(at + 4..at + 8).ok_or(Error::Corrupt("truncated chunk"))?.try_into().unwrap();
        let body = data.get(at + 8..at + 8 + len).ok_or(Error::Corrupt("truncated chunk"))?;
        let crc = u32::from_be_bytes(
            data.get(at + 8 + len..at + 12 + len).ok_or(Error::Corrupt("truncated chunk"))?.try_into().unwrap(),
        );
        if crc32(&data[at + 4..at + 8 + len]) != crc {
            return Err(Error::Corrupt("chunk CRC mismatch"));
        }
        if !kind.iter().all(|c| c.is_ascii_alphabetic()) {
            return Err(Error::Corrupt("invalid chunk name"));
        }
        names.push(String::from_utf8_lossy(&kind).into_owned());
        at += 12 + len;
        match &kind {
            b"IHDR" => {
                if header.is_some() || body.len() != 13 {
                    return Err(Error::Corrupt("bad IHDR"));
                }
                let h = Header {
                    width: u32::from_be_bytes(body[0..4].try_into().unwrap()),
                    height: u32::from_be_bytes(body[4..8].try_into().unwrap()),
                    depth: body[8],
                    ctype: body[9],
                    interlace: body[12] == 1,
                };
                let ok = match h.ctype {
                    0 => [1, 2, 4, 8, 16].contains(&h.depth),
                    3 => [1, 2, 4, 8].contains(&h.depth),
                    2 | 4 | 6 => [8, 16].contains(&h.depth),
                    _ => false,
                };
                if !ok
                    || body[10] != 0
                    || body[11] != 0
                    || body[12] > 1
                    || h.width == 0
                    || h.height == 0
                    || h.width > 1 << 31
                    || h.height > 1 << 31
                {
                    return Err(Error::Corrupt("invalid IHDR values"));
                }
                header = Some(h);
            }
            _ if header.is_none() => return Err(Error::Corrupt("first chunk is not IHDR")),
            b"PLTE" => {
                if body.len() % 3 != 0 || body.is_empty() || body.len() > 768 {
                    return Err(Error::Corrupt("bad palette"));
                }
                plte = Some(body.to_vec());
            }
            b"tRNS" => trns = Some(body.to_vec()),
            b"IDAT" => {
                seen_idat = true;
                idat.extend_from_slice(body);
            }
            b"IEND" => break,
            _ => {
                // An unknown critical chunk (uppercase first letter) cannot be skipped.
                if kind[0].is_ascii_uppercase() {
                    return Err(Error::Unsupported("unknown critical chunk"));
                }
            }
        }
    }
    let h = header.ok_or(Error::Corrupt("no IHDR"))?;
    if need_idat && !seen_idat {
        return Err(Error::Corrupt("no image data"));
    }
    if h.ctype == 3 && plte.is_none() {
        return Err(Error::Corrupt("palette image without PLTE"));
    }
    Ok((h, idat, plte, trns, names))
}

const ADAM7: [(usize, usize, usize, usize); 7] =
    [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)];

pub fn decode(data: &[u8]) -> Result<Image, Error> {
    decode_with(data, Limits::default())
}

pub fn decode_with(data: &[u8], limits: Limits) -> Result<Image, Error> {
    let (h, idat, plte, trns, _) = parse(data, true)?;
    if h.width as u64 * h.height as u64 > limits.max_pixels {
        return Err(Error::TooLarge);
    }
    let (w, ht) = (h.width as usize, h.height as usize);
    let bpp = (h.samples() * h.depth as usize).div_ceil(8).max(1);
    // Raw size: every pass's rows, each with its filter byte.
    let passes: Vec<(usize, usize, usize, usize)> = if h.interlace { ADAM7.to_vec() } else { vec![(0, 0, 1, 1)] };
    let pass_dims = |(x0, y0, dx, dy): (usize, usize, usize, usize)| ((w + dx - 1 - x0) / dx, (ht + dy - 1 - y0) / dy);
    let expected: usize = passes
        .iter()
        .map(|p| {
            let (pw, ph) = pass_dims(*p);
            if pw == 0 || ph == 0 {
                0
            } else {
                ph * (1 + h.row_bytes(pw as u32))
            }
        })
        .sum();
    let raw = zlib_decompress(&idat, expected + 1)?;
    if raw.len() < expected {
        return Err(Error::Corrupt("not enough image data"));
    }

    // Samples of the whole image, one per channel, before palette and transparency.
    let samples = h.samples();
    let mut full: Vec<u16> = vec![0; w * ht * samples];
    let mut at = 0;
    for p in &passes {
        let (pw, ph) = pass_dims(*p);
        if pw == 0 || ph == 0 {
            continue;
        }
        let rb = h.row_bytes(pw as u32);
        let mut prev = vec![0u8; rb];
        for y in 0..ph {
            let kind = raw[at];
            let mut row = raw[at + 1..at + 1 + rb].to_vec();
            at += 1 + rb;
            unfilter(kind, &mut row, &prev, bpp)?;
            let (x0, y0, dx, dy) = *p;
            let yy = y0 + y * dy;
            for x in 0..pw {
                let xx = x0 + x * dx;
                for s in 0..samples {
                    let i = x * samples + s;
                    let v = match h.depth {
                        16 => u16::from_be_bytes([row[i * 2], row[i * 2 + 1]]),
                        8 => row[i] as u16,
                        d => {
                            let bit = i * d as usize;
                            ((row[bit / 8] >> (8 - d as usize - bit % 8)) & ((1 << d) - 1)) as u16
                        }
                    };
                    full[(yy * w + xx) * samples + s] = v;
                }
            }
            prev = row;
        }
    }

    // Palette, transparency, scaling of sub-byte gray.
    let n = w * ht;
    if h.ctype == 3 {
        let pal = plte.unwrap();
        let entries = pal.len() / 3;
        let alpha = trns.as_ref().filter(|t| !t.is_empty());
        let mut out = Vec::with_capacity(n * if alpha.is_some() { 4 } else { 3 });
        for v in &full {
            let i = *v as usize;
            if i >= entries {
                return Err(Error::Corrupt("palette index out of range"));
            }
            out.extend_from_slice(&pal[i * 3..i * 3 + 3]);
            if let Some(a) = alpha {
                out.push(*a.get(i).unwrap_or(&255));
            }
        }
        let color = if alpha.is_some() { Color::Rgba } else { Color::Rgb };
        return Ok(Image { width: h.width, height: h.height, color, pixels: Pixels::U8(out) });
    }
    if h.depth < 8 {
        let scale = 255 / ((1u16 << h.depth) - 1);
        for v in &mut full {
            *v *= scale;
        }
    }
    let base = match h.ctype {
        0 => Color::Gray,
        2 => Color::Rgb,
        4 => Color::GrayAlpha,
        _ => Color::Rgba,
    };
    // tRNS on gray or RGB names one color that is fully transparent: add an alpha channel.
    let key: Option<Vec<u16>> = trns.filter(|_| h.ctype == 0 || h.ctype == 2).and_then(|t| {
        let need = if h.ctype == 0 { 2 } else { 6 };
        (t.len() >= need).then(|| {
            t[..need]
                .chunks(2)
                .map(|c| {
                    let v = u16::from_be_bytes([c[0], c[1]]);
                    if h.depth < 8 {
                        (v & ((1 << h.depth) - 1)) * (255 / ((1 << h.depth) - 1))
                    } else {
                        v
                    }
                })
                .collect()
        })
    });
    let (color, full) = match key {
        Some(k) => {
            let mut out = Vec::with_capacity(n * (samples + 1));
            let opaque = if h.depth == 16 { 0xffff } else { 0xff };
            for px in full.chunks(samples) {
                out.extend_from_slice(px);
                out.push(if px == k.as_slice() { 0 } else { opaque });
            }
            (if base == Color::Gray { Color::GrayAlpha } else { Color::Rgba }, out)
        }
        None => (base, full),
    };
    let pixels = if h.depth == 16 { Pixels::U16(full) } else { Pixels::U8(full.into_iter().map(|v| v as u8).collect()) };
    Ok(Image { width: h.width, height: h.height, color, pixels })
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    out.extend((body.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    let c = crc32(&out[start..]);
    out.extend(c.to_be_bytes());
}

/// Encodes an image as PNG (8 or 16 bits, its own color layout). Each row gets the filter
/// whose output has the smallest sum of absolute values, a cheap predictor of what compresses.
pub fn encode(img: &Image, level: u32) -> Vec<u8> {
    let ch = img.color.channels();
    let (depth, bytes): (u8, Vec<u8>) = match &img.pixels {
        Pixels::U8(v) => (8, v.clone()),
        Pixels::U16(v) => (16, v.iter().flat_map(|s| s.to_be_bytes()).collect()),
    };
    let bpp = ch * depth as usize / 8;
    let rb = img.width as usize * bpp;
    let mut raw = Vec::with_capacity((rb + 1) * img.height as usize);
    let zero = vec![0u8; rb];
    let mut cand = vec![0u8; rb];
    let mut best = vec![0u8; rb];
    for y in 0..img.height as usize {
        let row = &bytes[y * rb..(y + 1) * rb];
        let prev = if y == 0 { &zero[..] } else { &bytes[(y - 1) * rb..y * rb] };
        let mut best_kind = 0u8;
        let mut best_score = u64::MAX;
        for kind in 0..5u8 {
            for i in 0..rb {
                let a = if i >= bpp { row[i - bpp] } else { 0 };
                let c = if i >= bpp { prev[i - bpp] } else { 0 };
                let pred = match kind {
                    0 => 0,
                    1 => a,
                    2 => prev[i],
                    3 => ((a as u16 + prev[i] as u16) / 2) as u8,
                    _ => paeth(a, prev[i], c),
                };
                cand[i] = row[i].wrapping_sub(pred);
            }
            let score: u64 = cand.iter().map(|b| (*b as i8).unsigned_abs() as u64).sum();
            if score < best_score {
                best_score = score;
                best_kind = kind;
                best.copy_from_slice(&cand);
            }
        }
        raw.push(best_kind);
        raw.extend_from_slice(&best);
    }
    let ctype = match img.color {
        Color::Gray => 0,
        Color::GrayAlpha => 4,
        Color::Rgb => 2,
        Color::Rgba => 6,
    };
    let mut out = SIGNATURE.to_vec();
    let mut ihdr = img.width.to_be_bytes().to_vec();
    ihdr.extend(img.height.to_be_bytes());
    ihdr.extend([depth, ctype, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &zlib_compress(&raw, level));
    chunk(&mut out, b"IEND", &[]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_of_known_strings() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        assert_eq!(crc32(b"IEND"), 0xae42_6082);
    }

    #[test]
    fn encode_then_decode_every_layout() {
        for color in [Color::Gray, Color::GrayAlpha, Color::Rgb, Color::Rgba] {
            let (w, h) = (37u32, 23u32);
            let n = (w * h) as usize * color.channels();
            let data: Vec<u8> = (0..n).map(|i| ((i * 7) ^ (i / 13)) as u8).collect();
            let img = Image::new8(w, h, color, data).unwrap();
            assert_eq!(decode(&encode(&img, 6)).unwrap(), img);
            let wide = Image { pixels: Pixels::U16((0..n).map(|i| (i * 2_741) as u16).collect()), ..img.clone() };
            assert_eq!(decode(&encode(&wide, 6)).unwrap(), wide);
        }
    }

    #[test]
    fn broken_files_are_errors() {
        assert!(decode(b"not a png").is_err());
        let img = Image::new8(4, 4, Color::Rgb, vec![9; 48]).unwrap();
        let good = encode(&img, 6);
        for i in 8..good.len() {
            let mut b = good.clone();
            b[i] ^= 0x10;
            assert!(decode(&b).is_err(), "flipped byte {i} went unnoticed");
        }
        for n in 0..good.len() {
            assert!(decode(&good[..n]).is_err());
        }
        assert!(matches!(decode_with(&good, Limits { max_pixels: 15 }), Err(Error::TooLarge)));
    }
}
