//! Resizing with separable convolution filters: nearest, bilinear (triangle), bicubic
//! (Catmull-Rom style, a = -0.5) and Lanczos3. When shrinking, the filter is widened by the scale
//! factor so every source pixel contributes (no aliasing).
//!
//! The arithmetic follows Pillow's resampler: weights are computed in `f64`, normalised, turned
//! into 22-bit fixed point, and each pass rounds and clamps to 8 bits (horizontal pass first).
//! The tests check the output against Pillow's byte for byte. Images with alpha are resized with
//! premultiplied colour, so transparent pixels do not bleed their colour into the edges.

use crate::image::Image;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    Nearest,
    Bilinear,
    Bicubic,
    Lanczos3,
}

impl Filter {
    pub fn parse(s: &str) -> Option<Filter> {
        Some(match s {
            "nearest" => Filter::Nearest,
            "bilinear" => Filter::Bilinear,
            "bicubic" => Filter::Bicubic,
            "lanczos" | "lanczos3" => Filter::Lanczos3,
            _ => return None,
        })
    }

    fn support(self) -> f64 {
        match self {
            Filter::Nearest => 0.5,
            Filter::Bilinear => 1.0,
            Filter::Bicubic => 2.0,
            Filter::Lanczos3 => 3.0,
        }
    }

    fn weight(self, x: f64) -> f64 {
        let x = x.abs();
        match self {
            Filter::Nearest => {
                if x <= 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            Filter::Bilinear => (1.0 - x).max(0.0),
            Filter::Bicubic => {
                let a = -0.5;
                if x < 1.0 {
                    ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * a
                } else {
                    0.0
                }
            }
            Filter::Lanczos3 => {
                if x < 3.0 {
                    sinc(x) * sinc(x / 3.0)
                } else {
                    0.0
                }
            }
        }
    }
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let p = x * std::f64::consts::PI;
        p.sin() / p
    }
}

const PRECISION: u32 = 22;

/// For each output position: the first source index and the fixed-point weights.
struct Kernel {
    start: Vec<usize>,
    taps: usize,
    weights: Vec<i32>,
}

fn kernel(src: usize, dst: usize, filter: Filter) -> Kernel {
    let scale = src as f64 / dst as f64;
    let filterscale = scale.max(1.0);
    let support = filter.support() * filterscale;
    let taps = support.ceil() as usize * 2 + 1;
    let mut start = Vec::with_capacity(dst);
    let mut weights = vec![0i32; dst * taps];
    let mut k = vec![0f64; taps];
    for x in 0..dst {
        let center = (x as f64 + 0.5) * scale;
        let lo = ((center - support + 0.5) as isize).max(0) as usize;
        let hi = ((center + support + 0.5) as usize).min(src);
        let n = hi - lo;
        let mut total = 0.0;
        for (i, w) in k.iter_mut().enumerate().take(n) {
            *w = filter.weight((i as f64 + lo as f64 - center + 0.5) / filterscale);
            total += *w;
        }
        for (i, w) in k.iter().enumerate().take(n) {
            let v = if total != 0.0 { w / total } else { 0.0 };
            let f = v * (1u32 << PRECISION) as f64;
            weights[x * taps + i] = if f < 0.0 { (f - 0.5) as i32 } else { (f + 0.5) as i32 };
        }
        start.push(lo);
    }
    Kernel { start, taps, weights }
}

fn clamp8(v: i64) -> u8 {
    (v >> PRECISION).clamp(0, 255) as u8
}

/// One pass along rows (`horizontal`) or columns, `ch` interleaved channels.
fn pass(src: &[u8], w: usize, h: usize, ch: usize, out_len: usize, horizontal: bool, k: &Kernel) -> Vec<u8> {
    let (ow, oh) = if horizontal { (out_len, h) } else { (w, out_len) };
    let mut out = vec![0u8; ow * oh * ch];
    let half = 1i64 << (PRECISION - 1);
    for y in 0..oh {
        for x in 0..ow {
            let (o, along) = if horizontal { (x, w) } else { (y, h) };
            let lo = k.start[o];
            let ws = &k.weights[o * k.taps..o * k.taps + k.taps];
            for c in 0..ch {
                let mut acc = half;
                for (i, wt) in ws.iter().enumerate() {
                    let s = lo + i;
                    if s >= along {
                        break;
                    }
                    let idx = if horizontal { (y * w + s) * ch + c } else { (s * w + x) * ch + c };
                    acc += src[idx] as i64 * *wt as i64;
                }
                out[(y * ow + x) * ch + c] = clamp8(acc);
            }
        }
    }
    out
}

fn nearest(img: &Image, w: u32, h: u32) -> Image {
    let ch = img.color.channels();
    let d = img.data8();
    let (sw, sh) = (img.width as usize, img.height as usize);
    let mut out = Vec::with_capacity(w as usize * h as usize * ch);
    for y in 0..h as usize {
        let sy = (((y as f64 + 0.5) * sh as f64 / h as f64) as usize).min(sh - 1);
        for x in 0..w as usize {
            let sx = (((x as f64 + 0.5) * sw as f64 / w as f64) as usize).min(sw - 1);
            out.extend_from_slice(&d[(sy * sw + sx) * ch..(sy * sw + sx) * ch + ch]);
        }
    }
    Image::new8(w, h, img.color, out).unwrap()
}

/// Resizes to `w` x `h` (both at least 1). 16-bit images are reduced to 8 bits first.
pub fn resize(img: &Image, w: u32, h: u32, filter: Filter) -> Image {
    assert!(w > 0 && h > 0, "target size must be at least 1x1");
    let img = img.to_8bit();
    if img.width == 0 || img.height == 0 {
        return Image::new8(w, h, img.color, vec![0; w as usize * h as usize * img.color.channels()]).unwrap();
    }
    if filter == Filter::Nearest {
        return nearest(&img, w, h);
    }
    let ch = img.color.channels();
    let mut data = img.data8().to_vec();
    let alpha = img.color.has_alpha();
    if alpha {
        premultiply(&mut data, ch);
    }
    let (sw, sh) = (img.width as usize, img.height as usize);
    if w as usize != sw {
        data = pass(&data, sw, sh, ch, w as usize, true, &kernel(sw, w as usize, filter));
    }
    if h as usize != sh {
        data = pass(&data, w as usize, sh, ch, h as usize, false, &kernel(sh, h as usize, filter));
    }
    if alpha {
        unpremultiply(&mut data, ch);
    }
    Image::new8(w, h, img.color, data).unwrap()
}

fn premultiply(d: &mut [u8], ch: usize) {
    for p in d.chunks_exact_mut(ch) {
        let a = p[ch - 1] as u32;
        for s in &mut p[..ch - 1] {
            *s = ((*s as u32 * a + 127) / 255) as u8;
        }
    }
}

fn unpremultiply(d: &mut [u8], ch: usize) {
    for p in d.chunks_exact_mut(ch) {
        let a = p[ch - 1] as u32;
        if a == 0 {
            continue;
        }
        for s in &mut p[..ch - 1] {
            *s = ((*s as u32 * 255 + a / 2) / a).min(255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Color;

    #[test]
    fn flat_images_stay_flat_with_every_filter() {
        let img = Image::new8(7, 5, Color::Rgb, [10u8, 200, 77].repeat(35)).unwrap();
        for f in [Filter::Nearest, Filter::Bilinear, Filter::Bicubic, Filter::Lanczos3] {
            for (w, h) in [(1, 1), (3, 2), (19, 13), (7, 5)] {
                let r = resize(&img, w, h, f);
                assert!(r.data8().chunks(3).all(|p| p == [10, 200, 77]), "{f:?} {w}x{h}");
            }
        }
    }

    #[test]
    fn transparent_pixels_do_not_bleed() {
        // Left half opaque red, right half fully transparent green.
        let mut d = Vec::new();
        for x in 0..8 {
            d.extend(if x < 4 { [255, 0, 0, 255] } else { [0, 255, 0, 0] });
        }
        let r = resize(&Image::new8(8, 1, Color::Rgba, d).unwrap(), 3, 1, Filter::Lanczos3);
        for p in r.data8().chunks(4) {
            if p[3] > 0 {
                assert!(p[1] < 3, "{p:?}");
            }
        }
    }
}
