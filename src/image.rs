//! The image type every codec reads into and writes from.

use crate::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
    Gray,
    GrayAlpha,
    Rgb,
    Rgba,
}

impl Color {
    pub fn channels(self) -> usize {
        match self {
            Color::Gray => 1,
            Color::GrayAlpha => 2,
            Color::Rgb => 3,
            Color::Rgba => 4,
        }
    }

    pub fn has_alpha(self) -> bool {
        matches!(self, Color::GrayAlpha | Color::Rgba)
    }
}

/// Samples: 8 bits, or 16 bits (as integers, not bytes).
#[derive(Clone, Debug, PartialEq)]
pub enum Pixels {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub color: Color,
    pub pixels: Pixels,
}

impl Image {
    pub fn new8(width: u32, height: u32, color: Color, data: Vec<u8>) -> Result<Image, Error> {
        if data.len() != width as usize * height as usize * color.channels() {
            return Err(Error::Corrupt("pixel buffer size does not match the dimensions"));
        }
        Ok(Image { width, height, color, pixels: Pixels::U8(data) })
    }

    pub fn is_16bit(&self) -> bool {
        matches!(self.pixels, Pixels::U16(_))
    }

    /// 8-bit samples (16-bit ones keep their high byte).
    pub fn to_8bit(&self) -> Image {
        match &self.pixels {
            Pixels::U8(_) => self.clone(),
            Pixels::U16(v) => Image { pixels: Pixels::U8(v.iter().map(|s| (s >> 8) as u8).collect()), ..*self },
        }
    }

    pub fn data8(&self) -> &[u8] {
        match &self.pixels {
            Pixels::U8(v) => v,
            Pixels::U16(_) => panic!("16-bit image; call to_8bit first"),
        }
    }

    /// Converts to another channel layout (8-bit). Gray from colour uses BT.601 luma weights, as
    /// most tools do; adding alpha makes it opaque; removing alpha drops it.
    pub fn convert(&self, to: Color) -> Image {
        let src = self.to_8bit();
        let d = src.data8();
        let n = (self.width * self.height) as usize;
        let ch = self.color.channels();
        let mut out = Vec::with_capacity(n * to.channels());
        for i in 0..n {
            let p = &d[i * ch..i * ch + ch];
            let (r, g, b, a) = match self.color {
                Color::Gray => (p[0], p[0], p[0], 255),
                Color::GrayAlpha => (p[0], p[0], p[0], p[1]),
                Color::Rgb => (p[0], p[1], p[2], 255),
                Color::Rgba => (p[0], p[1], p[2], p[3]),
            };
            let y = || -> u8 {
                if self.color == Color::Gray || self.color == Color::GrayAlpha {
                    r
                } else {
                    ((r as u32 * 19595 + g as u32 * 38470 + b as u32 * 7471 + 32768) >> 16) as u8
                }
            };
            match to {
                Color::Gray => out.push(y()),
                Color::GrayAlpha => out.extend([y(), a]),
                Color::Rgb => out.extend([r, g, b]),
                Color::Rgba => out.extend([r, g, b, a]),
            }
        }
        Image { width: self.width, height: self.height, color: to, pixels: Pixels::U8(out) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions_between_layouts() {
        let im = Image::new8(2, 1, Color::Rgb, vec![255, 0, 0, 10, 20, 30]).unwrap();
        let g = im.convert(Color::Gray);
        assert_eq!(g.data8(), &[76, 18]);
        let a = im.convert(Color::Rgba);
        assert_eq!(a.data8(), &[255, 0, 0, 255, 10, 20, 30, 255]);
        assert_eq!(a.convert(Color::Rgb), im);
        let w = Image { width: 1, height: 1, color: Color::Gray, pixels: Pixels::U16(vec![0xabcd]) };
        assert_eq!(w.to_8bit().data8(), &[0xab]);
        assert!(Image::new8(2, 2, Color::Rgb, vec![0; 5]).is_err());
    }
}
