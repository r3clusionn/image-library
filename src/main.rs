//! `lumen`: inspect, convert and resize PNG and JPEG files.

use std::process::ExitCode;
use std::time::Instant;

use lumen::image::{Color, Image};
use lumen::jpeg::{self, JpegOptions};
use lumen::resize::{resize, Filter};
use lumen::{png, Error};

const USAGE: &str = "usage:
  lumen info FILE...
  lumen convert IN OUT [--quality N] [--444] [--level N]
  lumen resize IN OUT WIDTHxHEIGHT [--filter nearest|bilinear|bicubic|lanczos3] [encode options]

The format of OUT comes from its extension (.png, .jpg, .jpeg). A WIDTH or HEIGHT of 0 keeps the
aspect ratio.";

fn read(path: &str) -> Result<(Image, &'static str), String> {
    let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let r = if data.starts_with(b"\x89PNG") {
        png::decode(&data).map(|i| (i, "PNG"))
    } else if data.starts_with(&[0xff, 0xd8]) {
        jpeg::decode(&data).map(|i| (i, "JPEG"))
    } else {
        Err(Error::Unsupported("not a PNG or JPEG file"))
    };
    r.map_err(|e| format!("{path}: {e}"))
}

struct EncodeOpts {
    quality: u8,
    subsample: bool,
    level: u32,
}

fn write(path: &str, img: &Image, o: &EncodeOpts) -> Result<(), String> {
    let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    let (bytes, color) = match ext.as_str() {
        "png" => (png::encode(img, o.level), img.color),
        "jpg" | "jpeg" => {
            let flat = if img.color.has_alpha() { flatten(img) } else { img.clone() };
            (jpeg::encode(&flat, JpegOptions { quality: o.quality, subsample: o.subsample, optimize: true }), flat.color)
        }
        _ => return Err(format!("{path}: unknown output format (use .png or .jpg)")),
    };
    std::fs::write(path, &bytes).map_err(|e| format!("{path}: {e}"))?;
    println!("wrote {path}: {}x{} {color:?}, {} bytes", img.width, img.height, bytes.len());
    Ok(())
}

/// JPEG has no alpha: composite over white.
fn flatten(img: &Image) -> Image {
    let rgba = img.convert(Color::Rgba);
    let d: Vec<u8> = rgba
        .data8()
        .chunks(4)
        .flat_map(|p| {
            let a = p[3] as u32;
            [0, 1, 2].map(|i| ((p[i] as u32 * a + 255 * (255 - a) + 127) / 255) as u8)
        })
        .collect();
    let rgb = Image::new8(img.width, img.height, Color::Rgb, d).unwrap();
    if matches!(img.color, Color::GrayAlpha) {
        rgb.convert(Color::Gray)
    } else {
        rgb
    }
}

/// "IHDR IDAT IDAT IDAT IEND" as "IHDR IDATx3 IEND".
fn collapse(chunks: &[String]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < chunks.len() {
        let n = chunks[i..].iter().take_while(|c| **c == chunks[i]).count();
        out.push(if n > 1 { format!("{}x{n}", chunks[i]) } else { chunks[i].clone() });
        i += n;
    }
    out.join(" ")
}

fn run(args: &[String]) -> Result<(), String> {
    let (cmd, rest) = args.split_first().ok_or(USAGE)?;
    let mut pos = Vec::new();
    let mut o = EncodeOpts { quality: 90, subsample: true, level: 6 };
    let mut filter = Filter::Lanczos3;
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        let mut val = |name: &str| it.next().cloned().ok_or(format!("{name} needs a value"));
        match a.as_str() {
            "--quality" => {
                o.quality = val("--quality")?.parse().ok().filter(|q| (1..=100).contains(q)).ok_or("quality is 1 to 100")?
            }
            "--level" => o.level = val("--level")?.parse().ok().filter(|l| *l <= 9).ok_or("level is 0 to 9")?,
            "--444" => o.subsample = false,
            "--filter" => filter = Filter::parse(&val("--filter")?).ok_or("unknown filter")?,
            s if s.starts_with("--") => return Err(format!("unknown option {s}\n{USAGE}")),
            _ => pos.push(a.clone()),
        }
    }
    match (cmd.as_str(), pos.as_slice()) {
        ("info", files) if !files.is_empty() => {
            for f in files {
                let t = Instant::now();
                let (img, kind) = read(f)?;
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let bits = if img.is_16bit() { 16 } else { 8 };
                println!("{f}: {kind} {}x{} {:?} {bits}-bit, decoded in {ms:.1} ms", img.width, img.height, img.color);
                if kind == "PNG" {
                    let i = png::info(&std::fs::read(f).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
                    println!("  color type {}, interlaced {}, chunks {}", i.color_type, i.interlaced, collapse(&i.chunks));
                }
            }
            Ok(())
        }
        ("convert", [input, output]) => write(output, &read(input)?.0, &o),
        ("resize", [input, output, size]) => {
            let img = read(input)?.0;
            let (w, h) = size.split_once('x').ok_or("size is WIDTHxHEIGHT")?;
            let (mut w, mut h): (u32, u32) = (w.parse().map_err(|_| "bad width")?, h.parse().map_err(|_| "bad height")?);
            if w == 0 && h == 0 {
                return Err("width and height cannot both be 0".into());
            }
            if w == 0 {
                w = ((img.width as u64 * h as u64 + img.height as u64 / 2) / img.height as u64).max(1) as u32;
            }
            if h == 0 {
                h = ((img.height as u64 * w as u64 + img.width as u64 / 2) / img.width as u64).max(1) as u32;
            }
            write(output, &resize(&img, w, h, filter), &o)
        }
        _ => Err(USAGE.into()),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lumen: {e}");
            ExitCode::FAILURE
        }
    }
}
