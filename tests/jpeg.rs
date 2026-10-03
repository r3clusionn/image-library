//! Decodes JPEGs written by Pillow (libjpeg-turbo) and compares the pixels with what Pillow
//! decodes from the same files (`scripts/make_jpeg_references.py`). The two decoders use
//! different IDCTs, so samples may differ by a little; the test bounds the largest and the mean
//! difference.

use std::path::Path;

use lumen::image::{Color, Image};
use lumen::jpeg::{self, JpegOptions};
use lumen::{png, Error};

fn dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/jpeg")
}

fn diff(a: &[u8], b: &[u8]) -> (u8, f64) {
    assert_eq!(a.len(), b.len());
    let max = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0);
    let mean = a.iter().zip(b).map(|(x, y)| x.abs_diff(*y) as f64).sum::<f64>() / a.len() as f64;
    (max, mean)
}

#[test]
fn decodes_like_libjpeg_turbo() {
    let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir().join("cases.json")).unwrap()).unwrap();
    let mut report = Vec::new();
    for (name, c) in cases["jpeg"].as_object().unwrap() {
        let img = jpeg::decode(&std::fs::read(dir().join(format!("{name}.jpg"))).unwrap()).unwrap();
        assert_eq!((img.width as u64, img.height as u64), (c["w"].as_u64().unwrap(), c["h"].as_u64().unwrap()), "{name}");
        let gray = c["mode"] == "L";
        assert_eq!(img.color, if gray { Color::Gray } else { Color::Rgb }, "{name}");
        let want = std::fs::read(dir().join(format!("{name}.{}", if gray { "gray" } else { "rgb" }))).unwrap();
        let (max, mean) = diff(img.data8(), &want);
        report.push(format!("{name}: max {max}, mean {mean:.3}"));
        assert!(max <= 4 && mean < 0.6, "{name}: max {max}, mean {mean:.3}");
    }
    println!("{}", report.join("\n"));
}

#[test]
fn progressive_files_are_refused_as_unsupported() {
    let got = jpeg::decode(&std::fs::read(dir().join("progressive.jpg")).unwrap());
    assert!(matches!(got, Err(Error::Unsupported(_))), "{got:?}");
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>() / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// Pillow 12.3 (libjpeg-turbo, `optimize=True`) on source_big.png: (quality, 4:2:0?, bytes, PSNR dB).
const PILLOW: [(u8, bool, usize, f64); 8] = [
    (30, true, 2932, 28.02),
    (30, false, 4420, 29.69),
    (60, true, 4895, 29.40),
    (60, false, 7260, 31.16),
    (85, true, 10051, 30.31),
    (85, false, 15050, 32.75),
    (95, true, 21966, 30.96),
    (95, false, 40200, 36.13),
];

#[test]
fn encoder_matches_libjpeg_turbo_quality_and_size() {
    let src = png::decode(&std::fs::read(dir().join("source_big.png")).unwrap()).unwrap();
    for (q, subsample, size, db) in PILLOW {
        let bytes = jpeg::encode(&src, JpegOptions { quality: q, subsample, optimize: true });
        let back: Image = jpeg::decode(&bytes).unwrap();
        let p = psnr(back.data8(), src.data8());
        println!("q{q} 4:2:0={subsample}: {} bytes {p:.2} dB (Pillow {size} bytes {db:.2} dB)", bytes.len());
        assert!(p > db - 0.3, "q{q} {subsample}: {p:.2} dB vs {db:.2}");
        assert!((bytes.len() as f64) < size as f64 * 1.08, "q{q} {subsample}: {} bytes vs {size}", bytes.len());
    }
}
