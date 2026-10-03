//! Times lumen on the files `scripts/bench.py` writes: prints one JSON object of median
//! milliseconds per operation. Run through the script, which also times Pillow.

use std::time::Instant;

use lumen::jpeg::{self, JpegOptions};
use lumen::png;
use lumen::resize::{resize, Filter};

fn median_ms(runs: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut t: Vec<f64> = (0..runs)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(|a, b| a.total_cmp(b));
    t[runs / 2]
}

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("usage: bench DIR"));
    let png_bytes = std::fs::read(dir.join("photo.png")).unwrap();
    let jpg_bytes = std::fs::read(dir.join("photo.jpg")).unwrap();
    let img = png::decode(&png_bytes).unwrap();
    let mut r = Vec::new();
    r.push(("png_decode", median_ms(15, || drop(png::decode(&png_bytes).unwrap()))));
    r.push(("png_encode", median_ms(5, || drop(png::encode(&img, 6)))));
    r.push(("jpeg_decode", median_ms(15, || drop(jpeg::decode(&jpg_bytes).unwrap()))));
    let opt = JpegOptions { quality: 90, subsample: true, optimize: true };
    r.push(("jpeg_encode", median_ms(15, || drop(jpeg::encode(&img, opt)))));
    r.push(("resize_lanczos3", median_ms(15, || drop(resize(&img, 960, 540, Filter::Lanczos3)))));
    let body: Vec<String> = r.iter().map(|(k, v)| format!("\"{k}\": {v:.3}")).collect();
    println!("{{{}}}", body.join(", "));
    // Sizes, for the README.
    eprintln!("png {} bytes, jpeg {} bytes", png::encode(&img, 6).len(), jpeg::encode(&img, opt).len());
}
