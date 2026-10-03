//! Resizes the fixtures with every filter and compares with Pillow's output
//! (`scripts/make_jpeg_references.py`).

use std::path::Path;

use lumen::png;
use lumen::resize::{resize, Filter};

#[test]
fn resizes_exactly_like_pillow() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/jpeg");
    let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("cases.json")).unwrap()).unwrap();
    let mut failures = Vec::new();
    for c in cases["resize"].as_array().unwrap() {
        let src = png::decode(&std::fs::read(dir.join(c["src"].as_str().unwrap())).unwrap()).unwrap();
        let (f, w, h) = (c["filter"].as_str().unwrap(), c["w"].as_u64().unwrap() as u32, c["h"].as_u64().unwrap() as u32);
        let got = resize(&src, w, h, Filter::parse(f).unwrap());
        let want = std::fs::read(dir.join(format!("resize_{f}_{w}x{h}.rgb"))).unwrap();
        let max = got.data8().iter().zip(&want).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
        let differ = got.data8().iter().zip(&want).filter(|(a, b)| a != b).count();
        if max > 0 {
            failures.push(format!("{f} {w}x{h}: {differ} samples differ, max {max}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
