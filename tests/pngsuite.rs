//! Decodes every image of PngSuite (Willem van Schaik's PNG test set) and compares the pixels
//! with Pillow's (`scripts/make_references.py`). Every deliberately broken file (`x*`) must be
//! refused.

use std::path::Path;

use lumen::image::{Color, Pixels};
use lumen::png;

#[test]
fn every_pngsuite_image_matches_pillow_and_every_broken_one_is_refused() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pngsuite");
    let refs: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap()).unwrap();
    let (mut checked, mut refused, mut with_trns) = (0, 0, 0);
    let mut failures = Vec::new();
    for (name, r) in refs.as_object().unwrap() {
        let data = std::fs::read(dir.join(name)).unwrap();
        let got = png::decode(&data);
        if name.starts_with('x') {
            if got.is_ok() {
                failures.push(format!("{name}: a broken file was accepted"));
            }
            refused += 1;
            continue;
        }
        let img = match got {
            Ok(i) => i,
            Err(e) => {
                failures.push(format!("{name}: {e}"));
                continue;
            }
        };
        assert_eq!((img.width as u64, img.height as u64), (r["w"].as_u64().unwrap(), r["h"].as_u64().unwrap()), "{name}");
        let crc = match r["kind"].as_str().unwrap() {
            "gray16" => {
                assert_eq!(img.color, Color::Gray, "{name}");
                let Pixels::U16(v) = &img.pixels else { panic!("{name}: expected 16 bits") };
                png::crc32(&v.iter().flat_map(|s| s.to_be_bytes()).collect::<Vec<u8>>())
            }
            _ => png::crc32(img.convert(Color::Rgba).data8()),
        };
        // Pillow ignores tRNS on 4-bit gray and makes every pixel transparent with tRNS on 16-bit
        // gray; for those two the colors (raw samples at 16 bits) come from Pillow and the alpha
        // from pypng.
        let pillow_trns_bug = ["tbbn0g04.png", "tbwn0g16.png"].contains(&name.as_str());
        if pillow_trns_bug {
            let same = match (r.get("gray16_crc"), &img.pixels) {
                // 16-bit: compare the raw gray samples (Pillow clamps them when converting).
                (Some(c), Pixels::U16(v)) => {
                    let gray: Vec<u8> = v.iter().step_by(img.color.channels()).flat_map(|s| s.to_be_bytes()).collect();
                    png::crc32(&gray) as u64 == c.as_u64().unwrap()
                }
                _ => png::crc32(img.convert(Color::Rgb).data8()) as u64 == r["rgb_crc"].as_u64().unwrap(),
            };
            if !same {
                failures.push(format!("{name}: colors differ from Pillow"));
            }
        } else if crc as u64 != r["crc"].as_u64().unwrap() {
            failures.push(format!(
                "{name}: pixels differ from Pillow ({:?}, {})",
                img.color,
                if img.is_16bit() { 16 } else { 8 }
            ));
        }
        if let Some(a) = r.get("trns_alpha_crc") {
            let rgba = img.convert(Color::Rgba);
            let alpha: Vec<u8> = rgba.data8().iter().skip(3).step_by(4).copied().collect();
            if png::crc32(&alpha) as u64 != a.as_u64().unwrap() {
                failures.push(format!("{name}: transparency differs from pypng"));
            }
            with_trns += 1;
        }
        checked += 1;
    }
    assert!(failures.is_empty(), "{} of {} differ:\n{}", failures.len(), checked + refused, failures.join("\n"));
    assert_eq!(refused, 14);
    assert!(checked >= 160, "{checked}");
    assert!(with_trns >= 10, "{with_trns}");
}

#[test]
fn every_pngsuite_image_survives_encode_and_decode() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pngsuite");
    for e in std::fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        if !name.ends_with(".png") || name.starts_with('x') {
            continue;
        }
        let img = png::decode(&std::fs::read(&p).unwrap()).unwrap();
        let again = png::decode(&png::encode(&img, 6)).unwrap();
        assert_eq!(again, img, "{name}");
    }
}
