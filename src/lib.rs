//! `lumen`: image codecs and processing written from scratch: DEFLATE and zlib, PNG decoding
//! and encoding, baseline JPEG decoding and encoding, resizing and colour conversion.

pub mod deflate;
pub mod image;
pub mod inflate;
pub mod jpeg;
pub mod png;
pub mod resize;

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The data breaks the format.
    Corrupt(&'static str),
    /// Valid, but a feature this library does not implement.
    Unsupported(&'static str),
    /// Larger than the configured limit.
    TooLarge,
    Io(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Corrupt(m) => write!(f, "corrupt data: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::TooLarge => write!(f, "image larger than the limit"),
            Error::Io(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}
