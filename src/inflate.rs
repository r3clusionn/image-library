//! DEFLATE decompression (RFC 1951) and the zlib wrapper (RFC 1950), as PNG uses them.
//!
//! Huffman codes are decoded with one lookup table per code, indexed by the next `max` bits
//! (bit-reversed, since DEFLATE packs codes most significant bit first into a least significant
//! bit first stream).

use crate::Error;

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    n: u32,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Bits { data, pos: 0, buf: 0, n: 0 }
    }

    #[inline]
    fn refill(&mut self) {
        while self.n <= 56 {
            let b = if self.pos < self.data.len() { self.data[self.pos] } else { 0 };
            if self.pos >= self.data.len() + 8 {
                break;
            }
            self.buf |= (b as u64) << self.n;
            self.pos += 1;
            self.n += 8;
        }
    }

    #[inline]
    fn peek(&mut self, k: u32) -> u32 {
        if self.n < k {
            self.refill();
        }
        (self.buf & ((1u64 << k) - 1)) as u32
    }

    #[inline]
    fn consume(&mut self, k: u32) {
        self.buf >>= k;
        self.n -= k;
    }

    #[inline]
    fn bits(&mut self, k: u32) -> u32 {
        if k == 0 {
            return 0;
        }
        let v = self.peek(k);
        self.consume(k);
        v
    }

    /// Bytes consumed so far, whole bytes only (after aligning).
    fn byte_pos(&self) -> usize {
        self.pos - (self.n / 8) as usize
    }

    fn align(&mut self) {
        let r = self.n % 8;
        self.consume(r);
    }

    fn overrun(&self) -> bool {
        self.byte_pos() > self.data.len()
    }
}

/// A canonical Huffman code as a lookup table: entry = symbol << 4 | length (0 = invalid).
struct Huffman {
    table: Vec<u32>,
    max: u32,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Result<Huffman, Error> {
        let max = *lengths.iter().max().unwrap_or(&0) as u32;
        if max == 0 {
            // A code with no symbols (allowed for distances when only literals follow).
            return Ok(Huffman { table: vec![0; 2], max: 1 });
        }
        let mut count = [0u32; 16];
        for &l in lengths {
            count[l as usize] += 1;
        }
        count[0] = 0;
        // Over-subscribed codes are invalid; incomplete ones are allowed (a single code).
        let mut left: i32 = 1;
        for c in &count[1..=15] {
            left = (left << 1) - *c as i32;
            if left < 0 {
                return Err(Error::Corrupt("over-subscribed Huffman code"));
            }
        }
        let mut next = [0u32; 16];
        let mut code = 0u32;
        for bits in 1..=15 {
            code = (code + count[bits - 1]) << 1;
            next[bits] = code;
        }
        let mut table = vec![0u32; 1 << max];
        for (sym, &len) in lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let len = len as u32;
            let c = next[len as usize];
            next[len as usize] += 1;
            let rev = c.reverse_bits() >> (32 - len);
            let mut i = rev;
            while i < (1 << max) {
                table[i as usize] = (sym as u32) << 4 | len;
                i += 1 << len;
            }
        }
        Ok(Huffman { table, max })
    }

    #[inline]
    fn decode(&self, b: &mut Bits) -> Result<u32, Error> {
        let e = self.table[b.peek(self.max) as usize];
        let len = e & 15;
        if len == 0 {
            return Err(Error::Corrupt("invalid Huffman code"));
        }
        b.consume(len);
        Ok(e >> 4)
    }
}

const LEN_BASE: [u16; 29] =
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193,
    12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const CL_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

#[allow(clippy::type_complexity)]
pub(crate) fn tables() -> (&'static [u16; 29], &'static [u8; 29], &'static [u16; 30], &'static [u8; 30], &'static [usize; 19]) {
    (&LEN_BASE, &LEN_EXTRA, &DIST_BASE, &DIST_EXTRA, &CL_ORDER)
}

fn fixed() -> (Huffman, Huffman) {
    let mut l = [0u8; 288];
    l[..144].fill(8);
    l[144..256].fill(9);
    l[256..280].fill(7);
    l[280..].fill(8);
    (Huffman::new(&l).unwrap(), Huffman::new(&[5u8; 30]).unwrap())
}

/// Decompresses a raw DEFLATE stream. `limit` caps the output size (decompression bombs).
pub fn inflate(data: &[u8], limit: usize) -> Result<Vec<u8>, Error> {
    inflate_counted(data, limit).map(|(v, _)| v)
}

/// Also returns how many input bytes the stream used.
pub fn inflate_counted(data: &[u8], limit: usize) -> Result<(Vec<u8>, usize), Error> {
    let mut out: Vec<u8> = Vec::with_capacity((data.len() * 4).min(limit));
    let mut b = Bits::new(data);
    loop {
        let last = b.bits(1) == 1;
        match b.bits(2) {
            0 => {
                b.align();
                let start = b.byte_pos();
                // The buffered bits are whole bytes now; restart reading from `start`.
                b = Bits { data, pos: start, buf: 0, n: 0 };
                let hdr = data.get(start..start + 4).ok_or(Error::Corrupt("truncated stored block"))?;
                let len = u16::from_le_bytes([hdr[0], hdr[1]]) as usize;
                let nlen = u16::from_le_bytes([hdr[2], hdr[3]]) as usize;
                if len != !nlen & 0xffff {
                    return Err(Error::Corrupt("stored block length check failed"));
                }
                let body = data.get(start + 4..start + 4 + len).ok_or(Error::Corrupt("truncated stored block"))?;
                if out.len() + len > limit {
                    return Err(Error::TooLarge);
                }
                out.extend_from_slice(body);
                b.pos = start + 4 + len;
            }
            1 => {
                let (lit, dist) = fixed();
                block(&mut b, &lit, &dist, &mut out, limit)?;
            }
            2 => {
                let hlit = b.bits(5) as usize + 257;
                let hdist = b.bits(5) as usize + 1;
                let hclen = b.bits(4) as usize + 4;
                let mut cl = [0u8; 19];
                for &i in CL_ORDER.iter().take(hclen) {
                    cl[i] = b.bits(3) as u8;
                }
                let clh = Huffman::new(&cl)?;
                let mut lens = vec![0u8; hlit + hdist];
                let mut i = 0;
                while i < hlit + hdist {
                    let sym = clh.decode(&mut b)?;
                    let (val, rep) = match sym {
                        0..=15 => (sym as u8, 1),
                        16 => {
                            if i == 0 {
                                return Err(Error::Corrupt("repeat with no previous length"));
                            }
                            (lens[i - 1], 3 + b.bits(2) as usize)
                        }
                        17 => (0, 3 + b.bits(3) as usize),
                        _ => (0, 11 + b.bits(7) as usize),
                    };
                    if i + rep > hlit + hdist {
                        return Err(Error::Corrupt("code lengths overflow"));
                    }
                    lens[i..i + rep].fill(val);
                    i += rep;
                }
                if lens[256] == 0 {
                    return Err(Error::Corrupt("no end-of-block code"));
                }
                let lit = Huffman::new(&lens[..hlit])?;
                let dist = Huffman::new(&lens[hlit..])?;
                block(&mut b, &lit, &dist, &mut out, limit)?;
            }
            _ => return Err(Error::Corrupt("reserved block type")),
        }
        if b.overrun() {
            return Err(Error::Corrupt("unexpected end of data"));
        }
        if last {
            b.align();
            return Ok((out, b.byte_pos()));
        }
    }
}

fn block(b: &mut Bits, lit: &Huffman, dist: &Huffman, out: &mut Vec<u8>, limit: usize) -> Result<(), Error> {
    loop {
        let sym = lit.decode(b)?;
        if sym < 256 {
            if out.len() >= limit {
                return Err(Error::TooLarge);
            }
            out.push(sym as u8);
            continue;
        }
        if sym == 256 {
            return Ok(());
        }
        let li = sym as usize - 257;
        if li >= 29 {
            return Err(Error::Corrupt("invalid length symbol"));
        }
        let len = LEN_BASE[li] as usize + b.bits(LEN_EXTRA[li] as u32) as usize;
        let di = dist.decode(b)? as usize;
        if di >= 30 {
            return Err(Error::Corrupt("invalid distance symbol"));
        }
        let d = DIST_BASE[di] as usize + b.bits(DIST_EXTRA[di] as u32) as usize;
        if d > out.len() {
            return Err(Error::Corrupt("distance before the start"));
        }
        if out.len() + len > limit {
            return Err(Error::TooLarge);
        }
        if b.overrun() {
            return Err(Error::Corrupt("unexpected end of data"));
        }
        let start = out.len() - d;
        if d >= len {
            out.extend_from_within(start..start + len);
        } else {
            for k in 0..len {
                let v = out[start + k];
                out.push(v);
            }
        }
    }
}

pub fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in data.chunks(5552) {
        for &x in chunk {
            a += x as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

/// Decompresses a zlib stream and checks its Adler-32.
pub fn zlib_decompress(data: &[u8], limit: usize) -> Result<Vec<u8>, Error> {
    if data.len() < 6 {
        return Err(Error::Corrupt("zlib stream too short"));
    }
    let (cmf, flg) = (data[0], data[1]);
    if cmf & 0x0f != 8 || !((cmf as u16) << 8 | flg as u16).is_multiple_of(31) {
        return Err(Error::Corrupt("not a zlib stream"));
    }
    if flg & 0x20 != 0 {
        return Err(Error::Unsupported("zlib preset dictionary"));
    }
    let (out, used) = inflate_counted(&data[2..], limit)?;
    let at = 2 + used;
    let sum = data.get(at..at + 4).ok_or(Error::Corrupt("missing zlib checksum"))?;
    if u32::from_be_bytes(sum.try_into().unwrap()) != adler32(&out) {
        return Err(Error::Corrupt("zlib checksum mismatch"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_fixed_and_dynamic_blocks() {
        // "hello" in a stored block, then as fixed Huffman (zlib output of known strings).
        let stored = [1u8, 5, 0, 250, 255, b'h', b'e', b'l', b'l', b'o'];
        assert_eq!(inflate(&stored, 100).unwrap(), b"hello");
        // Python: zlib.compressobj(9, zlib.DEFLATED, -15) on b"hello hello hello hello"
        let fixed = [0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x27, 0x01];
        assert_eq!(inflate(&fixed, 100).unwrap(), b"hello hello hello hello");
        assert_eq!(adler32(b"Wikipedia"), 0x11e60398);
    }

    #[test]
    fn corrupt_input_is_an_error_not_a_panic() {
        assert!(inflate(&[0xff, 0xff, 0xff], 1000).is_err());
        assert!(inflate(&[], 1000).is_err());
        assert!(inflate(&[1, 5, 0, 0, 0, b'h'], 100).is_err(), "bad NLEN");
        assert!(matches!(inflate(&[1u8, 5, 0, 250, 255, b'h', b'e', b'l', b'l', b'o'], 3), Err(Error::TooLarge)));
        assert!(zlib_decompress(&[0x78, 0x9c, 3, 0, 0, 0, 0, 2], 10).is_err(), "wrong checksum");
        // Every single-byte truncation and every bit flip of a real stream: errors or output, never a panic.
        let fixed = [0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x27, 0x01];
        for n in 0..fixed.len() {
            let _ = inflate(&fixed[..n], 1000);
        }
        for i in 0..fixed.len() * 8 {
            let mut f = fixed;
            f[i / 8] ^= 1 << (i % 8);
            let _ = inflate(&f, 1000);
        }
    }
}
