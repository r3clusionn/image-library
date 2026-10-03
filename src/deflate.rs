//! DEFLATE compression: LZ77 with hash chains and lazy matching, then per block the smallest of
//! stored, fixed-Huffman or dynamic-Huffman coding (with length-limited codes).

use crate::inflate::{adler32, tables};

struct BitWriter {
    out: Vec<u8>,
    buf: u64,
    n: u32,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter { out: Vec::new(), buf: 0, n: 0 }
    }

    #[inline]
    fn put(&mut self, v: u32, k: u32) {
        self.buf |= (v as u64) << self.n;
        self.n += k;
        while self.n >= 8 {
            self.out.push(self.buf as u8);
            self.buf >>= 8;
            self.n -= 8;
        }
    }

    /// A Huffman code, which is sent most significant bit first.
    #[inline]
    fn code(&mut self, code: u32, len: u32) {
        self.put(code.reverse_bits() >> (32 - len), len);
    }

    fn align(&mut self) {
        if self.n > 0 {
            self.put(0, 8 - self.n % 8);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        self.align();
        self.out
    }
}

#[derive(Clone, Copy)]
enum Tok {
    Lit(u8),
    Match { len: u16, dist: u16 },
}

const WINDOW: usize = 32768;
const HASH_BITS: u32 = 15;

/// LZ77 parse. `level` 1 to 9 sets how hard to search, with zlib's parameters per level: a
/// match of at least `good` searches only a quarter of the chain for a better one, a match of at
/// least `lazy` is taken without checking the next position, and `nice` ends a search early.
fn lz77(data: &[u8], level: u32) -> Vec<Tok> {
    // (good, lazy, nice, chain); levels 1 to 3 do not match lazily.
    let (good, max_lazy, nice, max_chain) = match level {
        0 => (0, 0, 0, 0),
        1 => (4, 0, 8, 4),
        2 => (4, 0, 16, 8),
        3 => (4, 0, 32, 32),
        4 => (4, 4, 16, 16),
        5 => (8, 16, 32, 32),
        6 => (8, 16, 128, 128),
        7 => (8, 32, 128, 256),
        8 => (32, 128, 258, 1024),
        _ => (32, 258, 258, 4096),
    };
    let lazy = max_lazy > 0;
    let mut toks = Vec::with_capacity(data.len() / 2);
    if max_chain == 0 || data.len() < 3 {
        toks.extend(data.iter().map(|b| Tok::Lit(*b)));
        return toks;
    }
    let mut head = vec![u32::MAX; 1 << HASH_BITS];
    // Chain links in a ring twice the window (zlib keeps one window): small enough to stay in
    // cache, and no slot is reused while its position can still be matched.
    const RING: usize = 2 * WINDOW;
    let mut prev = vec![u32::MAX; RING];
    let hash = |i: usize| -> usize {
        let v = (data[i] as u32) | (data[i + 1] as u32) << 8 | (data[i + 2] as u32) << 16;
        (v.wrapping_mul(0x9e37_79b1) >> (32 - HASH_BITS)) as usize
    };
    let insert = |i: usize, head: &mut Vec<u32>, prev: &mut Vec<u32>| {
        if i + 2 < data.len() {
            let h = hash(i);
            prev[i % RING] = head[h];
            head[h] = i as u32;
        }
    };
    let find = |i: usize, prev_len: usize, head: &Vec<u32>, prev: &Vec<u32>| -> (usize, usize) {
        if i + 2 >= data.len() {
            return (0, 0);
        }
        let max_len = (data.len() - i).min(258);
        let mut best = (prev_len.min(max_len - 1), 0);
        let mut cand = head[hash(i)];
        let mut chain = if prev_len >= good { max_chain >> 2 } else { max_chain };
        while cand != u32::MAX && chain > 0 {
            let c = cand as usize;
            if i - c > WINDOW {
                break;
            }
            if data[c + best.0.min(max_len - 1)] == data[i + best.0.min(max_len - 1)] {
                let l = match_len(&data[c..], &data[i..], max_len);
                if l > best.0 {
                    best = (l, i - c);
                    if l >= nice {
                        break;
                    }
                }
            }
            let next = prev[c % RING];
            if next != u32::MAX && next as usize >= c {
                break;
            }
            cand = next;
            chain -= 1;
        }
        if best.0 >= 3 && best.1 > 0 {
            best
        } else {
            (0, 0)
        }
    };
    let mut i = 0;
    while i < data.len() {
        let (len, dist) = find(i, 0, &head, &prev);
        if len >= 3 {
            if lazy && len < max_lazy && i + 1 < data.len() {
                insert(i, &mut head, &mut prev);
                let (len2, dist2) = find(i + 1, len, &head, &prev);
                if len2 > len {
                    toks.push(Tok::Lit(data[i]));
                    i += 1;
                    toks.push(Tok::Match { len: len2 as u16, dist: dist2 as u16 });
                    for k in i + 1..i + len2 {
                        insert(k, &mut head, &mut prev);
                    }
                    i += len2;
                    continue;
                }
                toks.push(Tok::Match { len: len as u16, dist: dist as u16 });
                for k in i + 1..i + len {
                    insert(k, &mut head, &mut prev);
                }
                i += len;
                continue;
            }
            toks.push(Tok::Match { len: len as u16, dist: dist as u16 });
            for k in i..i + len {
                insert(k, &mut head, &mut prev);
            }
            i += len;
        } else {
            insert(i, &mut head, &mut prev);
            toks.push(Tok::Lit(data[i]));
            i += 1;
        }
    }
    toks
}

/// Length of the common prefix of `a` and `b`, at most `max`, eight bytes at a time.
#[inline]
fn match_len(a: &[u8], b: &[u8], max: usize) -> usize {
    let mut l = 0;
    while l + 8 <= max {
        let x = u64::from_le_bytes(a[l..l + 8].try_into().unwrap()) ^ u64::from_le_bytes(b[l..l + 8].try_into().unwrap());
        if x != 0 {
            return l + (x.trailing_zeros() / 8) as usize;
        }
        l += 8;
    }
    while l < max && a[l] == b[l] {
        l += 1;
    }
    l
}

/// Symbol index per match length (3 to 258) and per distance (1 to 32768), built once.
fn sym_tables() -> &'static (Vec<u8>, Vec<u8>) {
    static T: std::sync::OnceLock<(Vec<u8>, Vec<u8>)> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let (lbase, _, dbase, _, _) = tables();
        let l = (0..=258u16).map(|v| lbase.iter().rposition(|b| *b <= v).unwrap_or(0) as u8).collect();
        let d = (0..=32768u16).map(|v| dbase.iter().rposition(|b| *b <= v).unwrap_or(0) as u8).collect();
        (l, d)
    })
}

#[inline]
fn len_sym(len: u16) -> (usize, u32, u32) {
    let (base, extra, _, _, _) = tables();
    let i = sym_tables().0[len as usize] as usize;
    (257 + i, extra[i] as u32, (len - base[i]) as u32)
}

#[inline]
fn dist_sym(d: u16) -> (usize, u32, u32) {
    let (_, _, base, extra, _) = tables();
    let i = sym_tables().1[d as usize] as usize;
    (i, extra[i] as u32, (d - base[i]) as u32)
}

/// Code lengths for `freq`, none longer than `limit` (Huffman, then the usual repair that moves
/// overlong codes up while keeping Kraft's inequality).
pub fn code_lengths(freq: &[u32], limit: u8) -> Vec<u8> {
    let mut lens = vec![0u8; freq.len()];
    let mut syms: Vec<usize> = (0..freq.len()).filter(|i| freq[*i] > 0).collect();
    if syms.is_empty() {
        return lens;
    }
    if syms.len() == 1 {
        lens[syms[0]] = 1;
        return lens;
    }
    // Huffman with a sorted list of (weight, node); nodes beyond the leaves are internal.
    syms.sort_by_key(|s| (freq[*s], *s));
    let n = syms.len();
    let mut weight: Vec<u64> = syms.iter().map(|s| freq[*s] as u64).collect();
    let mut parent = vec![usize::MAX; 2 * n - 1];
    weight.resize(2 * n - 1, 0);
    let (mut leaf, mut internal, mut next) = (0usize, n, n);
    let pick = |weight: &Vec<u64>, leaf: &mut usize, internal: &mut usize, next: usize| -> usize {
        if *leaf < n && (*internal >= next || weight[*leaf] <= weight[*internal]) {
            *leaf += 1;
            *leaf - 1
        } else {
            *internal += 1;
            *internal - 1
        }
    };
    while next < 2 * n - 1 {
        let a = pick(&weight, &mut leaf, &mut internal, next);
        let b = pick(&weight, &mut leaf, &mut internal, next);
        weight[next] = weight[a] + weight[b];
        parent[a] = next;
        parent[b] = next;
        next += 1;
    }
    let mut depth = vec![0u32; 2 * n - 1];
    for i in (0..2 * n - 2).rev() {
        depth[i] = depth[parent[i]] + 1;
    }
    let mut count = vec![0u32; 64];
    for d in &depth[..n] {
        count[*d as usize] += 1;
    }
    // Limit: push codes deeper than `limit` up, then rebalance (as zlib does).
    let limit = limit as usize;
    let mut overflow = 0;
    for d in limit + 1..64 {
        overflow += count[d];
        count[limit] += count[d];
        count[d] = 0;
    }
    while overflow > 0 {
        let mut d = limit - 1;
        while count[d] == 0 {
            d -= 1;
        }
        count[d] -= 1;
        count[d + 1] += 2;
        count[limit] -= 1;
        overflow = overflow.saturating_sub(2);
    }
    // Kraft check and fix (the loop above can leave the code over-full by one at most).
    loop {
        let kraft: u64 = (1..=limit).map(|d| (count[d] as u64) << (limit - d)).sum();
        if kraft <= 1 << limit {
            break;
        }
        let mut d = limit - 1;
        while count[d] == 0 {
            d -= 1;
        }
        count[d] -= 1;
        count[d + 1] += 1;
    }
    // Longest codes to the rarest symbols.
    let mut d = limit;
    let mut it = syms.iter();
    while d > 0 {
        for _ in 0..count[d] {
            lens[*it.next().unwrap()] = d as u8;
        }
        d -= 1;
    }
    lens
}

fn canonical(lens: &[u8]) -> Vec<u32> {
    let mut count = [0u32; 16];
    for l in lens {
        count[*l as usize] += 1;
    }
    count[0] = 0;
    let mut next = [0u32; 16];
    let mut code = 0;
    for b in 1..16 {
        code = (code + count[b - 1]) << 1;
        next[b] = code;
    }
    lens.iter()
        .map(|l| {
            if *l == 0 {
                0
            } else {
                let c = next[*l as usize];
                next[*l as usize] += 1;
                c
            }
        })
        .collect()
}

fn write_block(w: &mut BitWriter, toks: &[Tok], raw: &[u8], last: bool) {
    let mut lf = vec![0u32; 286];
    let mut df = vec![0u32; 30];
    for t in toks {
        match *t {
            Tok::Lit(b) => lf[b as usize] += 1,
            Tok::Match { len, dist } => {
                lf[len_sym(len).0] += 1;
                df[dist_sym(dist).0] += 1;
            }
        }
    }
    lf[256] = 1;
    let ll = code_lengths(&lf, 15);
    let mut dl = code_lengths(&df, 15);
    if dl.iter().all(|l| *l == 0) {
        dl[0] = 1;
    }
    let hlit = 257.max(ll.iter().rposition(|l| *l > 0).unwrap() + 1);
    let hdist = 1.max(dl.iter().rposition(|l| *l > 0).map(|p| p + 1).unwrap_or(1));
    // Run-length code the lengths (16: repeat previous, 17/18: zeros).
    let all: Vec<u8> = ll[..hlit].iter().chain(dl[..hdist].iter()).copied().collect();
    let mut rle: Vec<(u8, u32)> = Vec::new();
    let mut i = 0;
    while i < all.len() {
        let v = all[i];
        let mut run = 1;
        while i + run < all.len() && all[i + run] == v {
            run += 1;
        }
        let mut left = run;
        if v == 0 {
            while left >= 11 {
                let k = left.min(138);
                rle.push((18, (k - 11) as u32));
                left -= k;
            }
            if left >= 3 {
                rle.push((17, (left - 3) as u32));
                left = 0;
            }
        } else {
            rle.push((v, 0));
            left -= 1;
            while left >= 3 {
                let k = left.min(6);
                rle.push((16, (k - 3) as u32));
                left -= k;
            }
        }
        for _ in 0..left {
            rle.push((v, 0));
        }
        i += run;
    }
    let mut cf = vec![0u32; 19];
    for (s, _) in &rle {
        cf[*s as usize] += 1;
    }
    let cl = code_lengths(&cf, 7);
    let (_, _, _, _, order) = tables();
    let hclen = 4.max(order.iter().rposition(|o| cl[*o] > 0).unwrap_or(0) + 1);

    // Sizes of the three ways to write the block, in bits.
    let body_bits = |ll: &[u8], dl: &[u8]| -> u64 {
        toks.iter()
            .map(|t| match *t {
                Tok::Lit(b) => ll[b as usize] as u64,
                Tok::Match { len, dist } => {
                    let (s, e, _) = len_sym(len);
                    let (d, de, _) = dist_sym(dist);
                    ll[s] as u64 + e as u64 + dl[d] as u64 + de as u64
                }
            })
            .sum::<u64>()
            + ll[256] as u64
    };
    let header = 14
        + 3 * hclen as u64
        + rle
            .iter()
            .map(|(s, _)| cl[*s as usize] as u64 + [2, 3, 7][(*s as usize).saturating_sub(16).min(2)] * (*s >= 16) as u64)
            .sum::<u64>();
    let dynamic = 3 + header + body_bits(&ll, &dl);
    let mut fl = vec![8u8; 288];
    fl[144..256].fill(9);
    fl[256..280].fill(7);
    let fd = vec![5u8; 30];
    let fixed_bits = 3 + body_bits(&fl, &fd);
    let stored = 3 + 7 + 32 + 8 * raw.len() as u64 + 40 * (raw.len() as u64 / 65535);

    if stored <= fixed_bits.min(dynamic) {
        for (k, chunk) in raw.chunks(65535).enumerate() {
            let fin = last && (k + 1) * 65535 >= raw.len();
            w.put(fin as u32, 1);
            w.put(0, 2);
            w.align();
            w.put(chunk.len() as u32, 16);
            w.put(!(chunk.len() as u32) & 0xffff, 16);
            for b in chunk {
                w.put(*b as u32, 8);
            }
        }
        if raw.is_empty() {
            w.put(last as u32, 1);
            w.put(0, 2);
            w.align();
            w.put(0, 16);
            w.put(0xffff, 16);
        }
        return;
    }
    let (lens_l, lens_d) = if fixed_bits <= dynamic { (fl, fd) } else { (ll.clone(), dl.clone()) };
    w.put(last as u32, 1);
    if fixed_bits <= dynamic {
        w.put(1, 2);
    } else {
        w.put(2, 2);
        w.put((hlit - 257) as u32, 5);
        w.put((hdist - 1) as u32, 5);
        w.put((hclen - 4) as u32, 4);
        for o in &order[..hclen] {
            w.put(cl[*o] as u32, 3);
        }
        let cc = canonical(&cl);
        for (s, extra) in &rle {
            w.code(cc[*s as usize], cl[*s as usize] as u32);
            match s {
                16 => w.put(*extra, 2),
                17 => w.put(*extra, 3),
                18 => w.put(*extra, 7),
                _ => {}
            }
        }
    }
    let lc = canonical(&lens_l);
    let dc = canonical(&lens_d);
    for t in toks {
        match *t {
            Tok::Lit(b) => w.code(lc[b as usize], lens_l[b as usize] as u32),
            Tok::Match { len, dist } => {
                let (s, e, ev) = len_sym(len);
                w.code(lc[s], lens_l[s] as u32);
                w.put(ev, e);
                let (d, de, dv) = dist_sym(dist);
                w.code(dc[d], lens_d[d] as u32);
                w.put(dv, de);
            }
        }
    }
    w.code(lc[256], lens_l[256] as u32);
}

/// Compresses to a raw DEFLATE stream. `level` 0 (store) to 9 (smallest).
pub fn deflate(data: &[u8], level: u32) -> Vec<u8> {
    let toks = lz77(data, level.min(9));
    let mut w = BitWriter::new();
    // Blocks of about 64K tokens, each with its own codes.
    const BLOCK: usize = 1 << 16;
    let mut pos = 0usize;
    let mut start = 0usize;
    if toks.is_empty() {
        write_block(&mut w, &[], &[], true);
        return w.finish();
    }
    while start < toks.len() {
        let end = (start + BLOCK).min(toks.len());
        let span: usize = toks[start..end].iter().map(|t| if let Tok::Match { len, .. } = t { *len as usize } else { 1 }).sum();
        write_block(&mut w, &toks[start..end], &data[pos..pos + span], end == toks.len());
        pos += span;
        start = end;
    }
    w.finish()
}

pub fn zlib_compress(data: &[u8], level: u32) -> Vec<u8> {
    let flevel = match level {
        0..=1 => 0,
        2..=5 => 1,
        6 => 2,
        _ => 3,
    };
    let cmf = 0x78u16;
    let mut flg = flevel << 6;
    flg += 31 - ((cmf << 8) | flg) % 31;
    let mut out = vec![cmf as u8, flg as u8];
    out.extend(deflate(data, level));
    out.extend(adler32(data).to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inflate::{inflate, zlib_decompress};

    fn sample(n: usize, seed: u64) -> Vec<u8> {
        // Text-like data with repeats, so matches of every length and distance occur.
        let words = ["the ", "quick ", "brown ", "fox ", "jumps ", "over ", "lazy ", "dog ", "\n", "0123456789"];
        let mut s = seed;
        let mut v = Vec::new();
        while v.len() < n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            if s.is_multiple_of(7) {
                v.push((s >> 40) as u8);
            } else {
                v.extend_from_slice(words[(s % 10) as usize].as_bytes());
            }
        }
        v.truncate(n);
        v
    }

    #[test]
    fn round_trips_at_every_level() {
        for level in 0..=9 {
            for n in [0, 1, 2, 3, 100, 70_000, 300_000] {
                let d = sample(n, level as u64 + 1);
                let z = zlib_compress(&d, level);
                assert_eq!(zlib_decompress(&z, usize::MAX).unwrap(), d, "level {level}, {n} bytes");
            }
        }
    }

    #[test]
    fn incompressible_data_is_stored_and_runs_compress_well() {
        let mut s = 0x1234_5678u64;
        let noise: Vec<u8> = (0..100_000)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s as u8
            })
            .collect();
        let z = deflate(&noise, 6);
        assert!(z.len() < noise.len() + noise.len() / 1000 + 64, "{}", z.len());
        assert_eq!(inflate(&z, usize::MAX).unwrap(), noise);
        let zeros = vec![0u8; 1_000_000];
        let z = deflate(&zeros, 6);
        assert!(z.len() < 2_000, "{}", z.len());
        assert_eq!(inflate(&z, usize::MAX).unwrap(), zeros);
    }

    #[test]
    fn code_lengths_respect_the_limit_and_kraft() {
        // Fibonacci frequencies make the deepest unrestricted Huffman trees.
        let mut f = vec![1u32, 1];
        while f.len() < 40 {
            let n = f[f.len() - 1] + f[f.len() - 2];
            f.push(n);
        }
        for limit in [7u8, 15] {
            let l = code_lengths(&f, limit);
            assert!(l.iter().all(|x| *x >= 1 && *x <= limit));
            let kraft: f64 = l.iter().map(|x| 0.5f64.powi(*x as i32)).sum();
            assert!(kraft <= 1.0 + 1e-12, "{kraft}");
        }
        assert_eq!(code_lengths(&[0, 5, 0], 15), vec![0, 1, 0]);
    }
}
