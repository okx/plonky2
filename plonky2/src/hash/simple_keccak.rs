// Based on the C code from here: https://raw.githubusercontent.com/coruus/saarinen-keccak/master/readable_keccak/keccak.c
// https://github.com/coruus/saarinen-keccak/tree/master/readable_keccak
// 19-Nov-11  Markku-Juhani O. Saarinen <mjos@iki.fi>
// A baseline Keccak (3rd round) implementation.

use crunchy::unroll;

const KECCAK_ROUNDS : i32 = 24;

const KECCAKF_RNDC : [u64; 24] = 
[
    1u64,
    0x8082u64,
    0x800000000000808au64,
    0x8000000080008000u64,
    0x808bu64,
    0x80000001u64,
    0x8000000080008081u64,
    0x8000000000008009u64,
    0x8au64,
    0x88u64,
    0x80008009u64,
    0x8000000au64,
    0x8000808bu64,
    0x800000000000008bu64,
    0x8000000000008089u64,
    0x8000000000008003u64,
    0x8000000000008002u64,
    0x8000000000000080u64,
    0x800au64,
    0x800000008000000au64,
    0x8000000080008081u64,
    0x8000000000008080u64,
    0x80000001u64,
    0x8000000080008008u64,
];

const KECCAKF_ROTC : [u32; 24] = 
[
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14, 
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
];

const KECCAKF_PILN : [usize; 24] = 
[
    10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4, 
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1 
];

pub fn keccakf(st: &mut [u64; 25], rounds: i32) {
    let mut bc: [u64; 5] = [0; 5];
    for round in 0..rounds {
        // Theta
        for i in 0..5 {
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        }
        for i in 0..5 {
            let t = bc[(i + 4) % 5] ^ bc[(i + 1) % 5].rotate_left(1);
            for j in (0..25).step_by(5) {
                st[j + i] ^= t;
            }
        }
        // Rho Pi
        let mut t = st[1];
        for i in 0..24 {
            let j = KECCAKF_PILN[i];
            let bc = st[j];
            st[j] = t.rotate_left(KECCAKF_ROTC[i]);
            t = bc;
        }
        // Chi
        for j in (0..25).step_by(5) {
            let mut bc: [u64; 5] = [0; 5];
            for i in 0..5 {
                bc[i] = st[j + i];
            }
            for i in 0..5 {
                st[j + i] ^= (!bc[(i + 1) % 5]) & bc[(i + 2) % 5];
            }
        }
        // Iota
        st[0] ^= KECCAKF_RNDC[round as usize];
    }
}

pub fn keccakf_tiny(a: &mut [u64; 25], rounds: i32) {

    for i in 0..rounds {
        let mut array: [u64; 5] = [0; 5];

        // Theta
        unroll! {
            for x in 0..5 {
                unroll! {
                    for y_count in 0..5 {
                        let y = y_count * 5;
                        array[x] ^= a[x + y];
                    }
                }
            }
        }

        unroll! {
            for x in 0..5 {
                unroll! {
                    for y_count in 0..5 {
                        let y = y_count * 5;
                        a[y + x] ^= array[(x + 4) % 5] ^ array[(x + 1) % 5].rotate_left(1);
                    }
                }
            }
        }

        // Rho and pi
        let mut last = a[1];
        unroll! {
            for x in 0..24 {
                array[0] = a[KECCAKF_PILN[x]];
                a[KECCAKF_PILN[x]] = last.rotate_left(KECCAKF_ROTC[x]);
                last = array[0];
            }
        }

        // Chi
        unroll! {
            for y_step in 0..5 {
                let y = y_step * 5;

                unroll! {
                    for x in 0..5 {
                        array[x] = a[y + x];
                    }
                }

                unroll! {
                    for x in 0..5 {
                        a[y + x] = array[x] ^ ((!array[(x + 1) % 5]) & (array[(x + 2) % 5]));
                    }
                }
            }
        };

        // Iota
        a[0] ^= KECCAKF_RNDC[i as usize];
    }
}

pub fn keccak_flex(inp: &[u8], inlen: usize, md: &mut [u8], mdlen: usize) {
    let mut st: [u64; 25] = [0; 25];
    let mut temp: [u8; 144] = [0; 144];
    let rsiz: usize;
    let rsizw: usize;
    rsiz = 200 - 2 * mdlen;
    rsizw = rsiz / 8;
    st.fill(0);
    for chunk in inp.chunks_exact(rsiz) {
        for i in 0..rsizw {
            st[i] ^= u64::from_ne_bytes(chunk[i * 8..(i + 1) * 8].try_into().unwrap());
        }
        keccakf(&mut st, KECCAK_ROUNDS);
    }
    // last block and padding
    let llen = inlen % rsiz;
    let loff = inlen - llen;
    temp[..llen].copy_from_slice(&inp[loff..]);
    temp[llen] = 1;
    temp[llen + 1..rsiz].fill(0);
    temp[rsiz - 1] |= 0x80;
    for i in 0..rsizw {
        st[i] ^= u64::from_ne_bytes(temp[i * 8..(i + 1) * 8].try_into().unwrap());
    }
    keccakf(&mut st, KECCAK_ROUNDS);
    unsafe { 
        let stb : [u8; 200] = std::mem::transmute(st);
        md.copy_from_slice(&stb[..mdlen]);
    }    
}

pub struct H256(pub [u8; 32]);

impl H256 {
    pub const fn to_fixed_bytes(self) -> [u8; 32] {
        self.0
    }
}

pub fn keccaks(inp: Vec<u8>) -> H256 {
    let ainp : &[u8] = &inp;
    let mut md : [u8; 32] = [0; 32];
    keccak_flex(ainp, inp.len(), &mut md, 32); 
    H256(md)    
}

#[cfg(test)]
mod tests {

    use anyhow::Result;
    use anyhow::ensure;
    use keccak_hash::keccak;

    use super::{keccak_flex, keccaks};

    const res1 : [u8; 28] = [
        0x30, 0x04, 0x5B, 0x34, 0x94, 0x6E, 0x1B, 0x2E, 
        0x09, 0x16, 0x13, 0x36, 0x2F, 0xD2, 0x2A, 0xA0, 
        0x8E, 0x2B, 0xEA, 0xFE, 0xC5, 0xE8, 0xDA, 0xEE, 
        0x42, 0xC2, 0xE6, 0x65
    ];

    const res2 : [u8; 32] = [
        0xA8, 0xD7, 0x1B, 0x07, 0xF4, 0xAF, 0x26, 0xA4, 
        0xFF, 0x21, 0x02, 0x7F, 0x62, 0xFF, 0x60, 0x26, 
        0x7F, 0xF9, 0x55, 0xC9, 0x63, 0xF0, 0x42, 0xC4, 
        0x6D, 0xA5, 0x2E, 0xE3, 0xCF, 0xAF, 0x3D, 0x3C
    ];

    const res3 : [u8; 48] = [
        0xE2, 0x13, 0xFD, 0x74, 0xAF, 0x0C, 0x5F, 0xF9, 
        0x1B, 0x42, 0x3C, 0x8B, 0xCE, 0xEC, 0xD7, 0x01, 
        0xF8, 0xDD, 0x64, 0xEC, 0x18, 0xFD, 0x6F, 0x92, 
        0x60, 0xFC, 0x9E, 0xC1, 0xED, 0xBD, 0x22, 0x30, 
        0xA6, 0x90, 0x86, 0x65, 0xBC, 0xD9, 0xFB, 0xF4, 
        0x1A, 0x99, 0xA1, 0x8A, 0x7D, 0x9E, 0x44, 0x6E 
    ];

    const res4 : [u8; 64] = [
        0x96, 0xEE, 0x47, 0x18, 0xDC, 0xBA, 0x3C, 0x74, 
        0x61, 0x9B, 0xA1, 0xFA, 0x7F, 0x57, 0xDF, 0xE7, 
        0x76, 0x9D, 0x3F, 0x66, 0x98, 0xA8, 0xB3, 0x3F, 
        0xA1, 0x01, 0x83, 0x89, 0x70, 0xA1, 0x31, 0xE6, 
        0x21, 0xCC, 0xFD, 0x05, 0xFE, 0xFF, 0xBC, 0x11, 
        0x80, 0xF2, 0x63, 0xC2, 0x7F, 0x1A, 0xDA, 0xB4, 
        0x60, 0x95, 0xD6, 0xF1, 0x25, 0x33, 0x14, 0x72, 
        0x4B, 0x5C, 0xBF, 0x78, 0x28, 0x65, 0x8E, 0x6A
    ];

    const str1 : &str = "Keccak-224 Test Hash";
    const str2 : &str = "Keccak-256 Test Hash";
    const str3 : &str = "Keccak-384 Test Hash";
    const str4 : &str = "Keccak-512 Test Hash";

    fn check_res(rs : &[u8], rf : &[u8], size : usize) -> Result<()> {
        println!("{:?}", rs);
        println!("{:?}", rf);

        for i in 0..size {
            ensure!(rs[i] == rf[i], "Keccak Hash is different");
        }
        
        Ok(())
    }

    #[test]
    fn test_simple_keccak_flex() -> Result<()> {
        let x = str1.as_bytes();
        let mut md : [u8; 28] = [0; 28];
        keccak_flex(x, x.len(), &mut md, 28);
        let r = check_res(&md, &res1, 28);
        if r.is_err() {
            return r
        }

        let x = str2.as_bytes();
        let mut md : [u8; 32] = [0; 32];
        keccak_flex(x, x.len(), &mut md, 32);
        let r = check_res(&md, &res2, 32);
        if r.is_err() {
            return r
        }

        let x = str3.as_bytes();
        let mut md : [u8; 48] = [0; 48];
        keccak_flex(x, x.len(), &mut md, 48);
        let r = check_res(&md, &res3, 48);
        if r.is_err() {
            return r
        }

        let x = str4.as_bytes();
        let mut md : [u8; 64] = [0; 64];
        keccak_flex(x, x.len(), &mut md, 64);
        let r = check_res(&md, &res4, 64);
        if r.is_err() {
            return r
        }
    
        Ok(())
    }

    #[test]
    fn test_tiny_keccak() -> Result<()> {
        let x = str2.as_bytes().to_vec();
        let md = keccak(x).0;
        let r = check_res(&md, &res2, 32);
        if r.is_err() {
            return r
        }
        Ok(())
    }

    fn random_data(n: usize, k: usize) -> Vec<Vec<u8>> {
        (0..n).map(|_| (0..k).map(|_| rand::random::<u8>()).collect()).collect()
    }

    #[test]
    fn test_tiny_keccak_2() -> Result<()> {
        let data = random_data(1000, 400);
        for elem in data {
            let celem = elem.clone();
            let md1 = keccak(elem).0;
            let md2 = keccaks(celem).0;
            let r = check_res(&md1, &md2, 32);
            if r.is_err() {
                return r
            }
        }
        Ok(())
    }

    #[test]
    fn test_simple_keccak() -> Result<()> {
        let x = str2.as_bytes().to_vec();
        let h = keccak(x);
        let md = h.0;
        let r = check_res(&md, &res2, 32);
        if r.is_err() {
            return r
        }

        Ok(())
    }

}