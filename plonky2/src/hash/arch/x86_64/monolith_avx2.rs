use core::arch::x86_64::*;

use super::goldilocks_avx2::{reduce_avx_96_64, sqr_avx};
use super::poseidon_bn128_avx2::{add64, add64_no_carry};
use crate::hash::monolith::{LOOKUP_BITS, SPONGE_WIDTH};
use crate::hash::monolith_goldilocks::{MONOLITH_MAT_12, MONOLITH_ROUND_CONSTANTS};

#[inline]
unsafe fn bar_avx(el: &mut __m256i) {
    if LOOKUP_BITS == 8 {
        let ct1 = _mm256_set_epi64x(
            0x8080808080808080u64 as i64,
            0x8080808080808080u64 as i64,
            0x8080808080808080u64 as i64,
            0x8080808080808080u64 as i64,
        );
        let ct2 = _mm256_set_epi64x(
            0x7F7F7F7F7F7F7F7Fu64 as i64,
            0x7F7F7F7F7F7F7F7Fu64 as i64,
            0x7F7F7F7F7F7F7F7Fu64 as i64,
            0x7F7F7F7F7F7F7F7Fu64 as i64,
        );
        let ct3 = _mm256_set_epi64x(
            0xC0C0C0C0C0C0C0C0u64 as i64,
            0xC0C0C0C0C0C0C0C0u64 as i64,
            0xC0C0C0C0C0C0C0C0u64 as i64,
            0xC0C0C0C0C0C0C0C0u64 as i64,
        );
        let ct4 = _mm256_set_epi64x(
            0x3F3F3F3F3F3F3F3Fu64 as i64,
            0x3F3F3F3F3F3F3F3Fu64 as i64,
            0x3F3F3F3F3F3F3F3Fu64 as i64,
            0x3F3F3F3F3F3F3F3Fu64 as i64,
        );
        let ct5 = _mm256_set_epi64x(
            0xE0E0E0E0E0E0E0E0u64 as i64,
            0xE0E0E0E0E0E0E0E0u64 as i64,
            0xE0E0E0E0E0E0E0E0u64 as i64,
            0xE0E0E0E0E0E0E0E0u64 as i64,
        );
        let ct6 = _mm256_set_epi64x(
            0x1F1F1F1F1F1F1F1Fu64 as i64,
            0x1F1F1F1F1F1F1F1Fu64 as i64,
            0x1F1F1F1F1F1F1F1Fu64 as i64,
            0x1F1F1F1F1F1F1F1Fu64 as i64,
        );
        let l1 = _mm256_andnot_si256(*el, ct1);
        let l2 = _mm256_srli_epi64(l1, 7);
        let l3 = _mm256_andnot_si256(*el, ct2);
        let l4 = _mm256_slli_epi64(l3, 1);
        let limb1 = _mm256_or_si256(l2, l4);
        let l1 = _mm256_and_si256(*el, ct3);
        let l2 = _mm256_srli_epi64(l1, 6);
        let l3 = _mm256_and_si256(*el, ct4);
        let l4 = _mm256_slli_epi64(l3, 2);
        let limb2 = _mm256_or_si256(l2, l4);
        let l1 = _mm256_and_si256(*el, ct5);
        let l2 = _mm256_srli_epi64(l1, 5);
        let l3 = _mm256_and_si256(*el, ct6);
        let l4 = _mm256_slli_epi64(l3, 3);
        let limb3 = _mm256_or_si256(l2, l4);
        let tmp = _mm256_and_si256(limb1, limb2);
        let tmp = _mm256_and_si256(tmp, limb3);
        let tmp = _mm256_xor_si256(*el, tmp);
        let l1 = _mm256_and_si256(tmp, ct1);
        let l2 = _mm256_srli_epi64(l1, 7);
        let l3 = _mm256_and_si256(tmp, ct2);
        let l4 = _mm256_slli_epi64(l3, 1);
        *el = _mm256_or_si256(l2, l4);
    } else if LOOKUP_BITS == 16 {
        let ct1 = _mm256_set_epi64x(
            0x8000800080008000u64 as i64,
            0x8000800080008000u64 as i64,
            0x8000800080008000u64 as i64,
            0x8000800080008000u64 as i64,
        );
        let ct2 = _mm256_set_epi64x(
            0x7FFF7FFF7FFF7FFFu64 as i64,
            0x7FFF7FFF7FFF7FFFu64 as i64,
            0x7FFF7FFF7FFF7FFFu64 as i64,
            0x7FFF7FFF7FFF7FFFu64 as i64,
        );
        let ct3 = _mm256_set_epi64x(
            0xC000C000C000C000u64 as i64,
            0xC000C000C000C000u64 as i64,
            0xC000C000C000C000u64 as i64,
            0xC000C000C000C000u64 as i64,
        );
        let ct4 = _mm256_set_epi64x(
            0x3FFF3FFF3FFF3FFFu64 as i64,
            0x3FFF3FFF3FFF3FFFu64 as i64,
            0x3FFF3FFF3FFF3FFFu64 as i64,
            0x3FFF3FFF3FFF3FFFu64 as i64,
        );
        let ct5 = _mm256_set_epi64x(
            0xE000E000E000E000u64 as i64,
            0xE000E000E000E000u64 as i64,
            0xE000E000E000E000u64 as i64,
            0xE000E000E000E000u64 as i64,
        );
        let ct6 = _mm256_set_epi64x(
            0x1FFF1FFF1FFF1FFFu64 as i64,
            0x1FFF1FFF1FFF1FFFu64 as i64,
            0x1FFF1FFF1FFF1FFFu64 as i64,
            0x1FFF1FFF1FFF1FFFu64 as i64,
        );
        let l1 = _mm256_andnot_si256(*el, ct1);
        let l2 = _mm256_srli_epi64(l1, 15);
        let l3 = _mm256_andnot_si256(*el, ct2);
        let l4 = _mm256_slli_epi64(l3, 1);
        let limb1 = _mm256_or_si256(l2, l4);
        let l1 = _mm256_and_si256(*el, ct3);
        let l2 = _mm256_srli_epi64(l1, 14);
        let l3 = _mm256_and_si256(*el, ct4);
        let l4 = _mm256_slli_epi64(l3, 2);
        let limb2 = _mm256_or_si256(l2, l4);
        let l1 = _mm256_and_si256(*el, ct5);
        let l2 = _mm256_srli_epi64(l1, 13);
        let l3 = _mm256_and_si256(*el, ct6);
        let l4 = _mm256_slli_epi64(l3, 3);
        let limb3 = _mm256_or_si256(l2, l4);
        let tmp = _mm256_and_si256(limb1, limb2);
        let tmp = _mm256_and_si256(tmp, limb3);
        let tmp = _mm256_xor_si256(*el, tmp);
        let l1 = _mm256_and_si256(tmp, ct1);
        let l2 = _mm256_srli_epi64(l1, 15);
        let l3 = _mm256_and_si256(tmp, ct2);
        let l4 = _mm256_slli_epi64(l3, 1);
        *el = _mm256_or_si256(l2, l4);
    }
}

// The result is put in (sumhi, sumlo). Uses two functions defined in poseidon_bn128_avx2.
#[inline]
unsafe fn add_u128(sumhi: &mut __m256i, sumlo: &mut __m256i, hi: &__m256i, lo: &__m256i) {
    let (r, c) = add64_no_carry(sumlo, lo);
    (*sumhi, _) = add64(sumhi, hi, &c);
    *sumlo = r;
}

/*
// Alternative to add_u128() using non-AVX ops.
unsafe fn add_u128(sumhi: &mut __m256i, sumlo: &mut __m256i, hi: &__m256i, lo: &__m256i) {
    let mut vsh = [0u64; 4];
    let mut vh = [0u64; 4];
    let mut vsl = [0u64; 4];
    let mut vl = [0u64; 4];
    _mm256_storeu_si256(vsh.as_mut_ptr().cast::<__m256i>(), *sumhi);
    _mm256_storeu_si256(vsl.as_mut_ptr().cast::<__m256i>(), *sumlo);
    _mm256_storeu_si256(vh.as_mut_ptr().cast::<__m256i>(), *hi);
    _mm256_storeu_si256(vl.as_mut_ptr().cast::<__m256i>(), *lo);

    for i in 0..4 {
        let r = vsl[i] as u128 + vl[i] as u128 + ((vsh[i] as u128) << 64) + ((vh[i] as u128) << 64);
        vl[i] = r as u64;
        vh[i] = (r >> 64) as u64;
    }
    *sumhi = _mm256_loadu_si256(vh.as_ptr().cast::<__m256i>());
    *sumlo = _mm256_loadu_si256(vl.as_ptr().cast::<__m256i>());
}
*/

// Multiply one u128 (ah, al) by one u64 (b) -> return on u128 (h, l).
// b is small (< 2^32)
// ah is only 0 or 1
// al = alh * 2^32 + all => al * b = alh * b * 2^32 + all * b
// result l = l1 + l2 where l1 = all * b (which is always < 2^64)
// l2 is the low part (<2^64) of alh * b * 2^32 which is (alh * b) << 32
// l1 + l2 may overflow, so we need to keep the carry out
// h = h1 + h2 + h3 where h1 = b if ah is 1 or 0 otherwise
// h2 is the high part of (>2^64) alh * b * 2^32 which is (alh * b) >> 32 which is < 2^32
// h3 is the carry (0 or 1) -- given h1, h2, h3 are < 2^32, h cannot overflow
#[inline]
unsafe fn mul_u128_x_u64(ah: &__m256i, al: &__m256i, b: &__m256i) -> (__m256i, __m256i) {
    let ones = _mm256_set_epi64x(1, 1, 1, 1);
    let m = _mm256_cmpeq_epi64(*ah, ones);
    let h = _mm256_and_si256(m, *b);
    let al_h = _mm256_srli_epi64(*al, 32);
    let r_h = _mm256_mul_epu32(al_h, *b);
    let r_l = _mm256_mul_epu32(*al, *b);
    let r_h_l = _mm256_slli_epi64(r_h, 32);
    let (l, carry) = add64_no_carry(&r_l, &r_h_l);
    let r_h_h = _mm256_srli_epi64(r_h, 32);
    let h = _mm256_add_epi64(h, r_h_h);
    let h = _mm256_add_epi64(h, carry);
    (h, l)
}

/*
// Alternative to using non-AVX ops.
unsafe fn mul_u128_x_u64(ah: &__m256i, al: &__m256i, b: &__m256i) -> (__m256i, __m256i) {
    let mut val = [0u64; 4];
    let mut vah = [0u64; 4];
    let mut vb = [0u64; 4];
    let mut vh = [0u64; 4];
    let mut vl = [0u64; 4];
    _mm256_storeu_si256(val.as_mut_ptr().cast::<__m256i>(), *al);
    _mm256_storeu_si256(vah.as_mut_ptr().cast::<__m256i>(), *ah);
    _mm256_storeu_si256(vb.as_mut_ptr().cast::<__m256i>(), *b);
    for i in 0..4 {
        let r = (val[i] as u128 + ((vah[i] as u128) << 64)) * (vb[i] as u128);
        vl[i] = r as u64;
        vh[i] = (r >> 64) as u64;
    }
    let h = _mm256_loadu_si256(vh.as_ptr().cast::<__m256i>());
    let l = _mm256_loadu_si256(vl.as_ptr().cast::<__m256i>());
    (h, l)
}
*/

/*
// Alternative to reduce_avx_128_64() using non-AVX ops.
unsafe fn reduce(h: &__m256i, l: &__m256i, s: &mut __m256i) {
    let mut vh = [0u64; 4];
    let mut vl = [0u64; 4];
    let mut v = [0u64; 4];
    _mm256_storeu_si256(vh.as_mut_ptr().cast::<__m256i>(), *h);
    _mm256_storeu_si256(vl.as_mut_ptr().cast::<__m256i>(), *l);

    for i in 0..4 {
        v[i] = GoldilocksField::from_noncanonical_u96((vl[i], vh[i] as u32)).to_noncanonical_u64();
    }

    *s = _mm256_loadu_si256(v.as_ptr().cast::<__m256i>());
}
*/

// (h0, h1, h2) is the high part (>2^64) of the input and can only be 0 or 1 (not higher).
// (l0, l1, l2) is the low part of the input (<2^64).
#[inline]
unsafe fn concrete_avx(h0: &__m256i, h1: &__m256i, h2: &__m256i, l0: &mut __m256i, l1: &mut __m256i, l2: &mut __m256i, round_constants: &[u64; SPONGE_WIDTH]) {
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let mut sh0 = zeros;
    let mut sh1 = zeros;
    let mut sh2 = zeros;
    let mut sl0 = zeros;
    let mut sl1 = zeros;
    let mut sl2 = zeros;
    for column in 0..SPONGE_WIDTH {
        let mm0 = _mm256_set_epi64x(MONOLITH_MAT_12[3][column] as i64, MONOLITH_MAT_12[2][column] as i64, MONOLITH_MAT_12[1][column] as i64, MONOLITH_MAT_12[0][column] as i64);
        let mm1 = _mm256_set_epi64x(MONOLITH_MAT_12[7][column] as i64, MONOLITH_MAT_12[6][column] as i64, MONOLITH_MAT_12[5][column] as i64, MONOLITH_MAT_12[4][column] as i64);
        let mm2 = _mm256_set_epi64x(MONOLITH_MAT_12[11][column] as i64, MONOLITH_MAT_12[10][column] as i64, MONOLITH_MAT_12[9][column] as i64, MONOLITH_MAT_12[8][column] as i64);
        let tl = match column {
            0 => _mm256_permute4x64_epi64(*l0, 0x0),
            1 => _mm256_permute4x64_epi64(*l0, 0x55),
            2 => _mm256_permute4x64_epi64(*l0, 0xAA),
            3 => _mm256_permute4x64_epi64(*l0, 0xFF),
            4 => _mm256_permute4x64_epi64(*l1, 0x0),
            5 => _mm256_permute4x64_epi64(*l1, 0x55),
            6 => _mm256_permute4x64_epi64(*l1, 0xAA),
            7 => _mm256_permute4x64_epi64(*l1, 0xFF),
            8 => _mm256_permute4x64_epi64(*l2, 0x0),
            9 => _mm256_permute4x64_epi64(*l2, 0x55),
            10 => _mm256_permute4x64_epi64(*l2, 0xAA),
            11 => _mm256_permute4x64_epi64(*l2, 0xFF),
            _ => zeros,
        };
        let th = match column {
            0 => _mm256_permute4x64_epi64(*h0, 0x0),
            1 => _mm256_permute4x64_epi64(*h0, 0x55),
            2 => _mm256_permute4x64_epi64(*h0, 0xAA),
            3 => _mm256_permute4x64_epi64(*h0, 0xFF),
            4 => _mm256_permute4x64_epi64(*h1, 0x0),
            5 => _mm256_permute4x64_epi64(*h1, 0x55),
            6 => _mm256_permute4x64_epi64(*h1, 0xAA),
            7 => _mm256_permute4x64_epi64(*h1, 0xFF),
            8 => _mm256_permute4x64_epi64(*h2, 0x0),
            9 => _mm256_permute4x64_epi64(*h2, 0x55),
            10 => _mm256_permute4x64_epi64(*h2, 0xAA),
            11 => _mm256_permute4x64_epi64(*h2, 0xFF),
            _ => zeros,
        };
        // change to simple mul
        let (mh0, ml0) = mul_u128_x_u64(&th, &tl, &mm0);
        let (mh1, ml1) = mul_u128_x_u64(&th, &tl, &mm1);
        let (mh2, ml2) = mul_u128_x_u64(&th, &tl, &mm2);

        // add with carry
        add_u128(&mut sh0, &mut sl0, &mh0, &ml0);
        add_u128(&mut sh1, &mut sl1, &mh1, &ml1);
        add_u128(&mut sh2, &mut sl2, &mh2, &ml2);
    }

    // add round constants
    let rc0 = _mm256_loadu_si256(round_constants[0..4].as_ptr().cast::<__m256i>());
    let rc1 = _mm256_loadu_si256(round_constants[4..8].as_ptr().cast::<__m256i>());
    let rc2 = _mm256_loadu_si256(round_constants[8..12].as_ptr().cast::<__m256i>());
    add_u128(&mut sh0, &mut sl0, &zeros, &rc0);
    add_u128(&mut sh1, &mut sl1, &zeros, &rc1);
    add_u128(&mut sh2, &mut sl2, &zeros, &rc2);

    // reduce u128 to u64 Goldilocks
    *l0 = reduce_avx_96_64(&sh0, &sl0);
    *l1 = reduce_avx_96_64(&sh1, &sl1);
    *l2 = reduce_avx_96_64(&sh2, &sl2);
}

// The high part (h0, h1, h2) is only for output (there is no high part from the previous operation).
// Note that h can only be 0 or 1, not higher.
unsafe fn bricks_avx(h0: &mut __m256i, h1: &mut __m256i, h2: &mut __m256i, l0: &mut __m256i, l1: &mut __m256i, l2: &mut __m256i) {
    // get prev using permute and blend
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let ss0 = _mm256_permute4x64_epi64(*l0, 0x90);
    let ss1 = _mm256_permute4x64_epi64(*l1, 0x93);
    let ss2 = _mm256_permute4x64_epi64(*l2, 0x93);
    let ss3 = _mm256_permute4x64_epi64(*l0, 0x3);
    let ss4 = _mm256_permute4x64_epi64(*l1, 0x3);
    let ss0 = _mm256_blend_epi32(ss0, zeros, 0x3);
    let ss1 = _mm256_blend_epi32(ss1, ss3, 0x3);
    let ss2 = _mm256_blend_epi32(ss2, ss4, 0x3);

    // square
    let p0 = sqr_avx(&ss0);
    let p1 = sqr_avx(&ss1);
    let p2 = sqr_avx(&ss2);

    // add
    (*l0, *h0) = add64_no_carry(l0, &p0);
    (*l1, *h1) = add64_no_carry(l1, &p1);
    (*l2, *h2) = add64_no_carry(l2, &p2);
}

/*
fn print_state(s0: &__m256i, s1:  &__m256i, s2:  &__m256i) {
    unsafe {
    println!("State:");
    let mut v: [u64; 4] = [0; 4];
    _mm256_storeu_si256(v.as_mut_ptr().cast::<__m256i>(), *s0);
    println!("{:X?}", v);
    _mm256_storeu_si256(v.as_mut_ptr().cast::<__m256i>(), *s1);
    println!("{:X?}", v);
    _mm256_storeu_si256(v.as_mut_ptr().cast::<__m256i>(), *s2);
    println!("{:X?}", v);
    }
}
*/

// input is obtained via to_noncanonical_u64()
#[inline]
pub fn monolith_avx(state: &mut [u64; SPONGE_WIDTH]) {
    unsafe {
        let zeros = _mm256_set_epi64x(0, 0, 0, 0);

        // low part of the state (< 2^64)
        let mut sl0 = _mm256_loadu_si256(state[0..4].as_ptr().cast::<__m256i>());
        let mut sl1 = _mm256_loadu_si256(state[4..8].as_ptr().cast::<__m256i>());
        let mut sl2 = _mm256_loadu_si256(state[8..12].as_ptr().cast::<__m256i>());

        // high part of the state (only after bricks operations)
        let mut sh0 = zeros;
        let mut sh1 = zeros;
        let mut sh2 = zeros;

        // rounds
        concrete_avx(&sh0, &sh1, &sh2,&mut sl0, &mut sl1, &mut sl2,&MONOLITH_ROUND_CONSTANTS[0]);
        for rc in MONOLITH_ROUND_CONSTANTS.iter().skip(1) {
            bar_avx(&mut sl0);
            bricks_avx(&mut sh0, &mut sh1, &mut sh2, &mut sl0, &mut sl1, &mut sl2);
            concrete_avx(&sh0, &sh1, &sh2,&mut sl0, &mut sl1, &mut sl2, rc);
        }

        // store states
        _mm256_storeu_si256(state[0..4].as_mut_ptr().cast::<__m256i>(), sl0);
        _mm256_storeu_si256(state[4..8].as_mut_ptr().cast::<__m256i>(), sl1);
        _mm256_storeu_si256(state[8..12].as_mut_ptr().cast::<__m256i>(), sl2);
    }
}
