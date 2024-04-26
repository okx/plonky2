use core::arch::x86_64::*;

use super::goldilocks_avx2::{mult_avx_128, reduce_avx_128_64, sqr_avx};
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
        let tmp = _mm256_xor_si256(*el, limb1);
        let tmp = _mm256_and_si256(tmp, limb2);
        let tmp = _mm256_and_si256(tmp, limb3);
        let l1 = _mm256_and_si256(tmp, ct1);
        let l2 = _mm256_srli_epi64(l1, 7);
        let l3 = _mm256_andnot_si256(tmp, ct2);
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
        let tmp = _mm256_xor_si256(*el, limb1);
        let tmp = _mm256_and_si256(tmp, limb2);
        let tmp = _mm256_and_si256(tmp, limb3);
        let l1 = _mm256_and_si256(tmp, ct1);
        let l2 = _mm256_srli_epi64(l1, 15);
        let l3 = _mm256_andnot_si256(tmp, ct2);
        let l4 = _mm256_slli_epi64(l3, 1);
        *el = _mm256_or_si256(l2, l4);
    }
}

#[inline]
unsafe fn bars_avx(s0: &mut __m256i, s1: &mut __m256i, s2: &mut __m256i) {
    bar_avx(s0);
    bar_avx(s1);
    bar_avx(s2);
}

#[inline]
unsafe fn add_u128(sumhi: &mut __m256i, sumlo: &mut __m256i, hi: &__m256i, lo: &__m256i) {
    let (r, c) = add64_no_carry(sumlo, lo);
    (*sumhi, _) = add64(sumhi, hi, &c);
    *sumlo = r;
}

#[inline]
unsafe fn concrete_avx(s0: &mut __m256i, s1: &mut __m256i, s2: &mut __m256i, round_constants: &[u64; SPONGE_WIDTH]) {
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let mut sh0 = zeros;
    let mut sh1 = zeros;
    let mut sh2 = zeros;
    let mut sl0 = zeros;
    let mut sl1 = zeros;
    let mut sl2 = zeros;
    let mut c0 = zeros;
    let mut c1 = zeros;
    let mut c2 = zeros;
    for column in 0..SPONGE_WIDTH {
        let mm0 = _mm256_set_epi64x(MONOLITH_MAT_12[3][column] as i64, MONOLITH_MAT_12[2][column] as i64, MONOLITH_MAT_12[1][column] as i64, MONOLITH_MAT_12[0][column] as i64);
        let mm1 = _mm256_set_epi64x(MONOLITH_MAT_12[7][column] as i64, MONOLITH_MAT_12[6][column] as i64, MONOLITH_MAT_12[5][column] as i64, MONOLITH_MAT_12[4][column] as i64);
        let mm2 = _mm256_set_epi64x(MONOLITH_MAT_12[11][column] as i64, MONOLITH_MAT_12[10][column] as i64, MONOLITH_MAT_12[9][column] as i64, MONOLITH_MAT_12[8][column] as i64);
        // change to simple mul
        let (h0, l0) = mult_avx_128(s0, &mm0);
        let (h1, l1) = mult_avx_128(s1, &mm1);
        let (h2, l2) = mult_avx_128(s2, &mm2);
        // add with carry
        add_u128(&mut sh0, &mut sl0, &h0, &l0);
        add_u128(&mut sh1, &mut sl1, &h1, &l1);
        add_u128(&mut sh2, &mut sl2, &h2, &l2);
    }
    let rc0 = _mm256_loadu_si256(round_constants[0..4].as_ptr().cast::<__m256i>());
    let rc1 = _mm256_loadu_si256(round_constants[0..4].as_ptr().cast::<__m256i>());
    let rc2 = _mm256_loadu_si256(round_constants[0..4].as_ptr().cast::<__m256i>());
    add_u128(&mut sh0, &mut sl0, &zeros, &rc0);
    add_u128(&mut sh1, &mut sl1, &zeros, &rc1);
    add_u128(&mut sh2, &mut sl2, &zeros, &rc2);
    *s0 = reduce_avx_128_64(&sh0, &sl0);
    *s1 = reduce_avx_128_64(&sh1, &sl1);
    *s2 = reduce_avx_128_64(&sh2, &sl2);
}

unsafe fn bricks_avx(s0: &mut __m256i, s1: &mut __m256i, s2: &mut __m256i) {
    // get prev using permute and blend
    let ss0 = _mm256_permute4x64_epi64(*s0, 0x39);
    let ss1 = _mm256_permute4x64_epi64(*s1, 0x39);
    let ss2 = _mm256_permute4x64_epi64(*s2, 0x39);
    let ss3 = _mm256_permute4x64_epi64(*s1, 0x0);
    let ss4 = _mm256_permute4x64_epi64(*s2, 0x0);
    let ss0 = _mm256_blend_epi32(ss0, ss3, 0xC0);
    let ss1 = _mm256_blend_epi32(ss1, ss4, 0xC0);

    // square
    let p0 = sqr_avx(&ss0);
    let p1 = sqr_avx(&ss1);
    let p2 = sqr_avx(&ss2);

    // add
    *s0 = _mm256_add_epi64(*s0, p0);
    *s0 = _mm256_add_epi64(*s1, p1);
    *s0 = _mm256_add_epi64(*s2, p2);
}

// input is obtained via to_noncanonical_u64() as u128
#[inline]
pub fn monolith_avx(state: &mut [u64; SPONGE_WIDTH]) {
    unsafe {
        let mut s0 = _mm256_loadu_si256(state[0..4].as_ptr().cast::<__m256i>());
        let mut s1 = _mm256_loadu_si256(state[4..8].as_ptr().cast::<__m256i>());
        let mut s2 = _mm256_loadu_si256(state[8..12].as_ptr().cast::<__m256i>());

        concrete_avx(&mut s0, &mut s1, &mut s2, &MONOLITH_ROUND_CONSTANTS[0]);
        for rc in MONOLITH_ROUND_CONSTANTS.iter().skip(1) {
            bars_avx(&mut s0, &mut s1, &mut s2);
            bricks_avx(&mut s0, &mut s1, &mut s2);
            concrete_avx(&mut s0, &mut s1, &mut s2, rc);
        }

        _mm256_storeu_si256(state[0..4].as_mut_ptr().cast::<__m256i>(), s0);
        _mm256_storeu_si256(state[4..8].as_mut_ptr().cast::<__m256i>(), s1);
        _mm256_storeu_si256(state[8..12].as_mut_ptr().cast::<__m256i>(), s2);
    }
}
