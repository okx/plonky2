// use core::arch::asm;
use core::arch::x86_64::*;

use crate::hash::hash_types::RichField;

const MSB_: i64 = 0x8000000000000000u64 as i64;
const P8_: i64 = 0xFFFFFFFF00000001u64 as i64;
const P8_n_: i64 = 0xFFFFFFFF;

#[allow(dead_code)]
#[inline(always)]
pub fn shift_avx512(a: &__m512i) -> __m512i {
    unsafe {
        let MSB = _mm512_set_epi64(MSB_, MSB_, MSB_, MSB_, MSB_, MSB_, MSB_, MSB_);
        _mm512_xor_si512(*a, MSB)
    }
}

#[allow(dead_code)]
#[inline(always)]
pub fn toCanonical_avx512(a_s: &__m512i) -> __m512i {
    unsafe {
        let P8 = _mm512_set_epi64(P8_, P8_, P8_, P8_, P8_, P8_, P8_, P8_);
        let P8_n = _mm512_set_epi64(P8_n_, P8_n_, P8_n_, P8_n_, P8_n_, P8_n_, P8_n_, P8_n_);
        let result_mask = _mm512_cmpge_epu64_mask(a, P8);
        _mm512_mask_add_epi64(a, result_mask, a, P8_n)
    }
}

#[inline(always)]
pub fn add_avx512_b_c(a: &__m512i, b: &__m512i) -> __m512i {
    unsafe {
        let P8_n = _mm512_set_epi64(P8_n_, P8_n_, P8_n_, P8_n_, P8_n_, P8_n_, P8_n_, P8_n_);
        let c0 = _mm512_add_epi64(a, b);
        let result_mask = _mm512_cmpgt_epu64_mask(a, c0);
        _mm512_mask_add_epi64(c0, result_mask, c0, P8_n);
    }
}

#[inline(always)]
pub fn sub_avx512_b_c(a: &__m512i, b: &__m512i) -> __m512i {
    unsafe {
        let P8 = _mm512_set_epi64(P8_, P8_, P8_, P8_, P8_, P8_, P8_, P8_);
        let c0 = _mm512_sub_epi64(a, b);
        let result_mask = _mm512_cmpgt_epu64_mask(b, a);
        _mm512_mask_add_epi64(c0, result_mask, c0, P8);
    }
}

#[inline(always)]
pub fn reduce_avx512_128_64(c_h: &__m512i, c_l: &__m512i) -> __m512i {
    unsafe {
        let P8_n = _mm512_set_epi64(P8_n_, P8_n_, P8_n_, P8_n_, P8_n_, P8_n_, P8_n_, P8_n_);
        let c_hh = _mm512_srli_epi64(c_h, 32);
        let c1 = sub_avx512_b_c(*c_l, c_hh);
        let c2 = _mm512_mul_epu32(c_h, P8_n);
        add_avx512_b_c(c1, c2)
    }
}

#[inline(always)]
pub fn mult_avx512_128(a: &__m512i, b: &__m512i) -> (__m512i, __m512i) {
    unsafe {
        let a_h = _mm512_srli_epi64(*a, 32);
        let b_h = _mm512_srli_epi64(*b, 32);
        let c_hh = _mm512_mul_epu32(a_h, b_h);
        let c_hl = _mm512_mul_epu32(a_h, *b);
        let c_lh = _mm512_mul_epu32(*a, b_h);
        let c_ll = _mm512_mul_epu32(*a, *b);
        let c_ll_h = _mm512_srli_epi64(c_ll, 32);
        let r0 = _mm512_add_epi64(c_hl, c_ll_h);
        let P8_n = _mm512_set_epi64(P8_n_, P8_n_, P8_n_, P8_n_, P8_n_, P8_n_, P8_n_, P8_n_);
        let r0_l = _mm512_and_si512(r0, P8_n);
        let r0_h = _mm512_srli_epi64(r0, 32);
        let r1 = _mm512_add_epi64(c_lh, r0_l);
        let r1_l = _mm512_slli_epi64(r1, 32);
        let mask = 0xAAAAu16;
        let c_l = _mm512_mask_blend_epi32(mask, c_ll, r1_l);
        let r2 = _mm512_add_epi64(c_hh, r0_h);
        let r1_h = _mm512_srli_epi64(r1, 32);
        let c_h = _mm512_add_epi64(r2, r1_h);
        (c_h, c_l)
    }
}

#[inline(always)]
pub fn mult_avx512(a: &__m512i, b: &__m512i) -> __m512i {
    let (c_h, c_l) = mult_avx512_128(a, b);
    reduce_avx512_128_64(&c_h, &c_l)
}

#[inline(always)]
pub fn sqr_avx512_128(a: &__m512i) -> (__m512i, __m512i) {
    unsafe {
        let a_h = _mm512_srli_epi64(*a, 32);
        let c_ll = _mm512_mul_epu32(*a, *a);
        let c_lh = _mm512_mul_epu32(*a, a_h);
        let c_hh = _mm512_mul_epu32(a_h, a_h);
        let c_ll_hi = _mm512_srli_epi64(c_ll, 33);
        let t0 = _mm512_add_epi64(c_lh, c_ll_hi);
        let t0_hi = _mm512_srli_epi64(t0, 31);
        let res_hi = _mm512_add_epi64(c_hh, t0_hi);
        let c_lh_lo = _mm512_slli_epi64(c_lh, 33);
        let res_lo = _mm512_add_epi64(c_ll, c_lh_lo);
        (res_hi, res_lo)
    }
}

#[inline(always)]
pub fn sqr_avx512(a: &__m512i) -> __m512i {
    let (c_h, c_l) = sqr_avx512_128(a);
    reduce_avx512_128_64(&c_h, &c_l)
}

#[inline(always)]
pub fn sbox_avx512<F>(state: &mut [F; 16])
where
    F: RichField,
{
    unsafe {
        let s0 = _mm512_loadu_si512((&state[0..8]).as_ptr().cast::<__m512i>());
        let s1 = _mm512_loadu_si512((&state[8..16]).as_ptr().cast::<__m512i>());
        // x^2
        let p10 = sqr_avx512(&s0);
        let p11 = sqr_avx512(&s1);
        // x^3
        let p20 = mult_avx512(&p10, &s0);
        let p21 = mult_avx512(&p11, &s1);
        // x^4 = (x^2)^2
        let s0 = sqr_avx512(&p10);
        let s1 = sqr_avx512(&p11);
        // x^7
        let p10 = mult_avx512(&s0, &p20);
        let p11 = mult_avx512(&s1, &p21);
        _mm512_storeu_si512((&mut state[0..8]).as_mut_ptr().cast::<__m512i>(), p10);
        _mm512_storeu_si512((&mut state[8..16]).as_mut_ptr().cast::<__m512i>(), p11);
    }
}

#[inline(always)]
pub fn sbox_avx512_one(s0: &__m512i) -> __m512i {
    // x^2
    let p10 = sqr_avx512(s0);
    // x^3
    let p30 = mult_avx512(&p10, s0);
    // x^4 = (x^2)^2
    let p40 = sqr_avx512(&p10);
    // x^7
    mult_avx512(&p40, &p30)
}
