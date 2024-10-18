// use core::arch::asm;
use core::arch::x86_64::*;

use crate::hash::hash_types::RichField;

const MSB_: i64 = 0x8000000000000000u64 as i64;
const P8_: i64 = 0xFFFFFFFF00000001u64 as i64;
const P8_N_: i64 = 0xFFFFFFFF;
const ONE_: i64 = 1;

#[allow(non_snake_case)]
#[repr(align(64))]
pub(crate) struct FieldConstants {
    pub(crate) MSB_V: [i64; 8],
    pub(crate) P8_V: [i64; 8],
    pub(crate) P8_N_V: [i64; 8],
    pub(crate) ONE_V: [i64; 8],
}

pub(crate) const FC: FieldConstants = FieldConstants {
    MSB_V: [MSB_, MSB_, MSB_, MSB_, MSB_, MSB_, MSB_, MSB_],
    P8_V: [P8_, P8_, P8_, P8_, P8_, P8_, P8_, P8_],
    P8_N_V: [P8_N_, P8_N_, P8_N_, P8_N_, P8_N_, P8_N_, P8_N_, P8_N_],
    ONE_V: [ONE_, ONE_, ONE_, ONE_, ONE_, ONE_, ONE_, ONE_],
};

#[allow(dead_code)]
#[inline(always)]
pub fn shift_avx512(a: &__m512i) -> __m512i {
    unsafe {
        // let msb = _mm512_set_epi64(MSB_, MSB_, MSB_, MSB_, MSB_, MSB_, MSB_, MSB_);
        let msb = _mm512_load_si512(FC.MSB_V.as_ptr().cast::<i32>());
        _mm512_xor_si512(*a, msb)
    }
}

#[allow(dead_code)]
#[inline(always)]
pub fn to_canonical_avx512(a: &__m512i) -> __m512i {
    unsafe {
        // let p8 = _mm512_set_epi64(P8_, P8_, P8_, P8_, P8_, P8_, P8_, P8_);
        // let p8_n = _mm512_set_epi64(P8_N_, P8_N_, P8_N_, P8_N_, P8_N_, P8_N_, P8_N_, P8_N_);
        let p8 = _mm512_load_si512(FC.P8_V.as_ptr().cast::<i32>());
        let p8_n = _mm512_load_si512(FC.P8_N_V.as_ptr().cast::<i32>());
        let result_mask = _mm512_cmpge_epu64_mask(*a, p8);
        _mm512_mask_add_epi64(*a, result_mask, *a, p8_n)
    }
}

#[inline(always)]
pub fn add_avx512(a: &__m512i, b: &__m512i) -> __m512i {
    /*
    unsafe {
        // let p8_n = _mm512_set_epi64(P8_N_, P8_N_, P8_N_, P8_N_, P8_N_, P8_N_, P8_N_, P8_N_);
        let p8_n = _mm512_load_epi64(FC.P8_N_V.as_ptr().cast::<i64>());
        let c0 = _mm512_add_epi64(*a, *b);
        let result_mask = _mm512_cmpgt_epu64_mask(*a, c0);
        _mm512_mask_add_epi64(c0, result_mask, c0, p8_n)
    }
    */
    unsafe {
        let msb = _mm512_load_epi64(FC.MSB_V.as_ptr().cast::<i64>());
        let a_sc = _mm512_xor_si512(*a, msb);
        let c0_s = _mm512_add_epi64(a_sc, *b);
        let p_n = _mm512_load_epi64(FC.P8_N_V.as_ptr().cast::<i64>());
        let mask_ = _mm512_cmpgt_epi64_mask(a_sc, c0_s);
        let c_s = _mm512_mask_add_epi64(c0_s, mask_, c0_s, p_n);
        _mm512_xor_si512(c_s, msb)
    }
}

#[inline(always)]
pub fn add_avx512_s_b_small(a_s: &__m512i, b_small: &__m512i) -> __m512i {
    unsafe {
        let corr = _mm512_load_epi64(FC.P8_N_V.as_ptr().cast::<i64>());
        let c0_s = _mm512_add_epi64(*a_s, *b_small);
        let mask_ = _mm512_cmpgt_epi64_mask(*a_s, c0_s);
        _mm512_mask_add_epi64(c0_s, mask_, c0_s, corr)
    }
}

#[inline(always)]
pub fn sub_avx512(a: &__m512i, b: &__m512i) -> __m512i {
    unsafe {
        // let p8 = _mm512_set_epi64(P8_, P8_, P8_, P8_, P8_, P8_, P8_, P8_);
        let p8 = _mm512_load_si512(FC.P8_V.as_ptr().cast::<i32>());
        let c0 = _mm512_sub_epi64(*a, *b);
        let result_mask = _mm512_cmpgt_epu64_mask(*b, *a);
        _mm512_mask_add_epi64(c0, result_mask, c0, p8)
    }
}

#[inline(always)]
pub fn reduce_avx512_128_64(c_h: &__m512i, c_l: &__m512i) -> __m512i {
    unsafe {
        // let p8_n = _mm512_set_epi64(P8_N_, P8_N_, P8_N_, P8_N_, P8_N_, P8_N_, P8_N_, P8_N_);
        let p8_n = _mm512_load_si512(FC.P8_N_V.as_ptr().cast::<i32>());
        let c_hh = _mm512_srli_epi64(*c_h, 32);
        let c1 = sub_avx512(c_l, &c_hh);
        let c2 = _mm512_mul_epu32(*c_h, p8_n);
        add_avx512(&c1, &c2)
    }
}

// Here we suppose c_h < 2^32
#[inline(always)]
pub fn reduce_avx512_96_64(c_h: &__m512i, c_l: &__m512i) -> __m512i {
    unsafe {
        let msb = _mm512_load_epi64(FC.MSB_V.as_ptr().cast::<i64>());
        let p_n = _mm512_load_epi64(FC.P8_N_V.as_ptr().cast::<i64>());
        let c_ls = _mm512_xor_si512(*c_l, msb);
        let c2 = _mm512_mul_epu32(*c_h, p_n);
        let c_s = add_avx512_s_b_small(&c_ls, &c2);
        // let c_s = add_avx512(&c_ls, &c2);
        _mm512_xor_si512(c_s, msb)
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
        // let p8_n = _mm512_set_epi64(P8_N_, P8_N_, P8_N_, P8_N_, P8_N_, P8_N_, P8_N_, P8_N_);
        let p8_n = _mm512_load_si512(FC.P8_N_V.as_ptr().cast::<i32>());
        let r0_l = _mm512_and_si512(r0, p8_n);
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

#[allow(dead_code)]
#[inline(always)]
pub fn sbox_avx512<F>(state: &mut [F; 16])
where
    F: RichField,
{
    unsafe {
        let s0 = _mm512_loadu_si512((&state[0..8]).as_ptr().cast::<i32>());
        let s1 = _mm512_loadu_si512((&state[8..16]).as_ptr().cast::<i32>());
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
        _mm512_storeu_si512((&mut state[0..8]).as_mut_ptr().cast::<i32>(), p10);
        _mm512_storeu_si512((&mut state[8..16]).as_mut_ptr().cast::<i32>(), p11);
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
