// use core::arch::asm;
use core::arch::x86_64::*;

use crate::hash::hash_types::RichField;

const MSB_1: i64 = 0x8000000000000000u64 as i64;
const P_S_1: i64 = 0x7FFFFFFF00000001u64 as i64;
const P_N_1: i64 = 0xFFFFFFFF;

#[inline(always)]
pub fn shift_avx(a: &__m256i) -> __m256i {
    unsafe {
        let msb = _mm256_set_epi64x(MSB_1, MSB_1, MSB_1, MSB_1);
        _mm256_xor_si256(*a, msb)
    }
}

#[allow(dead_code)]
#[inline(always)]
pub fn to_canonical_avx_s(a_s: &__m256i) -> __m256i {
    unsafe {
        let p_s = _mm256_set_epi64x(P_S_1, P_S_1, P_S_1, P_S_1);
        let p_n = _mm256_set_epi64x(P_N_1, P_N_1, P_N_1, P_N_1);
        let mask1_ = _mm256_cmpgt_epi64(p_s, *a_s);
        let corr1_ = _mm256_andnot_si256(mask1_, p_n);
        _mm256_add_epi64(*a_s, corr1_)
    }
}

#[inline(always)]
pub fn add_avx_a_sc(a_sc: &__m256i, b: &__m256i) -> __m256i {
    unsafe {
        let c0_s = _mm256_add_epi64(*a_sc, *b);
        let p_n = _mm256_set_epi64x(P_N_1, P_N_1, P_N_1, P_N_1);
        let mask_ = _mm256_cmpgt_epi64(*a_sc, c0_s);
        let corr_ = _mm256_and_si256(mask_, p_n);
        let c_s = _mm256_add_epi64(c0_s, corr_);
        shift_avx(&c_s)
    }
}

#[inline(always)]
pub fn add_avx(a: &__m256i, b: &__m256i) -> __m256i {
    unsafe {
        let msb = _mm256_set_epi64x(MSB_1, MSB_1, MSB_1, MSB_1);
        let a_sc = _mm256_xor_si256(*a, msb);
        let c0_s = _mm256_add_epi64(a_sc, *b);
        let p_n = _mm256_set_epi64x(P_N_1, P_N_1, P_N_1, P_N_1);
        let mask_ = _mm256_cmpgt_epi64(a_sc, c0_s);
        let corr_ = _mm256_and_si256(mask_, p_n);
        let c_s = _mm256_add_epi64(c0_s, corr_);
        _mm256_xor_si256(c_s, msb)
    }
}

#[inline(always)]
pub fn add_avx_s_b_small(a_s: &__m256i, b_small: &__m256i) -> __m256i {
    unsafe {
        let c0_s = _mm256_add_epi64(*a_s, *b_small);
        let mask_ = _mm256_cmpgt_epi32(*a_s, c0_s);
        let corr_ = _mm256_srli_epi64(mask_, 32);
        _mm256_add_epi64(c0_s, corr_)
    }
}

#[inline(always)]
pub fn sub_avx_s_b_small(a_s: &__m256i, b: &__m256i) -> __m256i {
    unsafe {
        let c0_s = _mm256_sub_epi64(*a_s, *b);
        let mask_ = _mm256_cmpgt_epi32(c0_s, *a_s);
        let corr_ = _mm256_srli_epi64(mask_, 32);
        _mm256_sub_epi64(c0_s, corr_)
    }
}

#[inline(always)]
pub fn reduce_avx_128_64(c_h: &__m256i, c_l: &__m256i) -> __m256i {
    unsafe {
        let msb = _mm256_set_epi64x(MSB_1, MSB_1, MSB_1, MSB_1);
        let c_hh = _mm256_srli_epi64(*c_h, 32);
        let c_ls = _mm256_xor_si256(*c_l, msb);
        let c1_s = sub_avx_s_b_small(&c_ls, &c_hh);
        let p_n = _mm256_set_epi64x(P_N_1, P_N_1, P_N_1, P_N_1);
        let c2 = _mm256_mul_epu32(*c_h, p_n);
        let c_s = add_avx_s_b_small(&c1_s, &c2);
        _mm256_xor_si256(c_s, msb)
    }
}

// Here we suppose c_h < 2^32
#[allow(dead_code)]
#[inline(always)]
pub fn reduce_avx_96_64(c_h: &__m256i, c_l: &__m256i) -> __m256i {
    unsafe {
        let msb = _mm256_set_epi64x(MSB_1, MSB_1, MSB_1, MSB_1);
        let p_n = _mm256_set_epi64x(P_N_1, P_N_1, P_N_1, P_N_1);
        let c_ls = _mm256_xor_si256(*c_l, msb);
        let c2 = _mm256_mul_epu32(*c_h, p_n);
        let c_s = add_avx_s_b_small(&c_ls, &c2);
        _mm256_xor_si256(c_s, msb)
    }
}

/*
#[inline(always)]
pub fn mult_avx_128_v2(a: &__m256i, b: &__m256i) -> (__m256i, __m256i) {
    unsafe {
        let c_l: __m256i;
        let c_h: __m256i;
        let P_n = _mm256_set_epi64x(P_n_, P_n_, P_n_, P_n_);
        // ymm0 - a and c_h
        // ymm1 - b and c_l
        asm!(
            "vpsrlq ymm2, ymm0, 32",
            "vpsrlq ymm3, ymm1, 32",
            "vpmuludq ymm4, ymm2, ymm3",
            "vpmuludq ymm5, ymm2, ymm1",
            "vpmuludq ymm6, ymm0, ymm3",
            "vpmuludq ymm7, ymm0, ymm1",
            "vpsrlq ymm2, ymm7, 32",
            "vpaddq ymm3, ymm5, ymm2",
            "vpand ymm0, ymm3, ymm9",   // r0_l
            "vpsrlq ymm8, ymm3, 32",    // r0_h
            "vpaddq ymm2, ymm6, ymm0", // r1
            "vpsllq ymm0, ymm2, 32",
            "vpblendd ymm1, ymm7, ymm0, 0xaa",
            "vpaddq ymm3, ymm4, ymm8", // r2
            "vpsrlq ymm4, ymm2, 32",
            "vpaddq ymm0, ymm3, ymm4",
            inout("ymm0") *a => c_h,
            inout("ymm1") *b => c_l,
            in("ymm9") P_n
        );
        (c_h, c_l)
    }
}
*/

#[inline(always)]
pub fn mult_avx_128(a: &__m256i, b: &__m256i) -> (__m256i, __m256i) {
    unsafe {
        // let a_h = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(*a)));
        // let b_h = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(*b)));
        let a_h = _mm256_srli_epi64(*a, 32);
        let b_h = _mm256_srli_epi64(*b, 32);
        let c_hh = _mm256_mul_epu32(a_h, b_h);
        let c_hl = _mm256_mul_epu32(a_h, *b);
        let c_lh = _mm256_mul_epu32(*a, b_h);
        let c_ll = _mm256_mul_epu32(*a, *b);
        let c_ll_h = _mm256_srli_epi64(c_ll, 32);
        let r0 = _mm256_add_epi64(c_hl, c_ll_h);
        let p_n = _mm256_set_epi64x(P_N_1, P_N_1, P_N_1, P_N_1);
        let r0_l = _mm256_and_si256(r0, p_n);
        let r0_h = _mm256_srli_epi64(r0, 32);
        let r1 = _mm256_add_epi64(c_lh, r0_l);
        // let r1_l = _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(r1)));
        let r1_l = _mm256_slli_epi64(r1, 32);
        let c_l = _mm256_blend_epi32(c_ll, r1_l, 0xaa);
        let r2 = _mm256_add_epi64(c_hh, r0_h);
        let r1_h = _mm256_srli_epi64(r1, 32);
        let c_h = _mm256_add_epi64(r2, r1_h);
        (c_h, c_l)
    }
}

#[inline(always)]
pub fn mult_avx(a: &__m256i, b: &__m256i) -> __m256i {
    let (c_h, c_l) = mult_avx_128(a, b);
    reduce_avx_128_64(&c_h, &c_l)
}

// Multiply two 64bit numbers with the assumption that the product does not averflow.
#[allow(dead_code)]
#[inline]
pub unsafe fn mul64_no_overflow(a: &__m256i, b: &__m256i) -> __m256i {
    let r = _mm256_mul_epu32(*a, *b);
    let ah = _mm256_srli_epi64(*a, 32);
    let bh = _mm256_srli_epi64(*b, 32);
    let r1 = _mm256_mul_epu32(*a, bh);
    let r1 = _mm256_slli_epi64(r1, 32);
    let r = _mm256_add_epi64(r, r1);
    let r1 = _mm256_mul_epu32(ah, *b);
    let r1 = _mm256_slli_epi64(r1, 32);
    let r = _mm256_add_epi64(r, r1);
    r
}

/*
#[inline(always)]
pub fn mult_avx_v2(a: &__m256i, b: &__m256i) -> __m256i {
unsafe {
    let c: __m256i;
    let P_n = _mm256_set_epi64x(P_n_, P_n_, P_n_, P_n_);
    let MSB = _mm256_set_epi64x(MSB_, MSB_, MSB_, MSB_);
    // ymm0 - a and c_h
    // ymm1 - b and c_l
    asm!(
        // mul
        "vpsrlq ymm2, ymm0, 32",
        "vpsrlq ymm3, ymm1, 32",
        "vpmuludq ymm4, ymm2, ymm3",
        "vpmuludq ymm5, ymm2, ymm1",
        "vpmuludq ymm6, ymm0, ymm3",
        "vpmuludq ymm7, ymm0, ymm1",
        "vpsrlq ymm2, ymm7, 32",
        "vpaddq ymm3, ymm5, ymm2",
        "vpand ymm0, ymm3, ymm9",   // r0_l
        "vpsrlq ymm8, ymm3, 32",    // r0_h
        "vpaddq ymm2, ymm6, ymm0", // r1
        "vpsllq ymm0, ymm2, 32",
        "vpblendd ymm1, ymm7, ymm0, 0xaa",
        "vpaddq ymm3, ymm4, ymm8", // r2
        "vpsrlq ymm4, ymm2, 32",
        "vpaddq ymm0, ymm3, ymm4",
        // reduce
        "vpsrlq ymm2, ymm0, 32",
        "vpxor ymm3, ymm1, ymm10",
        // sub
        "vpsubq ymm4, ymm3, ymm2",
        "vpcmpgtq ymm2, ymm4, ymm3",
        "vpsrlq ymm2, ymm2, 32",
        "vpsubq ymm3, ymm4, ymm2",
        "vpmuludq ymm7, ymm0, ymm9",
        // add
        "vpaddq ymm2, ymm3, ymm7",
        "vpcmpgtq ymm4, ymm3, ymm2",
        "vpsrlq ymm4, ymm4, 32",
        "vpaddq ymm3, ymm2, ymm4",
        "vpxor ymm0, ymm3, ymm10",
        inout("ymm0") *a => c,
        inout("ymm1") *b => _,
        in("ymm9") P_n,
        in("ymm10") MSB
    );
    c
}
}
*/

#[inline(always)]
pub fn sqr_avx_128(a: &__m256i) -> (__m256i, __m256i) {
    unsafe {
        let a_h = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(*a)));
        let c_ll = _mm256_mul_epu32(*a, *a);
        let c_lh = _mm256_mul_epu32(*a, a_h);
        let c_hh = _mm256_mul_epu32(a_h, a_h);
        let c_ll_hi = _mm256_srli_epi64(c_ll, 33);
        let t0 = _mm256_add_epi64(c_lh, c_ll_hi);
        let t0_hi = _mm256_srli_epi64(t0, 31);
        let res_hi = _mm256_add_epi64(c_hh, t0_hi);
        let c_lh_lo = _mm256_slli_epi64(c_lh, 33);
        let res_lo = _mm256_add_epi64(c_ll, c_lh_lo);
        (res_hi, res_lo)
    }
}

#[inline(always)]
pub fn sqr_avx(a: &__m256i) -> __m256i {
    let (c_h, c_l) = sqr_avx_128(a);
    reduce_avx_128_64(&c_h, &c_l)
}

#[inline(always)]
pub fn sbox_avx<F>(state: &mut [F; 12])
where
    F: RichField,
{
    unsafe {
        let s0 = _mm256_loadu_si256((&state[0..4]).as_ptr().cast::<__m256i>());
        let s1 = _mm256_loadu_si256((&state[4..8]).as_ptr().cast::<__m256i>());
        let s2 = _mm256_loadu_si256((&state[8..12]).as_ptr().cast::<__m256i>());
        // x^2
        let p10 = sqr_avx(&s0);
        let p11 = sqr_avx(&s1);
        let p12 = sqr_avx(&s2);
        // x^3
        let p20 = mult_avx(&p10, &s0);
        let p21 = mult_avx(&p11, &s1);
        let p22 = mult_avx(&p12, &s2);
        // x^4 = (x^2)^2
        let s0 = sqr_avx(&p10);
        let s1 = sqr_avx(&p11);
        let s2 = sqr_avx(&p12);
        // x^7
        let p10 = mult_avx(&s0, &p20);
        let p11 = mult_avx(&s1, &p21);
        let p12 = mult_avx(&s2, &p22);
        _mm256_storeu_si256((&mut state[0..4]).as_mut_ptr().cast::<__m256i>(), p10);
        _mm256_storeu_si256((&mut state[4..8]).as_mut_ptr().cast::<__m256i>(), p11);
        _mm256_storeu_si256((&mut state[8..12]).as_mut_ptr().cast::<__m256i>(), p12);
    }
}

#[allow(dead_code)]
#[inline(always)]
pub fn sbox_avx_m256i(s0: &__m256i, s1: &__m256i, s2: &__m256i) -> (__m256i, __m256i, __m256i) {
    // x^2
    let p10 = sqr_avx(s0);
    let p11 = sqr_avx(s1);
    let p12 = sqr_avx(s2);
    // x^3
    let p30 = mult_avx(&p10, s0);
    let p31 = mult_avx(&p11, s1);
    let p32 = mult_avx(&p12, s2);
    // x^4 = (x^2)^2
    let p40 = sqr_avx(&p10);
    let p41 = sqr_avx(&p11);
    let p42 = sqr_avx(&p12);
    // x^7
    let r0 = mult_avx(&p40, &p30);
    let r1 = mult_avx(&p41, &p31);
    let r2 = mult_avx(&p42, &p32);

    (r0, r1, r2)
}

#[allow(dead_code)]
#[inline(always)]
pub fn sbox_avx_one(s0: &__m256i) -> __m256i {
    // x^2
    let p10 = sqr_avx(s0);
    // x^3
    let p30 = mult_avx(&p10, s0);
    // x^4 = (x^2)^2
    let p40 = sqr_avx(&p10);
    // x^7
    mult_avx(&p40, &p30)
}
