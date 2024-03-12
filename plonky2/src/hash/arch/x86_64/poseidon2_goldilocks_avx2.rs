/// Code taken and adapted from: https://github.com/0xPolygonHermez/goldilocks/blob/master/src/goldilocks_base_field_avx.hpp

use crate::hash::{hash_types::RichField, poseidon2::{SPONGE_WIDTH}};
use core::arch::x86_64::*;

const MSB_: i64 = 0x8000000000000000u64 as i64;
const P_s_: i64 = 0x7FFFFFFF00000001u64 as i64;
const P_n_: i64 = 0xFFFFFFFF;

#[inline]
fn shift_avx(a_s: &mut __m256i, a: &__m256i)
{
    unsafe {
        let MSB = _mm256_set_epi64x(MSB_, MSB_, MSB_, MSB_);
        *a_s = _mm256_xor_si256(*a, MSB);
    }
}

#[inline]
fn toCanonical_avx_s(a_sc: &mut __m256i, a_s: &__m256i)
{
    unsafe {
        let P_s = _mm256_set_epi64x(P_s_, P_s_, P_s_, P_s_);
        let P_n = _mm256_set_epi64x(P_n_, P_n_, P_n_, P_n_);
        let mask1_ = _mm256_cmpgt_epi64(P_s, *a_s);
        let corr1_ = _mm256_andnot_si256(mask1_, P_n);
        *a_sc = _mm256_add_epi64(*a_s, corr1_);
    }
}

#[inline]
fn add_avx_a_sc(c: &mut __m256i, a_sc: &__m256i, b: &__m256i)
{
    unsafe {
        let c0_s = _mm256_add_epi64(*a_sc, *b);
        let P_n = _mm256_set_epi64x(P_n_, P_n_, P_n_, P_n_);
        let mask_ = _mm256_cmpgt_epi64(*a_sc, c0_s);
        let corr_ = _mm256_and_si256(mask_, P_n);
        let c_s = _mm256_add_epi64(c0_s, corr_);
        shift_avx(c, &c_s);
    }
}

#[inline]
fn add_avx(c: &mut __m256i, a: &__m256i, b: &__m256i)
{
    unsafe {
        let mut a_s: __m256i = _mm256_set_epi64x(0, 0, 0, 0);
        let mut a_sc: __m256i = _mm256_set_epi64x(0, 0, 0, 0);
        shift_avx(&mut a_s, a);
        toCanonical_avx_s(&mut a_sc, &a_s);
        add_avx_a_sc(c, &a_sc, b);
    }
}

#[inline] fn add_avx_s_b_small(c_s: &mut __m256i, a_s: &__m256i, b_small: &__m256i)
{
    unsafe {
        let c0_s = _mm256_add_epi64(*a_s, *b_small);
        let mask_ = _mm256_cmpgt_epi32(*a_s, c0_s);
        let corr_ = _mm256_srli_epi64(mask_, 32);
        *c_s = _mm256_add_epi64(c0_s, corr_);
    }
}

#[inline]
fn sub_avx_s_b_small(c_s: &mut __m256i, a_s: &__m256i, b: &__m256i)
{
    unsafe {
        let c0_s = _mm256_sub_epi64(*a_s, *b);
        let mask_ = _mm256_cmpgt_epi32(c0_s, *a_s);
        let corr_ = _mm256_srli_epi64(mask_, 32);
        *c_s = _mm256_sub_epi64(c0_s, corr_);
    }
}

#[inline] fn reduce_avx_128_64(c: &mut __m256i, c_h: &__m256i, c_l: &__m256i)
{
    unsafe {
        let c_hh = _mm256_srli_epi64(*c_h, 32);
        let mut c1_s: __m256i = _mm256_set_epi64x(0, 0, 0, 0);
        let mut c_ls: __m256i = _mm256_set_epi64x(0, 0, 0, 0);
        let mut c_s:__m256i = _mm256_set_epi64x(0, 0, 0, 0);
        shift_avx(&mut c_ls, c_l);
        sub_avx_s_b_small(&mut c1_s, &c_ls, &c_hh);
        let P_n = _mm256_set_epi64x(P_n_, P_n_, P_n_, P_n_);
        let c2 = _mm256_mul_epu32(*c_h, P_n);
        add_avx_s_b_small(&mut c_s, &c1_s, &c2);
        shift_avx(c, &c_s);
    }
}

#[inline ]
fn mult_avx_128(c_h: &mut __m256i, c_l: &mut __m256i, a: &__m256i, b: &__m256i)
{
    unsafe {
        let a_h = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(*a)));
        let b_h = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(*b)));
        let c_hh = _mm256_mul_epu32(a_h, b_h);
        let c_hl = _mm256_mul_epu32(a_h, *b);
        let c_lh = _mm256_mul_epu32(*a, b_h);
        let c_ll = _mm256_mul_epu32(*a, *b);
        let c_ll_h = _mm256_srli_epi64(c_ll, 32);
        let r0 = _mm256_add_epi64(c_hl, c_ll_h);
        let P_n = _mm256_set_epi64x(P_n_, P_n_, P_n_, P_n_);
        let r0_l = _mm256_and_si256(r0, P_n);
        let r1 = _mm256_add_epi64(c_lh, r0_l);
        let r1_l = _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(r1)));
        *c_l = _mm256_blend_epi32(c_ll, r1_l, 0xaa);
        let r0_h = _mm256_srli_epi64(r0, 32);
        let r2 = _mm256_add_epi64(c_hh, r0_h);
        let r1_h = _mm256_srli_epi64(r1, 32);
        *c_h = _mm256_add_epi64(r2, r1_h);
    }
}

#[inline]
fn mult_avx(c: &mut __m256i, a: &__m256i, b: &__m256i)
{
    unsafe {
        let mut c_h = _mm256_set_epi64x(0, 0, 0, 0);
        let mut c_l = _mm256_set_epi64x(0, 0, 0, 0);
        mult_avx_128(&mut c_h, &mut c_l, a, b);
        reduce_avx_128_64(c, &c_h, &c_l);
    }
}

pub fn add_rc_avx<F>(state: &mut [F; SPONGE_WIDTH], rc: &[u64; SPONGE_WIDTH])
where
    F: RichField,
{
    unsafe {
        let s0 = _mm256_loadu_si256((&state[0..4]).as_ptr().cast::<__m256i>());
        let s1 = _mm256_loadu_si256((&state[4..8]).as_ptr().cast::<__m256i>());
        let s2 = _mm256_loadu_si256((&state[8..12]).as_ptr().cast::<__m256i>());
        let rc0 = _mm256_loadu_si256((&rc[0..4]).as_ptr().cast::<__m256i>());
        let rc1 = _mm256_loadu_si256((&rc[4..8]).as_ptr().cast::<__m256i>());
        let rc2 = _mm256_loadu_si256((&rc[8..12]).as_ptr().cast::<__m256i>());
        let mut s : __m256i = _mm256_set_epi64x(0, 0, 0, 0);
        add_avx(&mut s, &s0, &rc0);
        _mm256_storeu_si256((&mut state[0..4]).as_mut_ptr().cast::<__m256i>(), s);
        add_avx(&mut s, &s1, &rc1);
        _mm256_storeu_si256((&mut state[4..8]).as_mut_ptr().cast::<__m256i>(), s);
        add_avx(&mut s, &s2, &rc2);
        _mm256_storeu_si256((&mut state[8..12]).as_mut_ptr().cast::<__m256i>(), s);
    }
}

pub fn sbox_avx<F>(state: &mut [F; SPONGE_WIDTH])
where
    F: RichField,
{
    unsafe {
        let mut s0 = _mm256_loadu_si256((&state[0..4]).as_ptr().cast::<__m256i>());
        let mut s1 = _mm256_loadu_si256((&state[4..8]).as_ptr().cast::<__m256i>());
        let mut s2 = _mm256_loadu_si256((&state[8..12]).as_ptr().cast::<__m256i>());
        let mut p10: __m256i = _mm256_set_epi64x(0, 0, 0, 0);
        let mut p11: __m256i = _mm256_set_epi64x(0, 0, 0, 0);
        let mut p12: __m256i = _mm256_set_epi64x(0, 0, 0, 0);
        let mut p20: __m256i = _mm256_set_epi64x(0, 0, 0, 0);
        let mut p21: __m256i = _mm256_set_epi64x(0, 0, 0, 0);
        let mut p22: __m256i = _mm256_set_epi64x(0, 0, 0, 0);
        // x^2
        mult_avx(&mut p10, &s0, &s0);
        mult_avx(&mut p11, &s1, &s1);
        mult_avx(&mut p12, &s2, &s2);
        // x^3
        mult_avx(&mut p20, &p10, &s0);
        mult_avx(&mut p21, &p11, &s1);
        mult_avx(&mut p22, &p12, &s2);
        // x^4
        mult_avx(&mut s0, &p10, &p10);
        mult_avx(&mut s1, &p11, &p11);
        mult_avx(&mut s2, &p12, &p12);
        // x^7
        mult_avx(&mut p10, &s0, &p20);
        mult_avx(&mut p11, &s1, &p21);
        mult_avx(&mut p12, &s2, &p22);
        _mm256_storeu_si256((&mut state[0..4]).as_mut_ptr().cast::<__m256i>(), p10);
        _mm256_storeu_si256((&mut state[4..8]).as_mut_ptr().cast::<__m256i>(), p11);
        _mm256_storeu_si256((&mut state[8..12]).as_mut_ptr().cast::<__m256i>(), p12);
    }
}

#[inline]
fn apply_m_4_avx<F>(r: &mut __m256i, x: &__m256i, s: &[F])
where
    F: RichField,
{
    // This is based on apply_m_4, but we pack 4 and then 2 operands per operation
    unsafe {
        let y = _mm256_set_epi64x(s[3].to_canonical_u64() as i64, s[3].to_canonical_u64() as i64, s[1].to_canonical_u64() as i64, s[1].to_canonical_u64() as i64);
        let mut t = _mm256_set_epi64x(0, 0, 0, 0);
        let mut v = _mm256_set_epi64x(0, 0, 0, 0);
        add_avx(&mut t, &x, &y);
        let mut tt: [i64; 4] = [0; 4];
        _mm256_storeu_si256((&mut tt).as_mut_ptr().cast::<__m256i>(), t);
        let y = _mm256_set_epi64x(tt[0], 0, tt[2], 0);
        add_avx(&mut v, &t, &y);
        _mm256_storeu_si256((&mut tt).as_mut_ptr().cast::<__m256i>(), v);
        let y = _mm256_set_epi64x(0, 0, tt[2], tt[0]);
        add_avx(&mut t, &y, &y);
        add_avx(&mut v, &t, &t);
        let y = _mm256_set_epi64x(0, 0, tt[3], tt[1]);
        add_avx(&mut t, &v, &y);
        let y = _mm256_set_epi64x(0, 0, tt[1], tt[3]);
        _mm256_storeu_si256((&mut tt).as_mut_ptr().cast::<__m256i>(), t);
        add_avx(&mut v, &t, &y);
        let mut vv: [i64; 4] = [0; 4];
        _mm256_storeu_si256((&mut vv).as_mut_ptr().cast::<__m256i>(), v);
        *r = _mm256_set_epi64x(tt[1], vv[1], tt[0], vv[0]);
    }
}

pub fn matmul_internal_avx<F>(
    state: &mut [F; SPONGE_WIDTH],
    mat_internal_diag_m_1: [u64; SPONGE_WIDTH],
)
where
    F: RichField,
{
    let mut sum = state[0];
    for i in 1..SPONGE_WIDTH {
        sum = sum + state[i];
    }
    let si64: i64 = sum.to_canonical_u64() as i64;
    unsafe {
        let s0 = _mm256_loadu_si256((&state[0..4]).as_ptr().cast::<__m256i>());
        let s1 = _mm256_loadu_si256((&state[4..8]).as_ptr().cast::<__m256i>());
        let s2 = _mm256_loadu_si256((&state[8..12]).as_ptr().cast::<__m256i>());
        let m0 = _mm256_loadu_si256((&mat_internal_diag_m_1[0..4]).as_ptr().cast::<__m256i>());
        let m1 = _mm256_loadu_si256((&mat_internal_diag_m_1[4..8]).as_ptr().cast::<__m256i>());
        let m2 = _mm256_loadu_si256((&mat_internal_diag_m_1[8..12]).as_ptr().cast::<__m256i>());
        let ss = _mm256_set_epi64x(si64, si64, si64, si64);
        let mut p10: __m256i = _mm256_set_epi64x(0, 0, 0, 0);
        let mut p11: __m256i = _mm256_set_epi64x(0, 0, 0, 0);
        let mut p12: __m256i = _mm256_set_epi64x(0, 0, 0, 0);
        let mut s: __m256i = _mm256_set_epi64x(0, 0, 0, 0);
        mult_avx(&mut p10, &s0, &m0);
        mult_avx(&mut p11, &s1, &m1);
        mult_avx(&mut p12, &s2, &m2);
        add_avx(&mut s, &p10, &ss);
        _mm256_storeu_si256((&mut state[0..4]).as_mut_ptr().cast::<__m256i>(), s);
        add_avx(&mut s, &p11, &ss);
        _mm256_storeu_si256((&mut state[4..8]).as_mut_ptr().cast::<__m256i>(), s);
        add_avx(&mut s, &p12, &ss);
        _mm256_storeu_si256((&mut state[8..12]).as_mut_ptr().cast::<__m256i>(), s);
    }
}

#[inline]
pub fn permute_mut_avx<F>(state: &mut [F; SPONGE_WIDTH])
where
    F: RichField,
{
    unsafe {
        let s0 = _mm256_loadu_si256((&state[0..4]).as_ptr().cast::<__m256i>());
        let s1 = _mm256_loadu_si256((&state[4..8]).as_ptr().cast::<__m256i>());
        let s2 = _mm256_loadu_si256((&state[8..12]).as_ptr().cast::<__m256i>());
        let mut r0: __m256i = _mm256_set_epi64x(0, 0, 0, 0);
        let mut r1: __m256i = _mm256_set_epi64x(0, 0, 0, 0);
        let mut r2: __m256i = _mm256_set_epi64x(0, 0, 0, 0);
        apply_m_4_avx(&mut r0, &s0, &state[0..4]);
        apply_m_4_avx(&mut r1, &s1, &state[4..8]);
        apply_m_4_avx(&mut r2, &s2, &state[8..12]);
        /*
        // Alternative
        for i in (0..SPONGE_WIDTH).step_by(4) {
            apply_m_4(&mut state[i..i + 4]);
        }
        let r0 = _mm256_loadu_si256((&state[0..4]).as_ptr().cast::<__m256i>());
        let r1 = _mm256_loadu_si256((&state[4..8]).as_ptr().cast::<__m256i>());
        let r2 = _mm256_loadu_si256((&state[8..12]).as_ptr().cast::<__m256i>());
        */
        let mut s3 = _mm256_set_epi64x(0, 0, 0, 0);
        let mut s = _mm256_set_epi64x(0, 0, 0, 0);
        add_avx(&mut s3, &r0, &r1);
        add_avx(&mut s, &r2, &s3);
        add_avx(&mut s3, &r0, &s);
        _mm256_storeu_si256((&mut state[0..4]).as_mut_ptr().cast::<__m256i>(), s3);
        add_avx(&mut s3, &r1, &s);
        _mm256_storeu_si256((&mut state[4..8]).as_mut_ptr().cast::<__m256i>(), s3);
        add_avx(&mut s3, &r2, &s);
        _mm256_storeu_si256((&mut state[8..12]).as_mut_ptr().cast::<__m256i>(), s3);
    }
}