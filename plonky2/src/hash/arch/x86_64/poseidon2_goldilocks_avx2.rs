/// Code taken and adapted from: https://github.com/0xPolygonHermez/goldilocks/blob/master/src/goldilocks_base_field_avx.hpp

use crate::hash::{hash_types::RichField, poseidon2::{SPONGE_WIDTH}};
use core::arch::x86_64::*;

const MSB_: i64 = 0x8000000000000000u64 as i64;
const P_s_: i64 = 0x7FFFFFFF00000001u64 as i64;
const P_n_: i64 = 0xFFFFFFFF;

#[inline]
fn shift_avx(a: &__m256i) -> __m256i
{
    unsafe {
        let MSB = _mm256_set_epi64x(MSB_, MSB_, MSB_, MSB_);
        _mm256_xor_si256(*a, MSB)
    }
}

#[inline]
fn toCanonical_avx_s(a_s: &__m256i) -> __m256i
{
    unsafe {
        let P_s = _mm256_set_epi64x(P_s_, P_s_, P_s_, P_s_);
        let P_n = _mm256_set_epi64x(P_n_, P_n_, P_n_, P_n_);
        let mask1_ = _mm256_cmpgt_epi64(P_s, *a_s);
        let corr1_ = _mm256_andnot_si256(mask1_, P_n);
        _mm256_add_epi64(*a_s, corr1_)
    }
}

#[inline]
fn add_avx_a_sc(a_sc: &__m256i, b: &__m256i) -> __m256i
{
    unsafe {
        let c0_s = _mm256_add_epi64(*a_sc, *b);
        let P_n = _mm256_set_epi64x(P_n_, P_n_, P_n_, P_n_);
        let mask_ = _mm256_cmpgt_epi64(*a_sc, c0_s);
        let corr_ = _mm256_and_si256(mask_, P_n);
        let c_s = _mm256_add_epi64(c0_s, corr_);
        shift_avx(&c_s)
    }
}

#[inline]
fn add_avx(a: &__m256i, b: &__m256i) -> __m256i
{
    let a_s = shift_avx(a);
    let a_sc = toCanonical_avx_s(&a_s);
    add_avx_a_sc(&a_sc, b)
}

#[inline]
fn add_avx_s_b_small(a_s: &__m256i, b_small: &__m256i) -> __m256i
{
    unsafe {
        let c0_s = _mm256_add_epi64(*a_s, *b_small);
        let mask_ = _mm256_cmpgt_epi32(*a_s, c0_s);
        let corr_ = _mm256_srli_epi64(mask_, 32);
        _mm256_add_epi64(c0_s, corr_)
    }
}

#[inline]
fn sub_avx_s_b_small(a_s: &__m256i, b: &__m256i) -> __m256i
{
    unsafe {
        let c0_s = _mm256_sub_epi64(*a_s, *b);
        let mask_ = _mm256_cmpgt_epi32(c0_s, *a_s);
        let corr_ = _mm256_srli_epi64(mask_, 32);
        _mm256_sub_epi64(c0_s, corr_)
    }
}

#[inline]
fn reduce_avx_128_64(c_h: &__m256i, c_l: &__m256i) -> __m256i
{
    unsafe {
        let c_hh = _mm256_srli_epi64(*c_h, 32);
        let c_ls = shift_avx(c_l);
        let c1_s = sub_avx_s_b_small(&c_ls, &c_hh);
        let P_n = _mm256_set_epi64x(P_n_, P_n_, P_n_, P_n_);
        let c2 = _mm256_mul_epu32(*c_h, P_n);
        let c_s = add_avx_s_b_small(&c1_s, &c2);
        shift_avx(&c_s)
    }
}

#[inline ]
fn mult_avx_128(a: &__m256i, b: &__m256i) -> (__m256i, __m256i)
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
        let r0_h = _mm256_srli_epi64(r0, 32);
        let r1 = _mm256_add_epi64(c_lh, r0_l);
        let r1_l = _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(r1)));
        let c_l = _mm256_blend_epi32(c_ll, r1_l, 0xaa);
        let r2 = _mm256_add_epi64(c_hh, r0_h);
        let r1_h = _mm256_srli_epi64(r1, 32);
        let c_h = _mm256_add_epi64(r2, r1_h);
        (c_h, c_l)
    }
}

#[inline]
fn mult_avx(a: &__m256i, b: &__m256i) -> __m256i
{
    let (c_h, c_l) = mult_avx_128(a, b);
    reduce_avx_128_64(&c_h, &c_l)
}

#[inline ]
fn sqr_avx_128(a: &__m256i) -> (__m256i, __m256i)
{
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

#[inline]
fn sqr_avx(a: &__m256i) -> __m256i
{
    let (c_h, c_l) = sqr_avx_128(a);
    reduce_avx_128_64(&c_h, &c_l)
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
        let s = add_avx(&s0, &rc0);
        _mm256_storeu_si256((&mut state[0..4]).as_mut_ptr().cast::<__m256i>(), s);
        let s = add_avx(&s1, &rc1);
        _mm256_storeu_si256((&mut state[4..8]).as_mut_ptr().cast::<__m256i>(), s);
        let s = add_avx(&s2, &rc2);
        _mm256_storeu_si256((&mut state[8..12]).as_mut_ptr().cast::<__m256i>(), s);
    }
}

pub fn sbox_avx<F>(state: &mut [F; SPONGE_WIDTH])
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

#[inline]
fn apply_m_4_avx<F>(x: &__m256i, s: &[F]) -> __m256i
where
    F: RichField,
{
    // This is based on apply_m_4, but we pack 4 and then 2 operands per operation
    unsafe {
        let y = _mm256_set_epi64x(s[3].to_canonical_u64() as i64, s[3].to_canonical_u64() as i64, s[1].to_canonical_u64() as i64, s[1].to_canonical_u64() as i64);
        let t = add_avx(&x, &y);
        let mut tt: [i64; 4] = [0; 4];
        _mm256_storeu_si256((&mut tt).as_mut_ptr().cast::<__m256i>(), t);
        let y = _mm256_set_epi64x(tt[0], 0, tt[2], 0);
        let v = add_avx(&t, &y);
        _mm256_storeu_si256((&mut tt).as_mut_ptr().cast::<__m256i>(), v);
        let y = _mm256_set_epi64x(0, 0, tt[2], tt[0]);
        let t = add_avx(&y, &y);
        let v = add_avx(&t, &t);
        let y = _mm256_set_epi64x(0, 0, tt[3], tt[1]);
        let t = add_avx(&v, &y);
        let y = _mm256_set_epi64x(0, 0, tt[1], tt[3]);
        _mm256_storeu_si256((&mut tt).as_mut_ptr().cast::<__m256i>(), t);
        let v = add_avx(&t, &y);
        let mut vv: [i64; 4] = [0; 4];
        _mm256_storeu_si256((&mut vv).as_mut_ptr().cast::<__m256i>(), v);
        _mm256_set_epi64x(tt[1], vv[1], tt[0], vv[0])
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
        let p10 = mult_avx(&s0, &m0);
        let p11 = mult_avx(&s1, &m1);
        let p12 = mult_avx(&s2, &m2);
        let s = add_avx(&p10, &ss);
        _mm256_storeu_si256((&mut state[0..4]).as_mut_ptr().cast::<__m256i>(), s);
        let s = add_avx(&p11, &ss);
        _mm256_storeu_si256((&mut state[4..8]).as_mut_ptr().cast::<__m256i>(), s);
        let s = add_avx(&p12, &ss);
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
        let r0 = apply_m_4_avx(&s0, &state[0..4]);
        let r1 = apply_m_4_avx(&s1, &state[4..8]);
        let r2 = apply_m_4_avx(&s2, &state[8..12]);
        /*
        // Alternative
        for i in (0..SPONGE_WIDTH).step_by(4) {
            apply_m_4(&mut state[i..i + 4]);
        }
        let r0 = _mm256_loadu_si256((&state[0..4]).as_ptr().cast::<__m256i>());
        let r1 = _mm256_loadu_si256((&state[4..8]).as_ptr().cast::<__m256i>());
        let r2 = _mm256_loadu_si256((&state[8..12]).as_ptr().cast::<__m256i>());
        */
        let s3 = add_avx(&r0, &r1);
        let s = add_avx(&r2, &s3);
        let s3 = add_avx(&r0, &s);
        _mm256_storeu_si256((&mut state[0..4]).as_mut_ptr().cast::<__m256i>(), s3);
        let s3 = add_avx(&r1, &s);
        _mm256_storeu_si256((&mut state[4..8]).as_mut_ptr().cast::<__m256i>(), s3);
        let s3 = add_avx(&r2, &s);
        _mm256_storeu_si256((&mut state[8..12]).as_mut_ptr().cast::<__m256i>(), s3);
    }
}