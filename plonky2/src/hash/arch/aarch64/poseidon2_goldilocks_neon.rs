use core::arch::aarch64::*;

/// Code taken and adapted from: https://github.com/0xPolygonHermez/goldilocks/blob/master/src/goldilocks_base_field_avx.hpp
use crate::hash::{
    hash_types::RichField, poseidon2::RC12, poseidon2::SPONGE_WIDTH,
};

use super::goldilocks_neon::add_neon;
use super::goldilocks_neon::add_neon_a_sc;
use super::goldilocks_neon::mult_neon;

#[inline(always)]
pub fn add_rc_neon<F>(state: &mut [F; SPONGE_WIDTH], rc: &[u64; SPONGE_WIDTH])
where
    F: RichField,
{
    unsafe {
        let s0 = vld1q_s64((&state[0..2]).as_ptr().cast::<i64>());
        let s1 = vld1q_s64((&state[2..4]).as_ptr().cast::<i64>());
        let s2 = vld1q_s64((&state[4..6]).as_ptr().cast::<i64>());
        let s3 = vld1q_s64((&state[6..8]).as_ptr().cast::<i64>());
        let s4 = vld1q_s64((&state[8..10]).as_ptr().cast::<i64>());
        let s5 = vld1q_s64((&state[10..12]).as_ptr().cast::<i64>());

        let rc0 = vld1q_s64((&rc[0..2]).as_ptr().cast::<i64>());
        let rc1 = vld1q_s64((&rc[2..4]).as_ptr().cast::<i64>());
        let rc2 = vld1q_s64((&rc[4..6]).as_ptr().cast::<i64>());
        let rc3 = vld1q_s64((&rc[6..8]).as_ptr().cast::<i64>());
        let rc4 = vld1q_s64((&rc[8..10]).as_ptr().cast::<i64>());
        let rc5 = vld1q_s64((&rc[10..12]).as_ptr().cast::<i64>());

        let ss0 = add_neon_a_sc(&rc0, &s0);
        let ss1 = add_neon_a_sc(&rc1, &s1);
        let ss2 = add_neon_a_sc(&rc2, &s2);
        let ss3 = add_neon_a_sc(&rc3, &s3);
        let ss4 = add_neon_a_sc(&rc4, &s4);
        let ss5 = add_neon_a_sc(&rc5, &s5);

        vst1q_s64((&mut state[0..2]).as_mut_ptr().cast::<i64>(), ss0);
        vst1q_s64((&mut state[2..4]).as_mut_ptr().cast::<i64>(), ss1);
        vst1q_s64((&mut state[4..6]).as_mut_ptr().cast::<i64>(), ss2);
        vst1q_s64((&mut state[6..8]).as_mut_ptr().cast::<i64>(), ss3);
        vst1q_s64((&mut state[8..10]).as_mut_ptr().cast::<i64>(), ss4);
        vst1q_s64((&mut state[10..12]).as_mut_ptr().cast::<i64>(), ss5);
    }
}

#[inline]
fn sbox_p<F>(input: &F) -> F
where
    F: RichField,
{
    let x2 = (*input) * (*input);
    let x4 = x2 * x2;
    let x3 = x2 * (*input);
    x3 * x4
}

#[inline(always)]
fn apply_m_4_neon<F>(x1: &int64x2_t, x2: &int64x2_t, s: &[F]) -> (int64x2_t, int64x2_t)
where
    F: RichField,
{
    // This is based on apply_m_4, but we pack 4 and then 2 operands per operation
    unsafe {
        let y1 = vdupq_n_s64(s[1].to_canonical_u64() as i64);
        let y3 = vdupq_n_s64(s[3].to_canonical_u64() as i64);
        let t1 = add_neon(&x1, &y1);
        let t2 = add_neon(&x2, &y3);
        let mut tt1: [i64; 2] = [0; 2];
        let mut tt2: [i64; 2] = [0; 2];
        vst1q_s64((&mut tt1).as_mut_ptr(), t1);
        vst1q_s64((&mut tt2).as_mut_ptr(), t2);
        let tmp = tt1[0];
        tt1[0] = 0;
        tt1[1] = tt2[0];
        tt2[0] = 0;
        tt2[1] = tmp;
        let y1 = vld1q_s64((&tt1).as_ptr());
        let y2 = vld1q_s64((&tt2).as_ptr());
        let t1 = add_neon(&t1, &y1);
        let t2 = add_neon(&t2, &y2);
        vst1q_s64((&mut tt1).as_mut_ptr(), t1);
        vst1q_s64((&mut tt2).as_mut_ptr(), t2);
        let tmp = tt1[1];
        tt1[1] = tt2[0];
        tt2[0] = tmp;
        let y1 = vld1q_s64((&tt1).as_ptr());
        let y2 = vld1q_s64((&tt2).as_ptr());
        let v1 = add_neon(&t1, &t1);
        let v2 = add_neon(&v1, &v1);
        let v3 = add_neon(&v2, &y2);    // t4, t5
        let v4 = add_neon(&v3, &y2);    // t6, t7
        vst1q_s64((&mut tt1).as_mut_ptr(), v3); // tt10 = t5, tt11 = t4
        vst1q_s64((&mut tt2).as_mut_ptr(), v4); // tt20 = t7, tt21 = t6
        let t5 = tt1[0];
        tt1[0] = tt2[0];
        tt2[0] = tt2[1];
        tt2[1] = t5;
        let x1 = vld1q_s64((&tt2).as_ptr());
        let x2 = vld1q_s64((&tt1).as_ptr());
        (x1, x2)
    }
}

#[inline(always)]
pub fn matmul_internal_neon<F>(
    state: &mut [F; SPONGE_WIDTH],
    mat_internal_diag_m_1: [u64; SPONGE_WIDTH],
) where
    F: RichField,
{
    /*
    let mut sum = state[0];
    for i in 1..SPONGE_WIDTH {
        sum = sum + state[i];
    }
    let si64: i64 = sum.to_canonical_u64() as i64;
    */
    unsafe {
        // let ss = _mm256_set_epi64x(si64, si64, si64, si64);
        let s0 = _mm256_loadu_si256((&state[0..4]).as_ptr().cast::<__m256i>());
        let s1 = _mm256_loadu_si256((&state[4..8]).as_ptr().cast::<__m256i>());
        let s2 = _mm256_loadu_si256((&state[8..12]).as_ptr().cast::<__m256i>());
        let ss0 = add_neon(&s0, &s1);
        let ss1 = add_neon(&s2, &ss0);
        let ss2 = _mm256_permute4x64_epi64(ss1, 0x93); // [0, 1, 2, 3] -> [3, 0, 1, 2]
        let ss0 = add_neon(&ss1, &ss2);
        let ss1 = _mm256_permute4x64_epi64(ss2, 0x93); // [0, 1, 2, 3] -> [3, 0, 1, 2]
        let ss2 = add_neon(&ss0, &ss1);
        let ss0 = _mm256_permute4x64_epi64(ss1, 0x93); // [0, 1, 2, 3] -> [3, 0, 1, 2]
        let ss = add_neon(&ss0, &ss2);
        let m0 = _mm256_loadu_si256((&mat_internal_diag_m_1[0..4]).as_ptr().cast::<__m256i>());
        let m1 = _mm256_loadu_si256((&mat_internal_diag_m_1[4..8]).as_ptr().cast::<__m256i>());
        let m2 = _mm256_loadu_si256((&mat_internal_diag_m_1[8..12]).as_ptr().cast::<__m256i>());
        let p10 = mult_neon(&s0, &m0);
        let p11 = mult_neon(&s1, &m1);
        let p12 = mult_neon(&s2, &m2);
        let s = add_neon(&p10, &ss);
        _mm256_storeu_si256((&mut state[0..4]).as_mut_ptr().cast::<__m256i>(), s);
        let s = add_neon(&p11, &ss);
        _mm256_storeu_si256((&mut state[4..8]).as_mut_ptr().cast::<__m256i>(), s);
        let s = add_neon(&p12, &ss);
        _mm256_storeu_si256((&mut state[8..12]).as_mut_ptr().cast::<__m256i>(), s);
    }
}

#[inline(always)]
pub fn permute_mut_neon<F>(state: &mut [F; SPONGE_WIDTH])
where
    F: RichField,
{
    unsafe {
        let s0 = vld1q_s64((&state[0..2]).as_ptr().cast::<i64>());
        let s1 = vld1q_s64((&state[2..4]).as_ptr().cast::<i64>());
        let s2 = vld1q_s64((&state[4..6]).as_ptr().cast::<i64>());
        let s3 = vld1q_s64((&state[6..8]).as_ptr().cast::<i64>());
        let s4 = vld1q_s64((&state[8..10]).as_ptr().cast::<i64>());
        let s5 = vld1q_s64((&state[10..12]).as_ptr().cast::<i64>());
        // apply m4
        let (r0, r1) = apply_m_4_neon(&s0, &s1, &state[0..4]);
        let (r2, r3) = apply_m_4_neon(&s2, &s3, &state[4..8]);
        let (r4, r5) = apply_m_4_neon(&s4, &s5, &state[8..12]);
        
        let t0 = add_neon(&r0, &r2);
        let t1 = add_neon(&r1, &r3);
        let t0 = add_neon(&t0, &r4);
        let t1 = add_neon(&t1, &r5);

        let r0 = add_neon(&r0, &t0);
        let r1 = add_neon(&r1, &t1);
        let r2 = add_neon(&r2, &t0);
        let r3 = add_neon(&r3, &t1);
        let r4 = add_neon(&r4, &t0);
        let r5 = add_neon(&r5, &t1);
        
        vst1q_s64((&mut state[0..2]).as_mut_ptr().cast::<i64>(), r0);
        vst1q_s64((&mut state[2..4]).as_mut_ptr().cast::<i64>(), r1);
        vst1q_s64((&mut state[4..6]).as_mut_ptr().cast::<i64>(), r2);
        vst1q_s64((&mut state[6..8]).as_mut_ptr().cast::<i64>(), r3);
        vst1q_s64((&mut state[8..10]).as_mut_ptr().cast::<i64>(), r4);
        vst1q_s64((&mut state[10..12]).as_mut_ptr().cast::<i64>(), r5);
    }
}

#[inline(always)]
pub fn internal_layer_neon<F>(
    state: &mut [F; SPONGE_WIDTH],
    mat_internal_diag_m_1: [u64; SPONGE_WIDTH],
    r_beg: usize,
    r_end: usize,
) where
    F: RichField,
{
    unsafe {
        // The internal rounds.
        // let mut s0 = _mm256_loadu_si256((&state[0..4]).as_ptr().cast::<__m256i>());
        let mut s1 = _mm256_loadu_si256((&state[4..8]).as_ptr().cast::<__m256i>());
        let mut s2 = _mm256_loadu_si256((&state[8..12]).as_ptr().cast::<__m256i>());

        let m0 = _mm256_loadu_si256((&mat_internal_diag_m_1[0..4]).as_ptr().cast::<__m256i>());
        let m1 = _mm256_loadu_si256((&mat_internal_diag_m_1[4..8]).as_ptr().cast::<__m256i>());
        let m2 = _mm256_loadu_si256((&mat_internal_diag_m_1[8..12]).as_ptr().cast::<__m256i>());

        // let mut sv: [F; 4] = [F::ZERO; 4];

        for r in r_beg..r_end {
            state[0] += F::from_canonical_u64(RC12[r][0]);
            state[0] = sbox_p(&state[0]);
            let mut s0 = _mm256_loadu_si256((&state[0..4]).as_ptr().cast::<__m256i>());
            /*
            // state[0] = state[0] + RC12[r][0]
            let rc = _mm256_set_epi64x(0, 0, 0, RC12[r][0] as i64);
            s0 = add_neon(&s0, &rc);
            // state[0] = sbox(state[0])
            _mm256_storeu_si256((&mut sv).as_mut_ptr().cast::<__m256i>(), s0);
            sv[0] = sbox_p(&sv[0]);
            s0 = _mm256_loadu_si256((&sv).as_ptr().cast::<__m256i>());
            */
            // mat mul
            let ss0 = add_neon(&s0, &s1);
            let ss1 = add_neon(&s2, &ss0);
            let ss2 = _mm256_permute4x64_epi64(ss1, 0x93); // [0, 1, 2, 3] -> [3, 0, 1, 2]
            let ss0 = add_neon(&ss1, &ss2);
            let ss1 = _mm256_permute4x64_epi64(ss2, 0x93); // [0, 1, 2, 3] -> [3, 0, 1, 2]
            let ss2 = add_neon(&ss0, &ss1);
            let ss0 = _mm256_permute4x64_epi64(ss1, 0x93); // [0, 1, 2, 3] -> [3, 0, 1, 2]
            let ss = add_neon(&ss0, &ss2);
            let p10 = mult_neon(&s0, &m0);
            let p11 = mult_neon(&s1, &m1);
            let p12 = mult_neon(&s2, &m2);
            s0 = add_neon(&p10, &ss);
            s1 = add_neon(&p11, &ss);
            s2 = add_neon(&p12, &ss);
            _mm256_storeu_si256((&mut state[0..4]).as_mut_ptr().cast::<__m256i>(), s0);
        }
        // _mm256_storeu_si256((&mut state[0..4]).as_mut_ptr().cast::<__m256i>(), s0);
        _mm256_storeu_si256((&mut state[4..8]).as_mut_ptr().cast::<__m256i>(), s1);
        _mm256_storeu_si256((&mut state[8..12]).as_mut_ptr().cast::<__m256i>(), s2);
    }
}
