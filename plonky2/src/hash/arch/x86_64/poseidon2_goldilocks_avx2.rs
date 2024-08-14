use core::arch::x86_64::*;

use super::goldilocks_avx2::{add_avx, add_avx_a_sc, mult_avx};
/// Code taken and adapted from: https://github.com/0xPolygonHermez/goldilocks/blob/master/src/goldilocks_base_field_avx.hpp
use crate::hash::{hash_types::RichField, poseidon2::RC12, poseidon2::SPONGE_WIDTH};

#[inline(always)]
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
        // let ss0 = add_avx(&s0, &rc0);
        // let ss1 = add_avx(&s1, &rc1);
        // let ss2 = add_avx(&s2, &rc2);
        let ss0 = add_avx_a_sc(&rc0, &s0);
        let ss1 = add_avx_a_sc(&rc1, &s1);
        let ss2 = add_avx_a_sc(&rc2, &s2);
        _mm256_storeu_si256((&mut state[0..4]).as_mut_ptr().cast::<__m256i>(), ss0);
        _mm256_storeu_si256((&mut state[4..8]).as_mut_ptr().cast::<__m256i>(), ss1);
        _mm256_storeu_si256((&mut state[8..12]).as_mut_ptr().cast::<__m256i>(), ss2);
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
fn apply_m_4_avx<F>(x: &__m256i, s: &[F]) -> __m256i
where
    F: RichField,
{
    // This is based on apply_m_4, but we pack 4 and then 2 operands per operation
    unsafe {
        let y = _mm256_set_epi64x(
            s[3].to_canonical_u64() as i64,
            s[3].to_canonical_u64() as i64,
            s[1].to_canonical_u64() as i64,
            s[1].to_canonical_u64() as i64,
        );
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

#[inline(always)]
pub fn matmul_internal_avx<F>(
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
        let ss0 = add_avx(&s0, &s1);
        let ss1 = add_avx(&s2, &ss0);
        let ss2 = _mm256_permute4x64_epi64(ss1, 0x93); // [0, 1, 2, 3] -> [3, 0, 1, 2]
        let ss0 = add_avx(&ss1, &ss2);
        let ss1 = _mm256_permute4x64_epi64(ss2, 0x93); // [0, 1, 2, 3] -> [3, 0, 1, 2]
        let ss2 = add_avx(&ss0, &ss1);
        let ss0 = _mm256_permute4x64_epi64(ss1, 0x93); // [0, 1, 2, 3] -> [3, 0, 1, 2]
        let ss = add_avx(&ss0, &ss2);
        let m0 = _mm256_loadu_si256((&mat_internal_diag_m_1[0..4]).as_ptr().cast::<__m256i>());
        let m1 = _mm256_loadu_si256((&mat_internal_diag_m_1[4..8]).as_ptr().cast::<__m256i>());
        let m2 = _mm256_loadu_si256((&mat_internal_diag_m_1[8..12]).as_ptr().cast::<__m256i>());
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

#[inline(always)]
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

#[inline(always)]
pub fn internal_layer_avx<F>(
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
            s0 = add_avx(&s0, &rc);
            // state[0] = sbox(state[0])
            _mm256_storeu_si256((&mut sv).as_mut_ptr().cast::<__m256i>(), s0);
            sv[0] = sbox_p(&sv[0]);
            s0 = _mm256_loadu_si256((&sv).as_ptr().cast::<__m256i>());
            */
            // mat mul
            let ss0 = add_avx(&s0, &s1);
            let ss1 = add_avx(&s2, &ss0);
            let ss2 = _mm256_permute4x64_epi64(ss1, 0x93); // [0, 1, 2, 3] -> [3, 0, 1, 2]
            let ss0 = add_avx(&ss1, &ss2);
            let ss1 = _mm256_permute4x64_epi64(ss2, 0x93); // [0, 1, 2, 3] -> [3, 0, 1, 2]
            let ss2 = add_avx(&ss0, &ss1);
            let ss0 = _mm256_permute4x64_epi64(ss1, 0x93); // [0, 1, 2, 3] -> [3, 0, 1, 2]
            let ss = add_avx(&ss0, &ss2);
            let p10 = mult_avx(&s0, &m0);
            let p11 = mult_avx(&s1, &m1);
            let p12 = mult_avx(&s2, &m2);
            s0 = add_avx(&p10, &ss);
            s1 = add_avx(&p11, &ss);
            s2 = add_avx(&p12, &ss);
            _mm256_storeu_si256((&mut state[0..4]).as_mut_ptr().cast::<__m256i>(), s0);
        }
        // _mm256_storeu_si256((&mut state[0..4]).as_mut_ptr().cast::<__m256i>(), s0);
        _mm256_storeu_si256((&mut state[4..8]).as_mut_ptr().cast::<__m256i>(), s1);
        _mm256_storeu_si256((&mut state[8..12]).as_mut_ptr().cast::<__m256i>(), s2);
    }
}
