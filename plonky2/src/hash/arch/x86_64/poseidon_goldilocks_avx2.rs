use core::arch::x86_64::*;

use unroll::unroll_for_loops;

use crate::field::types::PrimeField64;
use crate::hash::arch::x86_64::goldilocks_avx2::{
    add_avx, mult_avx, reduce_avx_128_64, sbox_avx_m256i,
};
use crate::hash::poseidon::{
    Poseidon, ALL_ROUND_CONSTANTS, HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS, N_ROUNDS, SPONGE_WIDTH,
};

#[allow(dead_code)]
const MDS_MATRIX_CIRC: [u64; 12] = [17, 15, 41, 16, 2, 28, 13, 13, 39, 18, 34, 20];

#[allow(dead_code)]
const MDS_MATRIX_DIAG: [u64; 12] = [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

const FAST_PARTIAL_FIRST_ROUND_CONSTANT: [u64; 12] = [
    0x3cc3f892184df408,
    0xe993fd841e7e97f1,
    0xf2831d3575f0f3af,
    0xd2500e0a350994ca,
    0xc5571f35d7288633,
    0x91d89c5184109a02,
    0xf37f925d04e5667b,
    0x2d6e448371955a69,
    0x740ef19ce01398a1,
    0x694d24c0752fdf45,
    0x60936af96ee2f148,
    0xc33448feadc78f0c,
];

const FAST_PARTIAL_ROUND_CONSTANTS: [u64; N_PARTIAL_ROUNDS] = [
    0x74cb2e819ae421ab,
    0xd2559d2370e7f663,
    0x62bf78acf843d17c,
    0xd5ab7b67e14d1fb4,
    0xb9fe2ae6e0969bdc,
    0xe33fdf79f92a10e8,
    0x0ea2bb4c2b25989b,
    0xca9121fbf9d38f06,
    0xbdd9b0aa81f58fa4,
    0x83079fa4ecf20d7e,
    0x650b838edfcc4ad3,
    0x77180c88583c76ac,
    0xaf8c20753143a180,
    0xb8ccfe9989a39175,
    0x954a1729f60cc9c5,
    0xdeb5b550c4dca53b,
    0xf01bb0b00f77011e,
    0xa1ebb404b676afd9,
    0x860b6e1597a0173e,
    0x308bb65a036acbce,
    0x1aca78f31c97c876,
    0x0,
];

const FAST_PARTIAL_ROUND_INITIAL_MATRIX: [[u64; 12]; 12] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        0,
        0x80772dc2645b280b,
        0xdc927721da922cf8,
        0xc1978156516879ad,
        0x90e80c591f48b603,
        0x3a2432625475e3ae,
        0x00a2d4321cca94fe,
        0x77736f524010c932,
        0x904d3f2804a36c54,
        0xbf9b39e28a16f354,
        0x3a1ded54a6cd058b,
        0x42392870da5737cf,
    ],
    [
        0,
        0xe796d293a47a64cb,
        0xb124c33152a2421a,
        0x0ee5dc0ce131268a,
        0xa9032a52f930fae6,
        0x7e33ca8c814280de,
        0xad11180f69a8c29e,
        0xc75ac6d5b5a10ff3,
        0xf0674a8dc5a387ec,
        0xb36d43120eaa5e2b,
        0x6f232aab4b533a25,
        0x3a1ded54a6cd058b,
    ],
    [
        0,
        0xdcedab70f40718ba,
        0x14a4a64da0b2668f,
        0x4715b8e5ab34653b,
        0x1e8916a99c93a88e,
        0xbba4b5d86b9a3b2c,
        0xe76649f9bd5d5c2e,
        0xaf8e2518a1ece54d,
        0xdcda1344cdca873f,
        0xcd080204256088e5,
        0xb36d43120eaa5e2b,
        0xbf9b39e28a16f354,
    ],
    [
        0,
        0xf4a437f2888ae909,
        0xc537d44dc2875403,
        0x7f68007619fd8ba9,
        0xa4911db6a32612da,
        0x2f7e9aade3fdaec1,
        0xe7ffd578da4ea43d,
        0x43a608e7afa6b5c2,
        0xca46546aa99e1575,
        0xdcda1344cdca873f,
        0xf0674a8dc5a387ec,
        0x904d3f2804a36c54,
    ],
    [
        0,
        0xf97abba0dffb6c50,
        0x5e40f0c9bb82aab5,
        0x5996a80497e24a6b,
        0x07084430a7307c9a,
        0xad2f570a5b8545aa,
        0xab7f81fef4274770,
        0xcb81f535cf98c9e9,
        0x43a608e7afa6b5c2,
        0xaf8e2518a1ece54d,
        0xc75ac6d5b5a10ff3,
        0x77736f524010c932,
    ],
    [
        0,
        0x7f8e41e0b0a6cdff,
        0x4b1ba8d40afca97d,
        0x623708f28fca70e8,
        0xbf150dc4914d380f,
        0xc26a083554767106,
        0x753b8b1126665c22,
        0xab7f81fef4274770,
        0xe7ffd578da4ea43d,
        0xe76649f9bd5d5c2e,
        0xad11180f69a8c29e,
        0x00a2d4321cca94fe,
    ],
    [
        0,
        0x726af914971c1374,
        0x1d7f8a2cce1a9d00,
        0x18737784700c75cd,
        0x7fb45d605dd82838,
        0x862361aeab0f9b6e,
        0xc26a083554767106,
        0xad2f570a5b8545aa,
        0x2f7e9aade3fdaec1,
        0xbba4b5d86b9a3b2c,
        0x7e33ca8c814280de,
        0x3a2432625475e3ae,
    ],
    [
        0,
        0x64dd936da878404d,
        0x4db9a2ead2bd7262,
        0xbe2e19f6d07f1a83,
        0x02290fe23c20351a,
        0x7fb45d605dd82838,
        0xbf150dc4914d380f,
        0x07084430a7307c9a,
        0xa4911db6a32612da,
        0x1e8916a99c93a88e,
        0xa9032a52f930fae6,
        0x90e80c591f48b603,
    ],
    [
        0,
        0x85418a9fef8a9890,
        0xd8a2eb7ef5e707ad,
        0xbfe85ababed2d882,
        0xbe2e19f6d07f1a83,
        0x18737784700c75cd,
        0x623708f28fca70e8,
        0x5996a80497e24a6b,
        0x7f68007619fd8ba9,
        0x4715b8e5ab34653b,
        0x0ee5dc0ce131268a,
        0xc1978156516879ad,
    ],
    [
        0,
        0x156048ee7a738154,
        0x91f7562377e81df5,
        0xd8a2eb7ef5e707ad,
        0x4db9a2ead2bd7262,
        0x1d7f8a2cce1a9d00,
        0x4b1ba8d40afca97d,
        0x5e40f0c9bb82aab5,
        0xc537d44dc2875403,
        0x14a4a64da0b2668f,
        0xb124c33152a2421a,
        0xdc927721da922cf8,
    ],
    [
        0,
        0xd841e8ef9dde8ba0,
        0x156048ee7a738154,
        0x85418a9fef8a9890,
        0x64dd936da878404d,
        0x726af914971c1374,
        0x7f8e41e0b0a6cdff,
        0xf97abba0dffb6c50,
        0xf4a437f2888ae909,
        0xdcedab70f40718ba,
        0xe796d293a47a64cb,
        0x80772dc2645b280b,
    ],
];

#[allow(dead_code)]
#[inline(always)]
#[unroll_for_loops]
fn mds_row_shf(r: usize, v: &[u64; SPONGE_WIDTH]) -> (u64, u64) {
    let mut res = 0u128;

    // This is a hacky way of fully unrolling the loop.
    for i in 0..12 {
        if i < SPONGE_WIDTH {
            res += (v[(i + r) % SPONGE_WIDTH] as u128) * (MDS_MATRIX_CIRC[i] as u128);
        }
    }
    res += (v[r] as u128) * (MDS_MATRIX_DIAG[r] as u128);

    ((res >> 64) as u64, res as u64)
}

#[allow(dead_code)]
#[inline(always)]
#[unroll_for_loops]
unsafe fn mds_layer_avx_v3<F>(state: &mut [F; SPONGE_WIDTH])
where
    F: PrimeField64 + Poseidon,
{
    let mut v: [u64; 24] = [0; 24];
    for i in 0..12 {
        v[i] = state[i].to_canonical_u64();
        v[i+12] = v[i];
    }

    let mut r0l = _mm256_set_epi64x(((v[0] & 0xFFFFFFFF) * MDS_MATRIX_DIAG[0]) as i64, 0, 0, 0);
    let mut r1l = _mm256_set_epi64x(0, 0, 0, 0);
    let mut r2l = r1l;
    let mut r0h = _mm256_set_epi64x(((v[0] >> 32) * MDS_MATRIX_DIAG[0]) as i64, 0, 0, 0);
    let mut r1h = r1l;
    let mut r2h = r2l;

    for i in 0..12 {
        let vv0 = _mm256_loadu_si256((&v[i+0..i+4]).as_ptr().cast::<__m256i>());
        let vv1 = _mm256_loadu_si256((&v[i+4..i+8]).as_ptr().cast::<__m256i>());
        let vv2 = _mm256_loadu_si256((&v[i+8..i+12]).as_ptr().cast::<__m256i>());
        let m = _mm256_set_epi64x(MDS_MATRIX_CIRC[i] as i64, MDS_MATRIX_CIRC[i] as i64, MDS_MATRIX_CIRC[i] as i64, MDS_MATRIX_CIRC[i] as i64);
        let c0l = _mm256_mul_epu32(vv0, m);
        let c1l = _mm256_mul_epu32(vv1, m);
        let c2l = _mm256_mul_epu32(vv2, m);
        let c0h = _mm256_mul_epu32(_mm256_srli_epi64(vv0, 32), m);
        let c1h = _mm256_mul_epu32(_mm256_srli_epi64(vv1, 32), m);
        let c2h = _mm256_mul_epu32(_mm256_srli_epi64(vv2, 32), m);
        r0l = _mm256_add_epi64(r0l, c0l);
        r1l = _mm256_add_epi64(r1l, c1l);
        r2l = _mm256_add_epi64(r2l, c2l);
        r0h = _mm256_add_epi64(r0h, c0h);
        r1h = _mm256_add_epi64(r1h, c1h);
        r2h = _mm256_add_epi64(r2h, c2h);
    }
    let r0 = reduce_avx_128_64(&r0h, &r0l);
    let r1 = reduce_avx_128_64(&r1h, &r1l);
    let r2 = reduce_avx_128_64(&r2h, &r2l);

    _mm256_storeu_si256((&mut state[0..4]).as_mut_ptr().cast::<__m256i>(), r0);
    _mm256_storeu_si256((&mut state[4..8]).as_mut_ptr().cast::<__m256i>(), r1);
    _mm256_storeu_si256((&mut state[8..12]).as_mut_ptr().cast::<__m256i>(), r2);
}

#[allow(dead_code)]
#[inline(always)]
#[unroll_for_loops]
unsafe fn mds_layer_avx(s0: &__m256i, s1: &__m256i, s2: &__m256i) -> (__m256i, __m256i, __m256i) {
    let mut st64 = [0u64; SPONGE_WIDTH];

    _mm256_storeu_si256((&mut st64[0..4]).as_mut_ptr().cast::<__m256i>(), *s0);
    _mm256_storeu_si256((&mut st64[4..8]).as_mut_ptr().cast::<__m256i>(), *s1);
    _mm256_storeu_si256((&mut st64[8..12]).as_mut_ptr().cast::<__m256i>(), *s2);

    let mut sumh: [u64; 12] = [0; 12];
    let mut suml: [u64; 12] = [0; 12];
    for r in 0..12 {
        if r < SPONGE_WIDTH {
            (sumh[r], suml[r]) = mds_row_shf(r, &st64);
        }
    }

    let ss0h = _mm256_loadu_si256((&sumh[0..4]).as_ptr().cast::<__m256i>());
    let ss0l = _mm256_loadu_si256((&suml[0..4]).as_ptr().cast::<__m256i>());
    let ss1h = _mm256_loadu_si256((&sumh[4..8]).as_ptr().cast::<__m256i>());
    let ss1l = _mm256_loadu_si256((&suml[4..8]).as_ptr().cast::<__m256i>());
    let ss2h = _mm256_loadu_si256((&sumh[8..12]).as_ptr().cast::<__m256i>());
    let ss2l = _mm256_loadu_si256((&suml[8..12]).as_ptr().cast::<__m256i>());
    let r0 = reduce_avx_128_64(&ss0h, &ss0l);
    let r1 = reduce_avx_128_64(&ss1h, &ss1l);
    let r2 = reduce_avx_128_64(&ss2h, &ss2l);

    (r0, r1, r2)
}

#[allow(dead_code)]
#[inline(always)]
#[unroll_for_loops]
unsafe fn mds_layer_avx_v2<F>(
    s0: &__m256i,
    s1: &__m256i,
    s2: &__m256i,
) -> (__m256i, __m256i, __m256i)
where
    F: PrimeField64,
{
    let mut st64 = [0u64; SPONGE_WIDTH];

    _mm256_storeu_si256((&mut st64[0..4]).as_mut_ptr().cast::<__m256i>(), *s0);
    _mm256_storeu_si256((&mut st64[4..8]).as_mut_ptr().cast::<__m256i>(), *s1);
    _mm256_storeu_si256((&mut st64[8..12]).as_mut_ptr().cast::<__m256i>(), *s2);

    let mut result = [F::ZERO; SPONGE_WIDTH];
    // This is a hacky way of fully unrolling the loop.
    for r in 0..12 {
        if r < SPONGE_WIDTH {
            let (sum_hi, sum_lo) = mds_row_shf(r, &st64);
            result[r] = F::from_noncanonical_u96((sum_lo, sum_hi.try_into().unwrap()));
        }
    }

    let r0 = _mm256_loadu_si256((&result[0..4]).as_ptr().cast::<__m256i>());
    let r1 = _mm256_loadu_si256((&result[4..8]).as_ptr().cast::<__m256i>());
    let r2 = _mm256_loadu_si256((&result[8..12]).as_ptr().cast::<__m256i>());

    (r0, r1, r2)
}

#[inline(always)]
#[unroll_for_loops]
fn mds_partial_layer_init_avx<F>(state: &mut [F; SPONGE_WIDTH])
where
    F: PrimeField64,
{
    let mut result = [F::ZERO; SPONGE_WIDTH];
    let res0 = state[0];
    unsafe {
        let mut r0 = _mm256_loadu_si256((&mut result[0..4]).as_mut_ptr().cast::<__m256i>());
        let mut r1 = _mm256_loadu_si256((&mut result[0..4]).as_mut_ptr().cast::<__m256i>());
        let mut r2 = _mm256_loadu_si256((&mut result[0..4]).as_mut_ptr().cast::<__m256i>());
        for r in 1..12 {
            let sr = _mm256_set_epi64x(
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
            );
            let t0 = _mm256_loadu_si256(
                (&FAST_PARTIAL_ROUND_INITIAL_MATRIX[r][0..4])
                    .as_ptr()
                    .cast::<__m256i>(),
            );
            let t1 = _mm256_loadu_si256(
                (&FAST_PARTIAL_ROUND_INITIAL_MATRIX[r][4..8])
                    .as_ptr()
                    .cast::<__m256i>(),
            );
            let t2 = _mm256_loadu_si256(
                (&FAST_PARTIAL_ROUND_INITIAL_MATRIX[r][8..12])
                    .as_ptr()
                    .cast::<__m256i>(),
            );
            let m0 = mult_avx(&sr, &t0);
            let m1 = mult_avx(&sr, &t1);
            let m2 = mult_avx(&sr, &t2);
            r0 = add_avx(&r0, &m0);
            r1 = add_avx(&r1, &m1);
            r2 = add_avx(&r2, &m2);
        }
        _mm256_storeu_si256((state[0..4]).as_mut_ptr().cast::<__m256i>(), r0);
        _mm256_storeu_si256((state[4..8]).as_mut_ptr().cast::<__m256i>(), r1);
        _mm256_storeu_si256((state[8..12]).as_mut_ptr().cast::<__m256i>(), r2);
        state[0] = res0;
    }
}

#[inline(always)]
#[unroll_for_loops]
fn partial_first_constant_layer_avx<F>(state: &mut [F; SPONGE_WIDTH])
where
    F: PrimeField64,
{
    unsafe {
        let c0 = _mm256_loadu_si256(
            (&FAST_PARTIAL_FIRST_ROUND_CONSTANT[0..4])
                .as_ptr()
                .cast::<__m256i>(),
        );
        let c1 = _mm256_loadu_si256(
            (&FAST_PARTIAL_FIRST_ROUND_CONSTANT[4..8])
                .as_ptr()
                .cast::<__m256i>(),
        );
        let c2 = _mm256_loadu_si256(
            (&FAST_PARTIAL_FIRST_ROUND_CONSTANT[8..12])
                .as_ptr()
                .cast::<__m256i>(),
        );

        let mut s0 = _mm256_loadu_si256((state[0..4]).as_ptr().cast::<__m256i>());
        let mut s1 = _mm256_loadu_si256((state[4..8]).as_ptr().cast::<__m256i>());
        let mut s2 = _mm256_loadu_si256((state[8..12]).as_ptr().cast::<__m256i>());
        s0 = add_avx(&s0, &c0);
        s1 = add_avx(&s1, &c1);
        s2 = add_avx(&s2, &c2);
        _mm256_storeu_si256((state[0..4]).as_mut_ptr().cast::<__m256i>(), s0);
        _mm256_storeu_si256((state[4..8]).as_mut_ptr().cast::<__m256i>(), s1);
        _mm256_storeu_si256((state[8..12]).as_mut_ptr().cast::<__m256i>(), s2);
    }
}

#[inline(always)]
fn sbox_monomial<F>(x: F) -> F
where
    F: PrimeField64,
{
    // x |--> x^7
    let x2 = x.square();
    let x4 = x2.square();
    let x3 = x * x2;
    x3 * x4
}

pub fn poseidon_avx<F>(input: &[F; SPONGE_WIDTH]) -> [F; SPONGE_WIDTH]
where
    F: PrimeField64 + Poseidon,
{
    let mut state = &mut input.clone();
    let mut round_ctr = 0;

    unsafe {
        // Self::full_rounds(&mut state, &mut round_ctr);
        for _ in 0..HALF_N_FULL_ROUNDS {
            // load state
            let s0 = _mm256_loadu_si256((&state[0..4]).as_ptr().cast::<__m256i>());
            let s1 = _mm256_loadu_si256((&state[4..8]).as_ptr().cast::<__m256i>());
            let s2 = _mm256_loadu_si256((&state[8..12]).as_ptr().cast::<__m256i>());

            let rc: &[u64; 12] = &ALL_ROUND_CONSTANTS[SPONGE_WIDTH * round_ctr..][..SPONGE_WIDTH]
                .try_into()
                .unwrap();
            let rc0 = _mm256_loadu_si256((&rc[0..4]).as_ptr().cast::<__m256i>());
            let rc1 = _mm256_loadu_si256((&rc[4..8]).as_ptr().cast::<__m256i>());
            let rc2 = _mm256_loadu_si256((&rc[8..12]).as_ptr().cast::<__m256i>());
            let ss0 = add_avx(&s0, &rc0);
            let ss1 = add_avx(&s1, &rc1);
            let ss2 = add_avx(&s2, &rc2);
            let (r0, r1, r2) = sbox_avx_m256i(&ss0, &ss1, &ss2);
            // let (s0, s1, s2) = mds_layer_avx(&r0, &r1, &r2);
            // let (s0, s1, s2) = mds_layer_avx_v2::<F>(&r0, &r1, &r2);

            // store state
            _mm256_storeu_si256((state[0..4]).as_mut_ptr().cast::<__m256i>(), r0);
            _mm256_storeu_si256((state[4..8]).as_mut_ptr().cast::<__m256i>(), r1);
            _mm256_storeu_si256((state[8..12]).as_mut_ptr().cast::<__m256i>(), r2);

            // mds_layer_avx_v3(&mut state);
            *state = <F as Poseidon>::mds_layer(&state);
            // mds_layer_avx::<F>(&mut s0, &mut s1, &mut s2);
            round_ctr += 1;
        }

        // Self::partial_rounds(&mut state, &mut round_ctr);
        partial_first_constant_layer_avx(&mut state);
        mds_partial_layer_init_avx(&mut state);

        for i in 0..N_PARTIAL_ROUNDS {
            state[0] = sbox_monomial(state[0]);
            state[0] = state[0].add_canonical_u64(FAST_PARTIAL_ROUND_CONSTANTS[i]);
            *state = <F as Poseidon>::mds_partial_layer_fast(&state, i);
        }
        round_ctr += N_PARTIAL_ROUNDS;

        // Self::full_rounds(&mut state, &mut round_ctr);
        for _ in 0..HALF_N_FULL_ROUNDS {
            // load state
            let s0 = _mm256_loadu_si256((&state[0..4]).as_ptr().cast::<__m256i>());
            let s1 = _mm256_loadu_si256((&state[4..8]).as_ptr().cast::<__m256i>());
            let s2 = _mm256_loadu_si256((&state[8..12]).as_ptr().cast::<__m256i>());

            let rc: &[u64; 12] = &ALL_ROUND_CONSTANTS[SPONGE_WIDTH * round_ctr..][..SPONGE_WIDTH]
                .try_into()
                .unwrap();
            let rc0 = _mm256_loadu_si256((&rc[0..4]).as_ptr().cast::<__m256i>());
            let rc1 = _mm256_loadu_si256((&rc[4..8]).as_ptr().cast::<__m256i>());
            let rc2 = _mm256_loadu_si256((&rc[8..12]).as_ptr().cast::<__m256i>());
            let ss0 = add_avx(&s0, &rc0);
            let ss1 = add_avx(&s1, &rc1);
            let ss2 = add_avx(&s2, &rc2);
            let (r0, r1, r2) = sbox_avx_m256i(&ss0, &ss1, &ss2);
            // let (s0, s1, s2) = mds_layer_avx(&r0, &r1, &r2);
            // let (s0, s1, s2) = mds_layer_avx_v2::<F>(&r0, &r1, &r2);

            // store state
            _mm256_storeu_si256((state[0..4]).as_mut_ptr().cast::<__m256i>(), r0);
            _mm256_storeu_si256((state[4..8]).as_mut_ptr().cast::<__m256i>(), r1);
            _mm256_storeu_si256((state[8..12]).as_mut_ptr().cast::<__m256i>(), r2);

            // mds_layer_avx_v3(&mut state);
            *state = <F as Poseidon>::mds_layer(&state);
            // mds_layer_avx::<F>(&mut s0, &mut s1, &mut s2);
            round_ctr += 1;
        }

        debug_assert_eq!(round_ctr, N_ROUNDS);
    };
    *state
}
