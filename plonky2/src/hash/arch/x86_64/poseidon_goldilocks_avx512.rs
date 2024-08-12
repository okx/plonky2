use core::arch::x86_64::*;

use unroll::unroll_for_loops;

use crate::field::types::PrimeField64;
use crate::hash::arch::x86_64::goldilocks_avx512::*;
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

#[inline(always)]
#[unroll_for_loops]
fn mds_partial_layer_init_avx<F>(state: &mut [F; SPONGE_WIDTH])
where
    F: PrimeField64,
{
    let mut result = [F::ZERO; SPONGE_WIDTH];
    let res0 = state[0];
    unsafe {
        let mut r0 = _mm512_loadu_si512((&mut result[0..8]).as_mut_ptr().cast::<i32>());
        let mut r1 = _mm512_loadu_si512((&mut result[4..12]).as_mut_ptr().cast::<i32>());

        for r in 1..12 {
            let sr512 = _mm512_set_epi64(
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
            );
            let t0 = _mm512_loadu_si512(
                (&FAST_PARTIAL_ROUND_INITIAL_MATRIX[r][0..8])
                    .as_ptr()
                    .cast::<i32>(),
            );
            let t1 = _mm512_loadu_si512(
                (&FAST_PARTIAL_ROUND_INITIAL_MATRIX[r][4..12])
                    .as_ptr()
                    .cast::<i32>(),
            );
            let m0 = mult_avx512(&sr512, &t0);
            let m1 = mult_avx512(&sr512, &t1);
            r0 = add_avx512_b_c(&r0, &m0);
            r1 = add_avx512_b_c(&r1, &m1);
        }
        _mm512_storeu_si512((state[0..8]).as_mut_ptr().cast::<i32>(), r0);
        _mm512_storeu_si512((state[4..12]).as_mut_ptr().cast::<i32>(), r1);
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
        let c0 = _mm512_loadu_si512(
            (&FAST_PARTIAL_FIRST_ROUND_CONSTANT[0..8])
                .as_ptr()
                .cast::<i32>(),
        );
        let c1 = _mm512_loadu_si512(
            (&FAST_PARTIAL_FIRST_ROUND_CONSTANT[4..12])
                .as_ptr()
                .cast::<i32>(),
        );
        let mut s0 = _mm512_loadu_si512((state[0..8]).as_ptr().cast::<i32>());
        let mut s1 = _mm512_loadu_si512((state[4..12]).as_ptr().cast::<i32>());
        s0 = add_avx512_b_c(&s0, &c0);
        s1 = add_avx512_b_c(&s1, &c1);
        _mm512_storeu_si512((state[0..8]).as_mut_ptr().cast::<i32>(), s0);
        _mm512_storeu_si512((state[4..12]).as_mut_ptr().cast::<i32>(), s1);
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

pub fn poseidon_avx512<F>(input: &[F; SPONGE_WIDTH]) -> [F; SPONGE_WIDTH]
where
    F: PrimeField64 + Poseidon,
{
    let mut state = &mut input.clone();
    let mut round_ctr = 0;

    unsafe {
        // Self::full_rounds(&mut state, &mut round_ctr);
        for _ in 0..HALF_N_FULL_ROUNDS {
            // load state
            let s0 = _mm512_loadu_si512((&state[0..8]).as_ptr().cast::<i32>());
            let s1 = _mm512_loadu_si512((&state[4..12]).as_ptr().cast::<i32>());

            let rc: &[u64; 12] = &ALL_ROUND_CONSTANTS[SPONGE_WIDTH * round_ctr..][..SPONGE_WIDTH]
                .try_into()
                .unwrap();
            let rc0 = _mm512_loadu_si512((&rc[0..8]).as_ptr().cast::<i32>());
            let rc1 = _mm512_loadu_si512((&rc[4..12]).as_ptr().cast::<i32>());
            let ss0 = add_avx512_b_c(&s0, &rc0);
            let ss1 = add_avx512_b_c(&s1, &rc1);
            let r0 = sbox_avx512_one(&ss0);
            let r1 = sbox_avx512_one(&ss1);

            // store state
            _mm512_storeu_si512((state[0..8]).as_mut_ptr().cast::<i32>(), r0);
            _mm512_storeu_si512((state[4..12]).as_mut_ptr().cast::<i32>(), r1);

            *state = <F as Poseidon>::mds_layer(&state);
            round_ctr += 1;
        }
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
            let s0 = _mm512_loadu_si512((&state[0..8]).as_ptr().cast::<i32>());
            let s1 = _mm512_loadu_si512((&state[4..12]).as_ptr().cast::<i32>());

            let rc: &[u64; 12] = &ALL_ROUND_CONSTANTS[SPONGE_WIDTH * round_ctr..][..SPONGE_WIDTH]
                .try_into()
                .unwrap();
            let rc0 = _mm512_loadu_si512((&rc[0..8]).as_ptr().cast::<i32>());
            let rc1 = _mm512_loadu_si512((&rc[4..12]).as_ptr().cast::<i32>());
            let ss0 = add_avx512_b_c(&s0, &rc0);
            let ss1 = add_avx512_b_c(&s1, &rc1);
            let r0 = sbox_avx512_one(&ss0);
            let r1 = sbox_avx512_one(&ss1);

            // store state
            _mm512_storeu_si512((state[0..8]).as_mut_ptr().cast::<i32>(), r0);
            _mm512_storeu_si512((state[4..12]).as_mut_ptr().cast::<i32>(), r1);

            *state = <F as Poseidon>::mds_layer(&state);
            // mds_layer_avx::<F>(&mut s0, &mut s1, &mut s2);
            round_ctr += 1;
        }

        debug_assert_eq!(round_ctr, N_ROUNDS);
    };
    *state
}
