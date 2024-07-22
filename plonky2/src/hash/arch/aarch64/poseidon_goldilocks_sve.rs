use core::arch::asm;

use crate::field::types::PrimeField64;
use crate::hash::poseidon::{Poseidon, ALL_ROUND_CONSTANTS, HALF_N_FULL_ROUNDS, SPONGE_WIDTH};

#[inline(always)]
// unsafe fn add_sbox<F>(mut state_ptr: *mut F, rc_ptr: *const u64)
unsafe fn add_sbox_sve_256<F>(state: &mut [F; 12], rc: &[u64; 12], idx: usize)
where
    F: PrimeField64 + Poseidon,
{
    let mut state_ptr = (state[idx..idx + 4]).as_mut_ptr();
    let rc_ptr = (rc[idx..idx + 4]).as_ptr();

    let x: F = *state_ptr;

    asm!(
        // add
        "ptrue   p7.b, all",
        "ld1d    z31.d, p7/z, [{p0}]",
        "ld1d    z28.d, p7/z, [{r0}]",
        "mov     z20.d, #4294967295",
        "mov     z27.d, #-4294967295",
        "sub     z27.d, z27.d, z31.d",
        "add     z31.d, z31.d, z28.d",
        "cmphi   p6.d, p7/z, z28.d, z27.d",
        "add     z31.d, p6/m, z31.d, z20.d",    // a (z31.d)
        "mov     z10.d, z31.d", // save a
        // sbox
        // square   -> a^2
        "lsr     z30.d, z31.d, #32",         // a_h
        "and     z31.d, z31.d, #0xFFFFFFFF", // a_l
        "mov     z29.d, z31.d",
        "mul     z29.d, p7/m, z29.d, z31.d",  // c_ll = a_l * a_l
        "lsr     z28.d, z29.d, #33",          // c_ll_h
        "and     z29.d, z29.d, #0x1FFFFFFFF", // c_ll_l
        "mad     z31.d, p7/m, z30.d, z28.d",  // r0
        "lsl     z28.d, z31.d, #33",          // r0_l
        "add     z28.d, p7/m, z28.d, z29.d",  // c_l
        "lsr     z31.d, z31.d, #31",          // r0_h
        "mad     z30.d, p7/m, z30.d, z31.d",  // c_h
        // reduce
        "mov     z31.d, z30.d",
        "and     z31.d, z31.d, #0xFFFFFFFF", // c_hl
        "lsr     z30.d, z30.d, #32",         // c_hh
        "mov     z27.d, #-4294967295",       // GP
        "sub     z29.d, z28.d, z30.d",       // c_l - c_hh
        "cmphs   p6.d, p7/z, z28.d, z30.d",
        "add     z27.d, z27.d, z29.d",
        "sel     z27.d, p6, z29.d, z27.d",   // c1
        "mov     z29.d, #0xFFFFFFFF",        // P_n
        "mov     z30.d, #-4294967295",       // GP
        "sub     z30.d, p7/m, z30.d, z27.d", // GP - c1 (GP-a)
        "mul     z31.d, p7/m, z31.d, z29.d", // c2 (c_hl * P_n)
        "add     z27.d, p7/m, z27.d, z31.d", // c1 + c2
        "cmphi   p6.d, p7/z, z31.d, z30.d",
        "add     z27.d, p6/m, z27.d, z29.d", // + P_n
        "mov     z11.d, z27.d",     // save a^2
        // square   -> a^4
        "mov     z31.d, z27.d",
        "lsr     z30.d, z31.d, #32",         // a_h
        "and     z31.d, z31.d, #0xFFFFFFFF", // a_l
        "mov     z29.d, z31.d",
        "mul     z29.d, p7/m, z29.d, z31.d",  // c_ll = a_l * a_l
        "lsr     z28.d, z29.d, #33",          // c_ll_h
        "and     z29.d, z29.d, #0x1FFFFFFFF", // c_ll_l
        "mad     z31.d, p7/m, z30.d, z28.d",  // r0
        "lsl     z28.d, z31.d, #33",          // r0_l
        "add     z28.d, p7/m, z28.d, z29.d",  // c_l
        "lsr     z31.d, z31.d, #31",          // r0_h
        "mad     z30.d, p7/m, z30.d, z31.d",  // c_h
        // reduce
        "mov    z31.d, z30.d",
        "and     z31.d, z31.d, #0xFFFFFFFF", // c_hl
        "lsr     z30.d, z30.d, #32",         // c_hh
        "mov     z27.d, #-4294967295",       // GP
        "sub     z28.d, z28.d, z30.d",       // c_l - c_hh
        "cmphs   p6.d, p7/z, z28.d, z30.d",
        "add     z27.d, z27.d, z28.d",
        "sel     z27.d, p6, z28.d, z27.d",   // c1
        "mov     z29.d, #0xFFFFFFFF",        // P_n
        "mov     z30.d, #-4294967295",       // GP
        "sub     z30.d, p7/m, z30.d, z27.d", // GP - c1 (GP-a)
        "mul     z31.d, p7/m, z31.d, z29.d", // c2 (c1 + c_hl * P_n)
        "add     z27.d, p7/m, z27.d, z31.d", // c1 + c2
        "cmphi   p6.d, p7/z, z31.d, z30.d",
        "add     z27.d, p6/m, z27.d, z29.d", // + P_n
        "mov     z12.d, z27.d",     // save a^4
        // mul  -> a * a^2
        "mov     z31.d, z11.d",  // a^2
        "mov     z30.d, z10.d",  // a
        "lsr     z29.d, z31.d, #32",         // a_h
        "lsr     z28.d, z30.d, #32",         // b_h
        "and     z31.d, z31.d, #0xFFFFFFFF", // a_l
        "and     z30.d, z30.d, #0xFFFFFFFF", // b_l
        "mov     z24.d, z31.d",              // save a_l
        "mul     z31.d, p7/m, z31.d, z30.d", // c_ll = a_l * b_l
        "lsr     z26.d, z31.d, #32",         // c_ll_h
        "and     z31.d, z31.d, #0xFFFFFFFF",
        "mad     z30.d, p7/m, z29.d, z26.d", // r0 (c_hl)
        "lsr     z26.d, z30.d, #32",         // r0_h
        "and     z30.d, z30.d, #0xFFFFFFFF", // r0_l
        "mad     z29.d, p7/m, z28.d, z26.d", // r2
        "mad     z24.d, p7/m, z28.d, z30.d", // r1
        "lsr     z30.d, z24.d, #32",         // r1_h
        "add     z30.d, p7/m, z30.d, z29.d", // c_h
        "lsl     z25.d, z24.d, #32",         // r1_l
        "orr     z25.d, z25.d, z31.d",       // c_l
        // reduce
        "mov     z29.d, z25.d",
        "mov     z31.d, z30.d",
        "lsr     z30.d, z31.d, #32",         // c_hh
        "and     z31.d, z31.d, #0xFFFFFFFF", // c_hl
        "mov     z27.d, #-4294967295",       // GP
        "sub     z28.d, z29.d, z30.d",       // c_l - c_hh
        "cmphs   p6.d, p7/z, z29.d, z30.d",
        "add     z27.d, z27.d, z28.d",
        "sel     z27.d, p6, z28.d, z27.d",   // c1
        "mov     z29.d, #0xFFFFFFFF",        // P_n
        "mov     z30.d, #-4294967295",       // GP
        "sub     z30.d, p7/m, z30.d, z27.d", // GP - c1 (GP-a)
        "mul     z31.d, p7/m, z31.d, z29.d", // c2 (c1 + c_hl * P_n)
        "add     z27.d, p7/m, z27.d, z31.d", // c1 + c2
        "cmphi   p6.d, p7/z, z31.d, z30.d",
        "add     z27.d, p6/m, z27.d, z29.d", // + P_n
        // mul -> a^3 * a^4
        "mov     z31.d, z27.d",  // a^3
        "mov     z30.d, z12.d",  // a^4
        "lsr     z29.d, z31.d, #32",         // a_h
        "lsr     z28.d, z30.d, #32",         // b_h
        "and     z31.d, z31.d, #0xFFFFFFFF", // a_l
        "and     z30.d, z30.d, #0xFFFFFFFF", // b_l
        "mov     z24.d, z31.d",              // save a_l
        "mul     z31.d, p7/m, z31.d, z30.d", // c_ll = a_l * b_l
        "lsr     z26.d, z31.d, #32",         // c_ll_h
        "and     z31.d, z31.d, #0xFFFFFFFF",
        "mad     z30.d, p7/m, z29.d, z26.d", // r0 (c_hl)
        "lsr     z26.d, z30.d, #32",         // r0_h
        "and     z30.d, z30.d, #0xFFFFFFFF", // r0_l
        "mad     z29.d, p7/m, z28.d, z26.d", // r2
        "mad     z24.d, p7/m, z28.d, z30.d", // r1
        "lsr     z30.d, z24.d, #32",         // r1_h
        "add     z30.d, p7/m, z30.d, z29.d", // c_h
        "lsl     z25.d, z24.d, #32",         // r1_l
        "orr     z25.d, z25.d, z31.d",       // c_l
        // reduce
        "mov     z31.d, z30.d",
        "mov     z29.d, z25.d",
        "lsr     z30.d, z31.d, #32",         // c_hh
        "and     z31.d, z31.d, #0xFFFFFFFF", // c_hl
        "mov     z27.d, #-4294967295",       // GP
        "sub     z28.d, z29.d, z30.d",       // c_l - c_hh
        "cmphs   p6.d, p7/z, z29.d, z30.d",
        "add     z27.d, z27.d, z28.d",
        "sel     z27.d, p6, z28.d, z27.d",   // c1
        "mov     z29.d, #0xFFFFFFFF",        // P_n
        "mov     z30.d, #-4294967295",       // GP
        "sub     z30.d, p7/m, z30.d, z27.d", // GP - c1 (GP-a)
        "mul     z31.d, p7/m, z31.d, z29.d", // c2 (c1 + c_hl * P_n)
        "add     z27.d, p7/m, z27.d, z31.d", // c1 + c2
        "cmphi   p6.d, p7/z, z31.d, z30.d",
        "add     z27.d, p6/m, z27.d, z29.d", // + P_n
        "st1d    z27.d, p7, [{p0}]",
        p0 = inout(reg) state_ptr,
        r0 = in(reg) rc_ptr,
    );
    assert_ne!(x, *state_ptr);
}

unsafe fn add_sbox_sve2_128<F>(state: &mut [F; 12], rc: &[u64; 12], idx: usize)
where
    F: PrimeField64 + Poseidon,
{
    let mut state_ptr = (state[idx..idx + 2]).as_mut_ptr();
    let rc_ptr = (rc[idx..idx + 2]).as_ptr();

    let x: F = *state_ptr;

    asm!(
        // add
        "ptrue   p7.b, all",
        "ld1d    z31.d, p7/z, [{p0}]",
        "ld1d    z28.d, p7/z, [{r0}]",
        "mov     z20.d, #4294967295",
        "mov     z27.d, #-4294967295",
        "sub     z27.d, z27.d, z31.d",
        "add     z31.d, z31.d, z28.d",
        "cmphi   p6.d, p7/z, z28.d, z27.d",
        "add     z31.d, p6/m, z31.d, z20.d",    // a (z31.d)
        "mov     z10.d, z31.d", // save a
        // sbox
        // square   -> a^2
        "umulh   z29.d, z31.d, z31.d",     // c_h
        "mul     z28.d, z31.d, z31.d",     // c_l
        // reduce
        "mov     z31.d, z29.d",
        "and     z31.d, z31.d, #0xFFFFFFFF", // c_hl
        "lsr     z30.d, z30.d, #32",         // c_hh
        "mov     z27.d, #-4294967295",       // GP
        "sub     z29.d, z28.d, z30.d",       // c_l - c_hh
        "cmphs   p6.d, p7/z, z28.d, z30.d",
        "add     z27.d, z27.d, z29.d",
        "sel     z27.d, p6, z29.d, z27.d",   // c1
        "mov     z29.d, #0xFFFFFFFF",        // P_n
        "mov     z30.d, #-4294967295",       // GP
        "sub     z30.d, p7/m, z30.d, z27.d", // GP - c1 (GP-a)
        "mul     z31.d, p7/m, z31.d, z29.d", // c2 (c_hl * P_n)
        "add     z27.d, p7/m, z27.d, z31.d", // c1 + c2
        "cmphi   p6.d, p7/z, z31.d, z30.d",
        "add     z27.d, p6/m, z27.d, z29.d", // + P_n
        "mov     z11.d, z27.d",     // save a^2
        // square   -> a^4
        "umulh   z29.d, z27.d, z27.d",     // c_h
        "mul     z28.d, z27.d, z27.d",     // c_l
        // reduce
        "mov     z31.d, z29.d",
        "and     z31.d, z31.d, #0xFFFFFFFF", // c_hl
        "lsr     z30.d, z30.d, #32",         // c_hh
        "mov     z27.d, #-4294967295",       // GP
        "sub     z28.d, z28.d, z30.d",       // c_l - c_hh
        "cmphs   p6.d, p7/z, z28.d, z30.d",
        "add     z27.d, z27.d, z28.d",
        "sel     z27.d, p6, z28.d, z27.d",   // c1
        "mov     z29.d, #0xFFFFFFFF",        // P_n
        "mov     z30.d, #-4294967295",       // GP
        "sub     z30.d, p7/m, z30.d, z27.d", // GP - c1 (GP-a)
        "mul     z31.d, p7/m, z31.d, z29.d", // c2 (c1 + c_hl * P_n)
        "add     z27.d, p7/m, z27.d, z31.d", // c1 + c2
        "cmphi   p6.d, p7/z, z31.d, z30.d",
        "add     z27.d, p6/m, z27.d, z29.d", // + P_n
        "mov     z12.d, z27.d",     // save a^4
        // mul  -> a * a^2
        "umulh   z31.d, z11.d, z10.d",     // c_h
        "mul     z28.d, z11.d, z10.d",     // c_l
        // reduce
        "lsr     z30.d, z31.d, #32",         // c_hh
        "and     z31.d, z31.d, #0xFFFFFFFF", // c_hl
        "mov     z27.d, #-4294967295",       // GP
        "sub     z28.d, z28.d, z30.d",       // c_l - c_hh
        "cmphs   p6.d, p7/z, z29.d, z30.d",
        "add     z27.d, z27.d, z28.d",
        "sel     z27.d, p6, z28.d, z27.d",   // c1
        "mov     z29.d, #0xFFFFFFFF",        // P_n
        "mov     z30.d, #-4294967295",       // GP
        "sub     z30.d, p7/m, z30.d, z27.d", // GP - c1 (GP-a)
        "mul     z31.d, p7/m, z31.d, z29.d", // c2 (c1 + c_hl * P_n)
        "add     z27.d, p7/m, z27.d, z31.d", // c1 + c2
        "cmphi   p6.d, p7/z, z31.d, z30.d",
        "add     z27.d, p6/m, z27.d, z29.d", // + P_n
        // mul -> a^3 * a^4
        "umulh   z31.d, z27.d, z12.d",     // c_h
        "mul     z28.d, z27.d, z12.d",     // c_l
        // reduce
        "lsr     z30.d, z31.d, #32",         // c_hh
        "and     z31.d, z31.d, #0xFFFFFFFF", // c_hl
        "mov     z27.d, #-4294967295",       // GP
        "sub     z28.d, z28.d, z30.d",       // c_l - c_hh
        "cmphs   p6.d, p7/z, z29.d, z30.d",
        "add     z27.d, z27.d, z28.d",
        "sel     z27.d, p6, z28.d, z27.d",   // c1
        "mov     z29.d, #0xFFFFFFFF",        // P_n
        "mov     z30.d, #-4294967295",       // GP
        "sub     z30.d, p7/m, z30.d, z27.d", // GP - c1 (GP-a)
        "mul     z31.d, p7/m, z31.d, z29.d", // c2 (c1 + c_hl * P_n)
        "add     z27.d, p7/m, z27.d, z31.d", // c1 + c2
        "cmphi   p6.d, p7/z, z31.d, z30.d",
        "add     z27.d, p6/m, z27.d, z29.d", // + P_n
        "st1d    z27.d, p7, [{p0}]",
        p0 = inout(reg) state_ptr,
        r0 = in(reg) rc_ptr,
    );
    assert_ne!(x, *state_ptr);
}

#[inline(always)]
unsafe fn add_sbox_all<F>(state: &mut [F; 12], rc: &[u64; 12])
where
    F: PrimeField64 + Poseidon,
{
    let mut state_ptr_0 = (state[0..4]).as_mut_ptr();
    let mut state_ptr_1 = (state[4..8]).as_mut_ptr();
    let mut state_ptr_2 = (state[8..12]).as_mut_ptr();
    let rc_ptr_0 = (rc[0..4]).as_ptr();
    let rc_ptr_1 = (rc[4..8]).as_ptr();
    let rc_ptr_2 = (rc[8..12]).as_ptr();

    // let x0: F = *state_ptr_0;
    // let x1: F = *state_ptr_1;
    // let x2: F = *state_ptr_2;

    asm!(
        // add
        "ptrue   p7.b, all",
        "ld1d    z31.d, p7/z, [{p0}]",
        "ld1d    z30.d, p7/z, [{p1}]",
        "ld1d    z29.d, p7/z, [{p2}]",
        "ld1d    z28.d, p7/z, [{r0}]",
        "ld1d    z27.d, p7/z, [{r1}]",
        "ld1d    z26.d, p7/z, [{r2}]",
        "mov     z20.d, #4294967295",
        "mov     z23.d, #-4294967295",
        "mov     z22.d, #-4294967295",
        "mov     z21.d, #-4294967295",
        "sub     z23.d, z23.d, z31.d",
        "sub     z22.d, z22.d, z30.d",
        "sub     z21.d, z21.d, z29.d",
        "add     z31.d, z31.d, z28.d",
        "add     z30.d, z30.d, z27.d",
        "add     z29.d, z29.d, z26.d",
        "cmphi   p6.d, p7/z, z28.d, z23.d",
        "cmphi   p5.d, p7/z, z27.d, z22.d",
        "cmphi   p4.d, p7/z, z26.d, z21.d",
        "add     z31.d, p6/m, z31.d, z20.d",
        "add     z30.d, p5/m, z30.d, z20.d",
        "add     z29.d, p4/m, z29.d, z20.d",
        "st1d    z31.d, p7, [{p0}]",
        "st1d    z30.d, p7, [{p1}]",
        "st1d    z29.d, p7, [{p2}]",
        p0 = inout(reg) state_ptr_0,
        p1 = inout(reg) state_ptr_1,
        p2 = inout(reg) state_ptr_2,
        r0 = in(reg) rc_ptr_0,
        r1 = in(reg) rc_ptr_1,
        r2 = in(reg) rc_ptr_2,
    );
    /*
    for i in 0..12 {
        state[i] += F::from_canonical_u64(rc[i]);
    }
    */

    // assert_ne!(x0, *state_ptr_0);
    // assert_ne!(x1, *state_ptr_1);
    // assert_ne!(x2, *state_ptr_2);

    F::sbox_layer(state);
}

pub fn poseidon_sve<F>(input: &[F; SPONGE_WIDTH]) -> [F; SPONGE_WIDTH]
where
    F: PrimeField64 + Poseidon,
{
    let state = &mut input.clone();
    let mut round_ctr = 0;

    unsafe {
        // load state
        // let mut pp0 = (state[0..4]).as_mut_ptr();
        // let mut pp1 = (state[4..8]).as_mut_ptr();
        // let mut pp2 = (state[8..12]).as_mut_ptr();

        for _ in 0..HALF_N_FULL_ROUNDS {
            let rc: &[u64; 12] = &ALL_ROUND_CONSTANTS[SPONGE_WIDTH * round_ctr..][..SPONGE_WIDTH]
                .try_into()
                .unwrap();
            // let pr0 = (rc[0..4]).as_ptr();
            // let pr1 = (rc[4..8]).as_ptr();
            // let pr2 = (rc[8..12]).as_ptr();

            add_sbox_sve_256(state, rc, 0);
            add_sbox_sve_256(state, rc, 4);
            add_sbox_sve_256(state, rc, 8);

            // add_sbox_all(state, rc);

            *state = F::mds_layer(state);
            round_ctr += 1;
        }
    }

    // F::full_rounds(state, &mut round_ctr);
    F::partial_rounds(state, &mut round_ctr);
    // F::full_rounds(state, &mut round_ctr);
    unsafe {
        // load state
        // let mut pp0 = (state[0..4]).as_mut_ptr();
        // let mut pp1 = (state[4..8]).as_mut_ptr();
        // let mut pp2 = (state[8..12]).as_mut_ptr();

        for _ in 0..HALF_N_FULL_ROUNDS {
            let rc: &[u64; 12] = &ALL_ROUND_CONSTANTS[SPONGE_WIDTH * round_ctr..][..SPONGE_WIDTH]
                .try_into()
                .unwrap();
            // let pr0 = (rc[0..4]).as_ptr();
            // let pr1 = (rc[4..8]).as_ptr();
            // let pr2 = (rc[8..12]).as_ptr();

            add_sbox_sve_256(state, rc, 0);
            add_sbox_sve_256(state, rc, 4);
            add_sbox_sve_256(state, rc, 8);

            // add_sbox_all(state, rc);

            *state = F::mds_layer(state);
            round_ctr += 1;
        }
    }
    *state
}
