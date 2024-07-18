use core::arch::asm;

use crate::field::types::PrimeField64;
use crate::hash::poseidon::{
    add_u160_u128, reduce_u160, Poseidon, ALL_ROUND_CONSTANTS, HALF_N_FULL_ROUNDS,
    N_PARTIAL_ROUNDS, N_ROUNDS, SPONGE_WIDTH,
};

pub fn poseidon_sve<F>(input: &[F; SPONGE_WIDTH]) -> [F; SPONGE_WIDTH]
where
    F: PrimeField64 + Poseidon,
{
    let mut state = &mut input.clone();
    let mut round_ctr = 0;

    unsafe {
        // load state
        let mut pp0 = (state[0..4]).as_mut_ptr();
        let mut pp1 = (state[4..8]).as_mut_ptr();
        let mut pp2 = (state[8..12]).as_mut_ptr();

        for _ in 0..HALF_N_FULL_ROUNDS {
            asm!(
                "ptrue   p7.b, all",
                "ld1d    z31.d, p7/z, [{p0}]",
                "ld1d    z30.d, p7/z, [{p1}]",
                "ld1d    z29.d, p7/z, [{p2}]",
                p0 = inout(reg) pp0,
                p1 = inout(reg) pp1,
                p2 = inout(reg) pp2,
            );
            let rc: &[u64; 12] = &ALL_ROUND_CONSTANTS[SPONGE_WIDTH * round_ctr..][..SPONGE_WIDTH]
                .try_into()
                .unwrap();
            let pr0 = (rc[0..4]).as_ptr();
            let pr1 = (rc[4..8]).as_ptr();
            let pr2 = (rc[8..12]).as_ptr();
            asm!(
                "mov     z20.d, #4294967295",
                // add 1
                "ld1d    z28.d, p7/z, [{r0}]",
                "mov     z27.d, #-4294967295",
                "sub     z27.d, z27.d, z31.d",
                "add     z31.d, z31.d, z28.d",
                "cmphi   p6.d, p7/z, z28.d, z27.d",
                "add     z31.d, p6/m, z31.d, z20.d",
                // add 2
                "ld1d    z28.d, p7/z, [{r1}]",
                "mov     z27.d, #-4294967295",
                "sub     z27.d, z27.d, z30.d",
                "add     z30.d, z30.d, z28.d",
                "cmphi   p6.d, p7/z, z28.d, z27.d",
                "add     z30.d, p6/m, z30.d, z20.d",
                // add 3
                "ld1d    z28.d, p7/z, [{r2}]",
                "mov     z27.d, #-4294967295",
                "sub     z27.d, z27.d, z29.d",
                "add     z29.d, z29.d, z28.d",
                "cmphi   p6.d, p7/z, z28.d, z27.d",
                "add     z29.d, p6/m, z29.d, z20.d",
                // save
                "mov z10.d, z31.d",
                "mov z11.d, z30.d",
                "mov z12.d, z29.d",
                // sbox 1 (a is in z31)
                // a^2
                "lsr     z30.d, z31.d, #32",         // a_h
                "and     z31.d, z31.d, #0xFFFFFFFF", // a_l
                "mov     z29.d, z31.d",
                "mul     z29.d, p7/m, z29.d, z31.d",  // c_ll = a_l * a_l
                "lsr     z28.d, z29.d, #33",          // c_ll_h
                "and     z29.d, z29.d, #0x1FFFFFFFF", // c_ll_l
                "mad     z31.d, p7/m, z30.d, z28.d",  // r0
                "lsl     z28.d, z31.d, #33",          // r0_l
                "add     z28.d, p7/m, z28.d, z29.d",  // ** c_l
                "lsr     z31.d, z31.d, #31",          // r0_h
                "mad     z30.d, p7/m, z30.d, z31.d",  // ** c_h
                "lsr     z31.d, z30.d, #32",         // c_hh
                "and     z30.d, z30.d, #0xFFFFFFFF", // c_hl
                "mov     z27.d, #-4294967295",       // GP
                "sub     z29.d, z28.d, z31.d",       // c_l - c_hh
                "cmphs   p6.d, p7/z, z28.d, z31.d",
                "add     z27.d, z27.d, z29.d",
                "sel     z27.d, p6, z29.d, z27.d",   // c1
                "mov     z29.d, #0xFFFFFFFF",        // P_n
                "mov     z31.d, #-4294967295",       // GP
                "sub     z31.d, p7/m, z31.d, z27.d", // GP - c1 (GP-a)
                "mul     z30.d, p7/m, z30.d, z29.d", // c2 (c1 + c_hl * P_n)
                "add     z27.d, p7/m, z27.d, z30.d", // c1 + c2
                "cmphi   p6.d, p7/z, z30.d, z31.d",
                "add     z27.d, p6/m, z27.d, z29.d", // + P_n -> a^2
                // a^4 (a^2 is in z27.d)
                "mov     z13.d, z27.d",
                "mov     z31.d, z27.d",
                "lsr     z30.d, z31.d, #32",         // a_h
                "and     z31.d, z31.d, #0xFFFFFFFF", // a_l
                "mov     z29.d, z31.d",
                "mul     z29.d, p7/m, z29.d, z31.d",  // c_ll = a_l * a_l
                "lsr     z28.d, z29.d, #33",          // c_ll_h
                "and     z29.d, z29.d, #0x1FFFFFFFF", // c_ll_l
                "mad     z31.d, p7/m, z30.d, z28.d",  // r0
                "lsl     z28.d, z31.d, #33",          // r0_l
                "add     z28.d, p7/m, z28.d, z29.d",  // ** c_l
                "lsr     z31.d, z31.d, #31",          // r0_h
                "mad     z30.d, p7/m, z30.d, z31.d",  // ** c_h
                "lsr     z31.d, z30.d, #32",         // c_hh
                "and     z30.d, z30.d, #0xFFFFFFFF", // c_hl
                "mov     z27.d, #-4294967295",       // GP
                "sub     z29.d, z28.d, z31.d",       // c_l - c_hh
                "cmphs   p6.d, p7/z, z28.d, z31.d",
                "add     z27.d, z27.d, z29.d",
                "sel     z27.d, p6, z29.d, z27.d",   // c1
                "mov     z29.d, #0xFFFFFFFF",        // P_n
                "mov     z31.d, #-4294967295",       // GP
                "sub     z31.d, p7/m, z31.d, z27.d", // GP - c1 (GP-a)
                "mul     z30.d, p7/m, z30.d, z29.d", // c2 (c1 + c_hl * P_n)
                "add     z27.d, p7/m, z27.d, z30.d", // c1 + c2
                "cmphi   p6.d, p7/z, z30.d, z31.d",
                "add     z27.d, p6/m, z27.d, z29.d", // + P_n -> a^4
                "mov    z14.d, z27.d",
                // a * a^2
                "mov    z31.d, z10.d",   // a
                "mov    z30.d, z13.d",   // a^2
                "lsr     z29.d, z31.d, #32",         // a_h
                "lsr     z28.d, z30.d, #32",         // b_h
                "and     z31.d, z31.d, #0xFFFFFFFF", // a_l
                "and     z30.d, z30.d, #0xFFFFFFFF", // b_l
                "mov     z24.d, z31.d",              // save a_l
                "mul     z31.d, p7/m, z31.d, z30.d", // c_ll = a_l * b_l
                "lsr     z26.d, z31.d, #32",         // c_ll_h
                "mad     z30.d, p7/m, z29.d, z26.d", // r0 (c_hl)
                "lsr     z26.d, z30.d, #32",         // r0_h
                "and     z30.d, z30.d, #0xFFFFFFFF", // r0_l
                "mad     z29.d, p7/m, z28.d, z26.d", // r2
                "mad     z24.d, p7/m, z28.d, z30.d", // r1
                "lsr     z25.d, z24.d, #32",         // r1_h
                "add     z29.d, p7/m, z29.d, z25.d", // ** c_h
                "lsl     z25.d, z24.d, #32", // r1_l
                "ptrue   p15.s, all",
                "eor     p15.b, p15/z, p7.b, p15.b", // sel
                "sel     z31.s, p15, z25.s, z31.s",  // ** c_l
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
                "add     z27.d, p6/m, z27.d, z29.d", // + P_n (a^3)
                // a^4 * a^3
                "mov    z31.d, z27.d",   // a^3
                "mov    z30.d, z14.d",   // a^4
                "lsr     z29.d, z31.d, #32",         // a_h
                "lsr     z28.d, z30.d, #32",         // b_h
                "and     z31.d, z31.d, #0xFFFFFFFF", // a_l
                "and     z30.d, z30.d, #0xFFFFFFFF", // b_l
                "mov     z24.d, z31.d",              // save a_l
                "mul     z31.d, p7/m, z31.d, z30.d", // c_ll = a_l * b_l
                "lsr     z26.d, z31.d, #32",         // c_ll_h
                "mad     z30.d, p7/m, z29.d, z26.d", // r0 (c_hl)
                "lsr     z26.d, z30.d, #32",         // r0_h
                "and     z30.d, z30.d, #0xFFFFFFFF", // r0_l
                "mad     z29.d, p7/m, z28.d, z26.d", // r2
                "mad     z24.d, p7/m, z28.d, z30.d", // r1
                "lsr     z25.d, z24.d, #32",         // r1_h
                "add     z29.d, p7/m, z29.d, z25.d", // ** c_h
                "lsl     z25.d, z24.d, #32", // r1_l
                "ptrue   p15.s, all",
                "eor     p15.b, p15/z, p7.b, p15.b", // sel
                "sel     z31.s, p15, z25.s, z31.s",  // ** c_l
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
                "add     z27.d, p6/m, z27.d, z29.d", // + P_n (a^7)
                "st1d   z27.d, p7, [{p0}]",
                // sbox 2
                "st1d   z11.d, p7, [{p1}]",
                "st1d   z12.d, p7, [{p2}]",
                // sbox 3
                // mds
                r0 = in(reg) pr0,
                r1 = in(reg) pr1,
                r2 = in(reg) pr2,
                p0 = inout(reg) pp0,
                p1 = inout(reg) pp1,
                p2 = inout(reg) pp2,
            );
            for i in 4..12 {
                state[i] = F::sbox_monomial(state[i]);
            }
        }
    };
    F::partial_rounds(state, &mut round_ctr);
    F::full_rounds(state, &mut round_ctr);

    *state
}
