// use core::arch::asm;
use core::arch::aarch64::*;

use crate::hash::hash_types::RichField;

const MSB_: i64 = 0x8000000000000000u64 as i64;
const P_s_: i64 = 0x7FFFFFFF00000001u64 as i64;
const P_n_: i64 = 0xFFFFFFFF;

#[inline(always)]
pub fn shift_neon(a: &int64x2_t) -> int64x2_t {
    unsafe {
        let MSB = vdupq_n_s64(MSB_);
        veorq_s64(*a, MSB)
    }
}

#[allow(dead_code)]
#[inline(always)]
pub fn toCanonical_neon_s(a_s: &int64x2_t) -> int64x2_t {
    unsafe {
        let P_s = vdupq_n_s64(P_s_);
        let P_n = vreinterpretq_u32_i64(vdupq_n_s64(P_n_));
        let mask1_ = vcgtq_s64(P_s, *a_s);
        let mn = vmvnq_u32(vreinterpretq_u32_u64(mask1_));
        let corr1_ = vreinterpretq_s64_u32(vandq_u32(mn, P_n));
        vaddq_s64(*a_s, corr1_)
    }
}

#[inline(always)]
fn mul_epu32(a: int64x2_t, b: int64x2_t)) -> int64x2_t {
    let a_l = vmovn_u64(a);
    let b_l = vmovn_u64(b);
    vmull_u32(a_l, b_l)
}

// similar to _mm256_blend_epi32(a, b, 0xaa);
#[inline(always)]
fn blend_epi32(a: int64x2_t, b: int64x2_t)) -> int64x2_t {
    let a_l = vmovn_u64(a);
    let b_l = vmovn_u64(b);
    vmull_u32(a_l, b_l)
}

#[inline(always)]
pub fn add_neon_a_sc(a_sc: &int64x2_t, b: &int64x2_t) -> int64x2_t {
    unsafe {
        let c0_s = vaddq_s64(*a_sc, *b);
        let P_n = vdupq_n_u64(P_n_);
        let mask_ = vcgtq_s64(*a_sc, c0_s);
        let corr_ = vreinterpretq_s64_u64(vandq_u64(mask_, P_n));
        let c_s = vaddq_s64(c0_s, corr_);
        shift_neon(&c_s)
    }
}

#[inline(always)]
pub fn add_neon(a: &int64x2_t, b: &int64x2_t) -> int64x2_t {
    let a_sc = shift_neon(a);
    // let a_sc = toCanonical_neon_s(&a_s);
    add_neon_a_sc(&a_sc, b)
}

#[inline(always)]
pub fn add_neon_s_b_small(a_s: &int64x2_t, b_small: &int64x2_t) -> int64x2_t {
    unsafe {
        let c0_s = vaddq_s64(*a_s, *b_small);
        let mask_ = vcgtq_s64(*a_s, c0_s);
        let corr_ = vreinterpretq_s64_u64(vshrq_n_u64(mask_, 32));
        vaddq_s64(c0_s, corr_)
    }
}

#[inline(always)]
pub fn sub_neon_s_b_small(a_s: &int64x2_t, b: &int64x2_t) -> int64x2_t {
    unsafe {
        let c0_s = vsubq_s64(*a_s, *b);
        let mask_ = vcgtq_s64(c0_s, *a_s);
        let corr_ = vreinterpretq_s64_u64(vshrq_n_u64(mask_, 32));
        vsubq_s64(c0_s, corr_)
    }
}

#[inline(always)]
pub fn reduce_neon_128_64(c_h: &int64x2_t, c_l: &int64x2_t) -> int64x2_t {
    unsafe {
        let MSB = vdupq_n_s64(MSB_);
        let c_hh = vshrq_n_s64(*c_h, 32);
        let c_ls = veorq_s64(*c_l, MSB);
        let c1_s = sub_neon_s_b_small(&c_ls, &c_hh);
        let P_n = vdupq_n_s64(P_n_);
        let c2 = mul_epu32(*c_h, P_n);
        let c_s = add_neon_s_b_small(&c1_s, &c2);
        veorq_s64(c_s, MSB)
    }
}

#[inline(always)]
pub fn mult_neon_128(a: &int64x2_t, b: &int64x2_t) -> (int64x2_t, int64x2_t) {
    unsafe {
        let a_h = vshrq_n_s64(*a, 32);
        let b_h = vshrq_n_s64(*b, 32);
        let c_hh = mul_epu32(a_h, b_h);
        let c_hl = mul_epu32(a_h, *b);
        let c_lh = mul_epu32(*a, b_h);
        let c_ll = mul_epu32(*a, *b);
        let c_ll_h = vshrq_n_s64(c_ll, 32);
        let r0 = vaddq_s64(c_hl, c_ll_h);
        let P_n = vdupq_n_s64(P_n_);
        let r0_l = vandq_s64(r0, P_n);
        let r0_h = vshrq_n_s64(r0, 32);
        let r1 = vaddq_s64(c_lh, r0_l);
        let r1_l = vshlq_s64(r1, 32);
        let c_l = _mm256_blend_epi32(c_ll, r1_l, 0xaa); // TODO
        let r2 = vaddq_s64(c_hh, r0_h);
        let r1_h = vshrq_n_s64(r1, 32);
        let c_h = vaddq_s64(r2, r1_h);
        (c_h, c_l)
    }
}

#[inline(always)]
pub fn mult_neon(a: &int64x2_t, b: &int64x2_t) -> int64x2_t {
    let (c_h, c_l) = mult_neon_128(a, b);
    reduce_neon_128_64(&c_h, &c_l)
}

#[inline(always)]
pub fn sqr_neon_128(a: &int64x2_t) -> (int64x2_t, int64x2_t) {
    unsafe {
        let a_h = vshrq_n_s64(*a);
        let c_ll = mul_epu32(*a, *a);
        let c_lh = mul_epu32(*a, a_h);
        let c_hh = mul_epu32(a_h, a_h);
        let c_ll_hi = vshrq_n_s64(c_ll, 33);
        let t0 = vaddq_s64(c_lh, c_ll_hi);
        let t0_hi = vshrq_n_s64(t0, 31);
        let res_hi = vaddq_s64(c_hh, t0_hi);
        let c_lh_lo = vshlq_s64(c_lh, 33);
        let res_lo = vaddq_s64(c_ll, c_lh_lo);
        (res_hi, res_lo)
    }
}

#[inline(always)]
pub fn sqr_neon(a: &int64x2_t) -> int64x2_t {
    let (c_h, c_l) = sqr_neon_128(a);
    reduce_neon_128_64(&c_h, &c_l)
}

#[inline(always)]
pub fn sbox_neon<F>(state: &mut [F; 12])
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
        // x^2
        let p10 = sqr_neon(&s0);
        let p11 = sqr_neon(&s1);
        let p12 = sqr_neon(&s2);
        let p13 = sqr_neon(&s3);
        let p14 = sqr_neon(&s4);
        let p15 = sqr_neon(&s5);
        // x^3
        let p20 = mult_neon(&p10, &s0);
        let p21 = mult_neon(&p11, &s1);
        let p22 = mult_neon(&p12, &s2);
        let p23 = mult_neon(&p13, &s3);
        let p24 = mult_neon(&p14, &s4);
        let p25 = mult_neon(&p15, &s5);
        // x^4 = (x^2)^2
        let s0 = sqr_neon(&p10);
        let s1 = sqr_neon(&p11);
        let s2 = sqr_neon(&p12);
        let s3 = sqr_neon(&p13);
        let s4 = sqr_neon(&p14);
        let s5 = sqr_neon(&p15);
        // x^7
        let p10 = mult_neon(&s0, &p20);
        let p11 = mult_neon(&s1, &p21);
        let p12 = mult_neon(&s2, &p22);
        let p13 = mult_neon(&s0, &p23);
        let p14 = mult_neon(&s1, &p24);
        let p15 = mult_neon(&s2, &p25);
        vst1q_s64((&mut state[0..2]).as_mut_ptr().cast::<i64>(), p10);
        vst1q_s64((&mut state[2..4]).as_mut_ptr().cast::<i64>(), p11);
        vst1q_s64((&mut state[4..6]).as_mut_ptr().cast::<i64>(), p12);
        vst1q_s64((&mut state[6..8]).as_mut_ptr().cast::<i64>(), p13);
        vst1q_s64((&mut state[8..10]).as_mut_ptr().cast::<i64>(), p14);
        vst1q_s64((&mut state[10..12]).as_mut_ptr().cast::<i64>(), p15);
    }
}

#[inline(always)]
pub fn sbox_neon_6pack(s0: &int64x2_t, s1: &int64x2_t, s2: &int64x2_t, s3: &int64x2_t, s4: &int64x2_t, s5: &int64x2_t) -> (int64x2_t, int64x2_t, int64x2_t, int64x2_t, int64x2_t, int64x2_t) {
    // x^2
    let p10 = sqr_neon(s0);
    let p11 = sqr_neon(s1);
    let p12 = sqr_neon(s2);
    let p13 = sqr_neon(s3);
    let p14 = sqr_neon(s4);
    let p15 = sqr_neon(s5);
    // x^3
    let p30 = mult_neon(&p10, s0);
    let p31 = mult_neon(&p11, s1);
    let p32 = mult_neon(&p12, s2);
    let p33 = mult_neon(&p13, s3);
    let p34 = mult_neon(&p14, s4);
    let p35 = mult_neon(&p15, s5);
    // x^4 = (x^2)^2
    let p40 = sqr_neon(&p10);
    let p41 = sqr_neon(&p11);
    let p42 = sqr_neon(&p12);
    let p43 = sqr_neon(&p13);
    let p44 = sqr_neon(&p14);
    let p45 = sqr_neon(&p15);
    // x^7
    let r0 = mult_neon(&p40, &p30);
    let r1 = mult_neon(&p41, &p31);
    let r2 = mult_neon(&p42, &p32);
    let r3 = mult_neon(&p43, &p33);
    let r4 = mult_neon(&p44, &p34);
    let r5 = mult_neon(&p45, &p35);

    (r0, r1, r2, r3, r4, r5)
}
