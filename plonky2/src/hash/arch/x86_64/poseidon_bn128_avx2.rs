use core::arch::x86_64::*;

use crate::hash::poseidon_bn128_ops::{ElementBN128, C, M, P, S};
#[cfg(feature = "papi")]
use crate::util::papi::{init_papi, stop_papi};

#[allow(dead_code)]
#[inline]
unsafe fn set_zero() -> __m256i {
    _mm256_set_epi64x(0, 0, 0, 0)
}

#[allow(dead_code)]
#[inline]
unsafe fn set_one() -> __m256i {
    _mm256_set_epi64x(
        1011752739694698287i64,
        7381016538464732718i64,
        3962172157175319849i64,
        12436184717236109307u64 as i64,
    )
}

#[inline]
pub unsafe fn add64_no_carry(a: &__m256i, b: &__m256i) -> (__m256i, __m256i) {
    /*
     * a and b are signed 4 x i64. Suppose a and b represent only one i64, then:
     * - (test 1): if a < 2^63 and b < 2^63 (this means a >= 0 and b >= 0) => sum does not overflow => cout = 0
     * - if a >= 2^63 and b >= 2^63 => sum overflows so sum = a + b and cout = 1
     * - (test 2): if (a < 2^63 and b >= 2^63) or (a >= 2^63 and b < 2^63)
     *   - (test 3): if a + b < 2^64 (this means a + b is negative in signed representation) => no overflow so cout = 0
     *   - (test 3): if a + b >= 2^64 (this means a + b becomes positive in signed representation, that is, a + b >= 0) => there is overflow so cout = 1
     */
    let ones = _mm256_set_epi64x(1, 1, 1, 1);
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let r = _mm256_add_epi64(*a, *b);
    let ma = _mm256_cmpgt_epi64(zeros, *a);
    let mb = _mm256_cmpgt_epi64(zeros, *b);
    let m1 = _mm256_and_si256(ma, mb); // test 1
    let m21 = _mm256_andnot_si256(ma, mb);
    let m22 = _mm256_andnot_si256(mb, ma);
    let m2 = _mm256_or_si256(m21, m22); // test 2
    let m23 = _mm256_cmpgt_epi64(zeros, r); // test 3
    let m2 = _mm256_andnot_si256(m23, m2);
    let m = _mm256_or_si256(m1, m2);
    let co = _mm256_and_si256(m, ones);
    (r, co)
}

// cin is carry in and must be 0 or 1
#[inline]
pub unsafe fn add64(a: &__m256i, b: &__m256i, cin: &__m256i) -> (__m256i, __m256i) {
    let (r1, c1) = add64_no_carry(a, b);
    let max = _mm256_set_epi64x(-1, -1, -1, -1);
    let m = _mm256_cmpeq_epi64(r1, max);
    let r = _mm256_add_epi64(r1, *cin);
    let m = _mm256_and_si256(*cin, m);
    let co = _mm256_or_si256(m, c1);
    (r, co)
}

#[inline]
unsafe fn sub64_no_borrow(a: &__m256i, b: &__m256i) -> (__m256i, __m256i) {
    let ones = _mm256_set_epi64x(1, 1, 1, 1);
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let r = _mm256_sub_epi64(*a, *b);
    let m1 = _mm256_cmpgt_epi64(zeros, *a); // a < 0 ?
    let m2 = _mm256_cmpgt_epi64(zeros, *b); // b < 0 ?
    let m3 = _mm256_cmpgt_epi64(*b, *a); // a < b ?
    let m4 = _mm256_or_si256(m2, m3);
    let m5 = _mm256_andnot_si256(m1, m4);
    let m6 = _mm256_and_si256(m2, m3);
    let m7 = _mm256_and_si256(m1, m6);
    let m = _mm256_or_si256(m5, m7);
    let bo = _mm256_and_si256(m, ones);
    (r, bo)
}

// bin is borrow in and must be 0 or 1
// TODO: revise
#[inline]
unsafe fn sub64(a: &__m256i, b: &__m256i, bin: &__m256i) -> (__m256i, __m256i) {
    let ones = _mm256_set_epi64x(1, 1, 1, 1);
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let (r1, b1) = sub64_no_borrow(a, b);

    let m1 = _mm256_cmpeq_epi64(*bin, ones);
    let m2 = _mm256_cmpeq_epi64(r1, zeros);
    let m = _mm256_and_si256(m1, m2);
    let bo = _mm256_and_si256(m, ones);
    let r = _mm256_sub_epi64(r1, *bin);
    let bo = _mm256_or_si256(bo, b1);

    (r, bo)
}

#[allow(dead_code)]
#[inline]
unsafe fn mul64_v1(a: &__m256i, b: &__m256i) -> (__m256i, __m256i) {
    let mut av: [u64; 4] = [0; 4];
    let mut bv: [u64; 4] = [0; 4];
    let mut hv: [u64; 4] = [0; 4];
    let mut lv: [u64; 4] = [0; 4];
    _mm256_storeu_si256(av.as_mut_ptr().cast::<__m256i>(), *a);
    _mm256_storeu_si256(bv.as_mut_ptr().cast::<__m256i>(), *b);
    let c0 = (av[0] as u128) * (bv[0] as u128);
    let c1 = (av[1] as u128) * (bv[1] as u128);
    let c2 = (av[2] as u128) * (bv[2] as u128);
    let c3 = (av[3] as u128) * (bv[3] as u128);
    (hv[0], lv[0]) = ((c0 >> 64) as u64, c0 as u64);
    (hv[1], lv[1]) = ((c1 >> 64) as u64, c1 as u64);
    (hv[2], lv[2]) = ((c2 >> 64) as u64, c2 as u64);
    (hv[3], lv[3]) = ((c3 >> 64) as u64, c3 as u64);
    let h = _mm256_loadu_si256(hv.as_mut_ptr().cast::<__m256i>());
    let l = _mm256_loadu_si256(lv.as_mut_ptr().cast::<__m256i>());
    (h, l)
}

unsafe fn mul64(a: &__m256i, b: &__m256i) -> (__m256i, __m256i) {
    let ah = _mm256_srli_epi64(*a, 32);
    let bh = _mm256_srli_epi64(*b, 32);
    let rl = _mm256_mul_epu32(*a, *b);
    let ahbl = _mm256_mul_epu32(ah, *b);
    let albh = _mm256_mul_epu32(*a, bh);
    let rh = _mm256_mul_epu32(ah, bh);
    let (rm, cm) = add64_no_carry(&ahbl, &albh);
    let rm_l = _mm256_slli_epi64(rm, 32);
    let rm_h = _mm256_srli_epi64(rm, 32);
    let (rl, cl) = add64_no_carry(&rl, &rm_l);
    let cm_s = _mm256_slli_epi64(cm, 32);
    let rtmp = _mm256_add_epi64(rh, cm_s);
    let (rh, _) = add64(&rtmp, &rm_h, &cl);

    (rh, rl)
}

// madd0 hi = a*b + c (discards lo bits)
#[inline]
unsafe fn madd0(a: &__m256i, b: &__m256i, c: &__m256i) -> __m256i {
    let (hi, lo) = mul64(a, b);
    let (_, cr) = add64_no_carry(&lo, c);
    let hi = _mm256_add_epi64(hi, cr);
    hi
}

// madd1 hi, lo = a * b + c
#[inline]
unsafe fn madd1(a: &__m256i, b: &__m256i, c: &__m256i) -> (__m256i, __m256i) {
    let (hi, lo) = mul64(a, b);
    let (lo, cr) = add64_no_carry(&lo, c);
    let hi = _mm256_add_epi64(hi, cr);
    (hi, lo)
}

// madd2 hi, lo = a * b + c + d
#[inline]
unsafe fn madd2(a: &__m256i, b: &__m256i, c: &__m256i, d: &__m256i) -> (__m256i, __m256i) {
    let (hi, lo) = mul64(a, b);
    let (c, cr) = add64_no_carry(c, d);
    let hi = _mm256_add_epi64(hi, cr);
    let (lo, cr) = add64_no_carry(&lo, &c);
    let hi = _mm256_add_epi64(hi, cr);
    (hi, lo)
}

#[inline]
unsafe fn madd3(
    a: &__m256i,
    b: &__m256i,
    c: &__m256i,
    d: &__m256i,
    e: &__m256i,
) -> (__m256i, __m256i) {
    let (hi, lo) = mul64(a, b);
    let (c, cr) = add64_no_carry(c, d);
    let hi = _mm256_add_epi64(hi, cr);
    let (lo, cr) = add64_no_carry(&lo, &c);
    let hi = _mm256_add_epi64(hi, cr);
    let hi = _mm256_add_epi64(hi, *e);
    (hi, lo)
}

#[inline]
pub unsafe fn _mm256_mullo_epi64(a: __m256i, b: __m256i) -> __m256i {
    let rl = _mm256_mul_epu32(a, b);
    let ah = _mm256_srli_epi64(a, 32);
    let bh = _mm256_srli_epi64(b, 32);
    let rh_1 = _mm256_mul_epu32(a, bh);
    let rh_2 = _mm256_mul_epu32(ah, b);
    let rh = _mm256_add_epi64(rh_1, rh_2);
    let rh = _mm256_slli_epi64(rh, 32);
    _mm256_add_epi64(rh, rl)
}

#[allow(dead_code)]
#[inline]
pub unsafe fn _mm256_mullo_epi64_v2(a: __m256i, b: __m256i) -> __m256i {
    let mut av: [u64; 4] = [0; 4];
    let mut bv: [u64; 4] = [0; 4];
    _mm256_storeu_si256(av.as_mut_ptr().cast::<__m256i>(), a);
    _mm256_storeu_si256(bv.as_mut_ptr().cast::<__m256i>(), b);
    for i in 0..4 {
        av[i] = ((av[i] as u128) * (bv[i] as u128)) as u64;
    }
    _mm256_loadu_si256(av.as_ptr().cast::<__m256i>())
}

#[inline]
unsafe fn _mul_generic(x: [__m256i; 4], y: [__m256i; 4]) -> [__m256i; 4] {
    let mut z: [__m256i; 4] = [_mm256_set_epi64x(0, 0, 0, 0); 4];
    let mut t: [__m256i; 4] = [_mm256_set_epi64x(0, 0, 0, 0); 4];
    let mut c: [__m256i; 3] = [_mm256_set_epi64x(0, 0, 0, 0); 3];

    let ct0 = _mm256_set_epi64x(
        4891460686036598785i64,
        4891460686036598785i64,
        4891460686036598785i64,
        4891460686036598785i64,
    );
    let ct1 = _mm256_set_epi64x(
        2896914383306846353i64,
        2896914383306846353i64,
        2896914383306846353i64,
        2896914383306846353i64,
    );
    let ct2 = _mm256_set_epi64x(
        13281191951274694749u64 as i64,
        13281191951274694749u64 as i64,
        13281191951274694749u64 as i64,
        13281191951274694749u64 as i64,
    );
    let ct3 = _mm256_set_epi64x(
        3486998266802970665i64,
        3486998266802970665i64,
        3486998266802970665i64,
        3486998266802970665i64,
    );
    let ct4 = _mm256_set_epi64x(
        14042775128853446655u64 as i64,
        14042775128853446655u64 as i64,
        14042775128853446655u64 as i64,
        14042775128853446655u64 as i64,
    );

    // round 0
    let mut v = x[0];
    (c[1], c[0]) = mul64(&v, &y[0]);
    let m = _mm256_mullo_epi64(c[0], ct4);
    c[2] = madd0(&m, &ct0, &c[0]);
    (c[1], c[0]) = madd1(&v, &y[1], &c[1]);
    (c[2], t[0]) = madd2(&m, &ct1, &c[2], &c[0]);
    (c[1], c[0]) = madd1(&v, &y[2], &c[1]);
    (c[2], t[1]) = madd2(&m, &ct2, &c[2], &c[0]);
    (c[1], c[0]) = madd1(&v, &y[3], &c[1]);
    (t[3], t[2]) = madd3(&m, &ct3, &c[0], &c[2], &c[1]);

    // round 1
    v = x[1];
    (c[1], c[0]) = madd1(&v, &y[0], &t[0]);
    let m = _mm256_mullo_epi64(c[0], ct4);
    c[2] = madd0(&m, &ct0, &c[0]);
    (c[1], c[0]) = madd2(&v, &y[1], &c[1], &t[1]);
    (c[2], t[0]) = madd2(&m, &ct1, &c[2], &c[0]);
    (c[1], c[0]) = madd2(&v, &y[2], &c[1], &t[2]);
    (c[2], t[1]) = madd2(&m, &ct2, &c[2], &c[0]);
    (c[1], c[0]) = madd2(&v, &y[3], &c[1], &t[3]);
    (t[3], t[2]) = madd3(&m, &ct3, &c[0], &c[2], &c[1]);

    // round 2
    v = x[2];
    (c[1], c[0]) = madd1(&v, &y[0], &t[0]);
    let m = _mm256_mullo_epi64(c[0], ct4);
    c[2] = madd0(&m, &ct0, &c[0]);
    (c[1], c[0]) = madd2(&v, &y[1], &c[1], &t[1]);
    (c[2], t[0]) = madd2(&m, &ct1, &c[2], &c[0]);
    (c[1], c[0]) = madd2(&v, &y[2], &c[1], &t[2]);
    (c[2], t[1]) = madd2(&m, &ct2, &c[2], &c[0]);
    (c[1], c[0]) = madd2(&v, &y[3], &c[1], &t[3]);
    (t[3], t[2]) = madd3(&m, &ct3, &c[0], &c[2], &c[1]);

    // round 3
    v = x[3];
    (c[1], c[0]) = madd1(&v, &y[0], &t[0]);
    let m = _mm256_mullo_epi64(c[0], ct4);
    c[2] = madd0(&m, &ct0, &c[0]);
    (c[1], c[0]) = madd2(&v, &y[1], &c[1], &t[1]);
    (c[2], z[0]) = madd2(&m, &ct1, &c[2], &c[0]);
    (c[1], c[0]) = madd2(&v, &y[2], &c[1], &t[2]);
    (c[2], z[1]) = madd2(&m, &ct2, &c[2], &c[0]);
    (c[1], c[0]) = madd2(&v, &y[3], &c[1], &t[3]);
    (z[3], z[2]) = madd3(&m, &ct3, &c[0], &c[2], &c[1]);

    // if z > q --> z -= q
    let cmp0 = _mm256_cmpgt_epi64(ct0, z[0]);
    let cmp1 = _mm256_cmpeq_epi64(ct1, z[1]);
    let cmp0 = _mm256_and_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpgt_epi64(ct1, z[1]);
    let cmp0 = _mm256_or_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpeq_epi64(ct2, z[2]);
    let cmp0 = _mm256_and_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpgt_epi64(ct2, z[2]);
    let cmp0 = _mm256_or_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpeq_epi64(ct3, z[3]);
    let cmp0 = _mm256_and_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpgt_epi64(ct3, z[3]);
    let cmp0 = _mm256_or_si256(cmp0, cmp1);
    let st0 = _mm256_andnot_si256(cmp0, ct0);
    let st1 = _mm256_andnot_si256(cmp0, ct1);
    let st2 = _mm256_andnot_si256(cmp0, ct2);
    let st3 = _mm256_andnot_si256(cmp0, ct3);
    let mut b;
    (z[0], b) = sub64_no_borrow(&z[0], &st0);
    (z[1], b) = sub64(&z[1], &st1, &b);
    (z[2], b) = sub64(&z[2], &st2, &b);
    let tmp = _mm256_sub_epi64(z[3], st3);
    z[3] = _mm256_sub_epi64(tmp, b);

    z
}

#[inline]
fn exp5state(state: &mut [__m256i; 8]) {
    let s: [__m256i; 4] = [state[0], state[1], state[2], state[3]];
    unsafe {
        let s2 = _mul_generic(s, s);
        let s4 = _mul_generic(s2, s2);
        let s5 = _mul_generic(s, s4);
        state[0] = s5[0];
        state[1] = s5[1];
        state[2] = s5[2];
        state[3] = s5[3];
    }
    let s: [__m256i; 4] = [state[4], state[5], state[6], state[7]];
    unsafe {
        let s2 = _mul_generic(s, s);
        let s4 = _mul_generic(s2, s2);
        let s5 = _mul_generic(s, s4);
        state[4] = s5[0];
        state[5] = s5[1];
        state[6] = s5[2];
        state[7] = s5[3];
    }
}

#[inline]
unsafe fn _add_generic(x: [__m256i; 4], y: [__m256i; 4]) -> [__m256i; 4] {
    let mut z: [__m256i; 4] = [_mm256_set_epi64x(0, 0, 0, 0); 4];
    let mut cr: __m256i;

    (z[0], cr) = add64_no_carry(&x[0], &y[0]);
    (z[1], cr) = add64(&x[1], &y[1], &cr);
    (z[2], cr) = add64(&x[2], &y[2], &cr);
    let tmp = _mm256_add_epi64(x[3], y[3]);
    z[3] = _mm256_add_epi64(tmp, cr);

    // if z > q --> z -= q
    let ct0 = _mm256_set_epi64x(
        4891460686036598785i64,
        4891460686036598785i64,
        4891460686036598785i64,
        4891460686036598785i64,
    );
    let ct1 = _mm256_set_epi64x(
        2896914383306846353i64,
        2896914383306846353i64,
        2896914383306846353i64,
        2896914383306846353i64,
    );
    let ct2 = _mm256_set_epi64x(
        13281191951274694749u64 as i64,
        13281191951274694749u64 as i64,
        13281191951274694749u64 as i64,
        13281191951274694749u64 as i64,
    );
    let ct3 = _mm256_set_epi64x(
        3486998266802970665i64,
        3486998266802970665i64,
        3486998266802970665i64,
        3486998266802970665i64,
    );

    // if z > q --> z -= q
    let cmp0 = _mm256_cmpgt_epi64(ct0, z[0]);
    let cmp1 = _mm256_cmpeq_epi64(ct1, z[1]);
    let cmp0 = _mm256_and_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpgt_epi64(ct1, z[1]);
    let cmp0 = _mm256_or_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpeq_epi64(ct2, z[2]);
    let cmp0 = _mm256_and_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpgt_epi64(ct2, z[2]);
    let cmp0 = _mm256_or_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpeq_epi64(ct3, z[3]);
    let cmp0 = _mm256_and_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpgt_epi64(ct3, z[3]);
    let cmp0 = _mm256_or_si256(cmp0, cmp1);
    let st0 = _mm256_andnot_si256(cmp0, ct0);
    let st1 = _mm256_andnot_si256(cmp0, ct1);
    let st2 = _mm256_andnot_si256(cmp0, ct2);
    let st3 = _mm256_andnot_si256(cmp0, ct3);
    let mut b;
    (z[0], b) = sub64_no_borrow(&z[0], &st0);
    (z[1], b) = sub64(&z[1], &st1, &b);
    (z[2], b) = sub64(&z[2], &st2, &b);
    let tmp = _mm256_sub_epi64(z[3], st3);
    z[3] = _mm256_sub_epi64(tmp, b);

    z
}

#[inline]
unsafe fn to_mont(a: [__m256i; 4]) -> [__m256i; 4] {
    let r_square_0 = _mm256_set_epi64x(
        1997599621687373223u64 as i64,
        1997599621687373223u64 as i64,
        1997599621687373223u64 as i64,
        1997599621687373223u64 as i64,
    );
    let r_square_1 = _mm256_set_epi64x(
        6052339484930628067u64 as i64,
        6052339484930628067u64 as i64,
        6052339484930628067u64 as i64,
        6052339484930628067u64 as i64,
    );
    let r_square_2 = _mm256_set_epi64x(
        10108755138030829701u64 as i64,
        10108755138030829701u64 as i64,
        10108755138030829701u64 as i64,
        10108755138030829701u64 as i64,
    );
    let r_square_3 = _mm256_set_epi64x(
        150537098327114917u64 as i64,
        150537098327114917u64 as i64,
        150537098327114917u64 as i64,
        150537098327114917u64 as i64,
    );
    let r: [__m256i; 4] = [r_square_0, r_square_1, r_square_2, r_square_3];
    _mul_generic(a, r)
}

#[inline]
unsafe fn from_mont(a: [__m256i; 4]) -> [__m256i; 4] {
    let ct0 = _mm256_set_epi64x(
        4891460686036598785i64,
        4891460686036598785i64,
        4891460686036598785i64,
        4891460686036598785i64,
    );
    let ct1 = _mm256_set_epi64x(
        2896914383306846353i64,
        2896914383306846353i64,
        2896914383306846353i64,
        2896914383306846353i64,
    );
    let ct2 = _mm256_set_epi64x(
        13281191951274694749u64 as i64,
        13281191951274694749u64 as i64,
        13281191951274694749u64 as i64,
        13281191951274694749u64 as i64,
    );
    let ct3 = _mm256_set_epi64x(
        3486998266802970665i64,
        3486998266802970665i64,
        3486998266802970665i64,
        3486998266802970665i64,
    );
    let ct4 = _mm256_set_epi64x(
        14042775128853446655u64 as i64,
        14042775128853446655u64 as i64,
        14042775128853446655u64 as i64,
        14042775128853446655u64 as i64,
    );

    let mut z: [__m256i; 4] = a;

    // m = z[0]n'[0] mod W
    let m = _mm256_mullo_epi64(z[0], ct4);
    let mut c = madd0(&m, &ct0, &z[0]);
    (c, z[0]) = madd2(&m, &ct1, &z[1], &c);
    (c, z[1]) = madd2(&m, &ct2, &z[2], &c);
    (c, z[2]) = madd2(&m, &ct3, &z[3], &c);
    z[3] = c;

    // m = z[0]n'[0] mod W
    let m = _mm256_mullo_epi64(z[0], ct4);
    let mut c = madd0(&m, &ct0, &z[0]);
    (c, z[0]) = madd2(&m, &ct1, &z[1], &c);
    (c, z[1]) = madd2(&m, &ct2, &z[2], &c);
    (c, z[2]) = madd2(&m, &ct3, &z[3], &c);
    z[3] = c;

    // m = z[0]n'[0] mod W
    let m = _mm256_mullo_epi64(z[0], ct4);
    let mut c = madd0(&m, &ct0, &z[0]);
    (c, z[0]) = madd2(&m, &ct1, &z[1], &c);
    (c, z[1]) = madd2(&m, &ct2, &z[2], &c);
    (c, z[2]) = madd2(&m, &ct3, &z[3], &c);
    z[3] = c;

    // m = z[0]n'[0] mod W
    let m = _mm256_mullo_epi64(z[0], ct4);
    let mut c = madd0(&m, &ct0, &z[0]);
    (c, z[0]) = madd2(&m, &ct1, &z[1], &c);
    (c, z[1]) = madd2(&m, &ct2, &z[2], &c);
    (c, z[2]) = madd2(&m, &ct3, &z[3], &c);
    z[3] = c;

    // if z > q --> z -= q
    let cmp0 = _mm256_cmpgt_epi64(ct0, z[0]);
    let cmp1 = _mm256_cmpeq_epi64(ct1, z[1]);
    let cmp0 = _mm256_and_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpgt_epi64(ct1, z[1]);
    let cmp0 = _mm256_or_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpeq_epi64(ct2, z[2]);
    let cmp0 = _mm256_and_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpgt_epi64(ct2, z[2]);
    let cmp0 = _mm256_or_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpeq_epi64(ct3, z[3]);
    let cmp0 = _mm256_and_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpgt_epi64(ct3, z[3]);
    let cmp0 = _mm256_or_si256(cmp0, cmp1);
    let st0 = _mm256_andnot_si256(cmp0, ct0);
    let st1 = _mm256_andnot_si256(cmp0, ct1);
    let st2 = _mm256_andnot_si256(cmp0, ct2);
    let st3 = _mm256_andnot_si256(cmp0, ct3);
    let mut b;
    (z[0], b) = sub64_no_borrow(&z[0], &st0);
    (z[1], b) = sub64(&z[1], &st1, &b);
    (z[2], b) = sub64(&z[2], &st2, &b);
    let tmp = _mm256_sub_epi64(z[3], st3);
    z[3] = _mm256_sub_epi64(tmp, b);

    z
}

#[inline]
unsafe fn ark(state: &mut [__m256i; 8], c: [[u64; 4]; 100], it: usize) {
    // 1st element
    let cc: [__m256i; 4] = [
        _mm256_set_epi64x(0, 0, 0, c[it][0] as i64),
        _mm256_set_epi64x(0, 0, 0, c[it][1] as i64),
        _mm256_set_epi64x(0, 0, 0, c[it][2] as i64),
        _mm256_set_epi64x(0, 0, 0, c[it][3] as i64),
    ];
    let mut ss: [__m256i; 4] = [state[0], state[1], state[2], state[3]];
    ss = _add_generic(ss, cc);
    state[0] = ss[0];
    state[1] = ss[1];
    state[2] = ss[2];
    state[3] = ss[3];

    // next 4 elements
    let cc: [__m256i; 4] = [
        _mm256_set_epi64x(
            c[it + 4][0] as i64,
            c[it + 3][0] as i64,
            c[it + 2][0] as i64,
            c[it + 1][0] as i64,
        ),
        _mm256_set_epi64x(
            c[it + 4][1] as i64,
            c[it + 3][1] as i64,
            c[it + 2][1] as i64,
            c[it + 1][1] as i64,
        ),
        _mm256_set_epi64x(
            c[it + 4][2] as i64,
            c[it + 3][2] as i64,
            c[it + 2][2] as i64,
            c[it + 1][2] as i64,
        ),
        _mm256_set_epi64x(
            c[it + 4][3] as i64,
            c[it + 3][3] as i64,
            c[it + 2][3] as i64,
            c[it + 1][3] as i64,
        ),
    ];
    let mut ss: [__m256i; 4] = [state[4], state[5], state[6], state[7]];
    ss = _add_generic(ss, cc);
    state[4] = ss[0];
    state[5] = ss[1];
    state[6] = ss[2];
    state[7] = ss[3];
}

#[inline]
unsafe fn mix(state: &mut [__m256i; 8], m: [[[u64; 4]; 5]; 5]) {
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let mut new_state: [__m256i; 8] = [zeros; 8];

    // s[0] -> new_state[0]
    let ss: [__m256i; 4] = [state[0], state[1], state[2], state[3]];
    let mm: [__m256i; 4] = [
        _mm256_set_epi64x(0, 0, 0, m[0][0][0] as i64),
        _mm256_set_epi64x(0, 0, 0, m[0][0][1] as i64),
        _mm256_set_epi64x(0, 0, 0, m[0][0][2] as i64),
        _mm256_set_epi64x(0, 0, 0, m[0][0][3] as i64),
    ];
    let mul = _mul_generic(mm, ss);
    let rr = _add_generic(new_state[0..4].try_into().unwrap(), mul);
    new_state[0] = rr[0];
    new_state[1] = rr[1];
    new_state[2] = rr[2];
    new_state[3] = rr[3];

    // s[1..4] -> new_state[0]
    let mut ss = [state[4], state[5], state[6], state[7]];
    for j in 1..5 {
        if j > 1 {
            ss[0] = _mm256_permute4x64_epi64(ss[0], 0x39);
            ss[1] = _mm256_permute4x64_epi64(ss[1], 0x39);
            ss[2] = _mm256_permute4x64_epi64(ss[2], 0x39);
            ss[3] = _mm256_permute4x64_epi64(ss[3], 0x39);
        }
        let mm: [__m256i; 4] = [
            _mm256_set_epi64x(0, 0, 0, m[j][0][0] as i64),
            _mm256_set_epi64x(0, 0, 0, m[j][0][1] as i64),
            _mm256_set_epi64x(0, 0, 0, m[j][0][2] as i64),
            _mm256_set_epi64x(0, 0, 0, m[j][0][3] as i64),
        ];
        let mul = _mul_generic(mm, ss);
        let rr = _add_generic(new_state[0..4].try_into().unwrap(), mul);
        new_state[0] = rr[0];
        new_state[1] = rr[1];
        new_state[2] = rr[2];
        new_state[3] = rr[3];
    }

    // s[0] -> new_state[1..4]
    let mm: [__m256i; 4] = [
        _mm256_set_epi64x(
            m[0][4][0] as i64,
            m[0][3][0] as i64,
            m[0][2][0] as i64,
            m[0][1][0] as i64,
        ),
        _mm256_set_epi64x(
            m[0][4][1] as i64,
            m[0][3][1] as i64,
            m[0][2][1] as i64,
            m[0][1][1] as i64,
        ),
        _mm256_set_epi64x(
            m[0][4][2] as i64,
            m[0][3][2] as i64,
            m[0][2][2] as i64,
            m[0][1][2] as i64,
        ),
        _mm256_set_epi64x(
            m[0][4][3] as i64,
            m[0][3][3] as i64,
            m[0][2][3] as i64,
            m[0][1][3] as i64,
        ),
    ];
    let mut ss: [__m256i; 4] = [state[0], state[1], state[2], state[3]];
    for j in 1..5 {
        if j > 1 {
            ss[0] = _mm256_permute4x64_epi64(ss[0], 0x93);
            ss[1] = _mm256_permute4x64_epi64(ss[1], 0x93);
            ss[2] = _mm256_permute4x64_epi64(ss[2], 0x93);
            ss[3] = _mm256_permute4x64_epi64(ss[3], 0x93);
        }
        let mul = _mul_generic(mm, ss);
        let rr = _add_generic(new_state[4..8].try_into().unwrap(), mul);
        new_state[4] = rr[0];
        new_state[5] = rr[1];
        new_state[6] = rr[2];
        new_state[7] = rr[3];
    }

    // s[1..4] -> new_state[1..4]
    for j in 1..5 {
        let mut sv4: [i64; 4] = [0; 4];
        let mut sv5: [i64; 4] = [0; 4];
        let mut sv6: [i64; 4] = [0; 4];
        let mut sv7: [i64; 4] = [0; 4];
        _mm256_storeu_si256(sv4.as_mut_ptr().cast::<__m256i>(), state[4]);
        _mm256_storeu_si256(sv5.as_mut_ptr().cast::<__m256i>(), state[5]);
        _mm256_storeu_si256(sv6.as_mut_ptr().cast::<__m256i>(), state[6]);
        _mm256_storeu_si256(sv7.as_mut_ptr().cast::<__m256i>(), state[7]);
        let k = j - 1;
        let ss = [
            _mm256_set_epi64x(sv4[k], sv4[k], sv4[k], sv4[k]),
            _mm256_set_epi64x(sv5[k], sv5[k], sv5[k], sv5[k]),
            _mm256_set_epi64x(sv6[k], sv6[k], sv6[k], sv6[k]),
            _mm256_set_epi64x(sv7[k], sv7[k], sv7[k], sv7[k]),
        ];
        let mm: [__m256i; 4] = [
            _mm256_set_epi64x(
                m[j][4][0] as i64,
                m[j][3][0] as i64,
                m[j][2][0] as i64,
                m[j][1][0] as i64,
            ),
            _mm256_set_epi64x(
                m[j][4][1] as i64,
                m[j][3][1] as i64,
                m[j][2][1] as i64,
                m[j][1][1] as i64,
            ),
            _mm256_set_epi64x(
                m[j][4][2] as i64,
                m[j][3][2] as i64,
                m[j][2][2] as i64,
                m[j][1][2] as i64,
            ),
            _mm256_set_epi64x(
                m[j][4][3] as i64,
                m[j][3][3] as i64,
                m[j][2][3] as i64,
                m[j][1][3] as i64,
            ),
        ];
        let mul = _mul_generic(mm, ss);
        let rr = _add_generic(new_state[4..8].try_into().unwrap(), mul);
        new_state[4] = rr[0];
        new_state[5] = rr[1];
        new_state[6] = rr[2];
        new_state[7] = rr[3];
    }

    for i in 0..8 {
        state[i] = new_state[i];
    }
}

#[allow(dead_code)]
fn print_state3(state: &[__m256i; 3]) {
    let mut a: [u64; 4] = [0; 4];
    println!("State3:");
    unsafe {
        _mm256_storeu_si256(a.as_mut_ptr().cast::<__m256i>(), state[0]);
        println!("{:?}", a);
        _mm256_storeu_si256(a.as_mut_ptr().cast::<__m256i>(), state[1]);
        println!("{:?}", a);
        _mm256_storeu_si256(a.as_mut_ptr().cast::<__m256i>(), state[2]);
        println!("{:?}", a);
    }
}

#[allow(dead_code)]
fn print_state4(state: &[__m256i; 4]) {
    let mut a: [u64; 4] = [0; 4];
    println!("State4:");
    unsafe {
        _mm256_storeu_si256(a.as_mut_ptr().cast::<__m256i>(), state[0]);
        println!("{:?}", a);
        _mm256_storeu_si256(a.as_mut_ptr().cast::<__m256i>(), state[1]);
        println!("{:?}", a);
        _mm256_storeu_si256(a.as_mut_ptr().cast::<__m256i>(), state[2]);
        println!("{:?}", a);
        _mm256_storeu_si256(a.as_mut_ptr().cast::<__m256i>(), state[3]);
        println!("{:?}", a);
    }
}

#[allow(dead_code)]
fn print_state8(state: &[__m256i; 8]) {
    let mut a: [u64; 4] = [0; 4];
    println!("State8:");
    unsafe {
        for i in 0..8 {
            _mm256_storeu_si256(a.as_mut_ptr().cast::<__m256i>(), state[i]);
            println!("{:?}", a);
        }
    }
}

#[allow(dead_code)]
fn print_state(state: &[ElementBN128; 5]) {
    println!("{:?}", state[0]);
    println!("{:?}", state[1]);
    println!("{:?}", state[2]);
    println!("{:?}", state[3]);
    println!("{:?}", state[4]);
    println!();
}

pub fn permute_bn128_avx(input: [u64; 12]) -> [u64; 12] {
    #[cfg(feature = "papi")]
    let mut event_set = init_papi();
    #[cfg(feature = "papi")]
    event_set.start().unwrap();

    const CT: usize = 5;
    const N_ROUNDS_F: usize = 8;
    const N_ROUNDS_P: usize = 60;

    unsafe {
        // load states
        let mut inp: [__m256i; 4] = [
            _mm256_set_epi64x(
                input[11] as i64,
                input[8] as i64,
                input[5] as i64,
                input[2] as i64,
            ),
            _mm256_set_epi64x(
                input[10] as i64,
                input[7] as i64,
                input[4] as i64,
                input[1] as i64,
            ),
            _mm256_set_epi64x(
                input[9] as i64,
                input[6] as i64,
                input[3] as i64,
                input[0] as i64,
            ),
            _mm256_set_epi64x(0i64, 0i64, 0i64, 0i64),
        ];

        // to mont
        inp = to_mont(inp);

        #[cfg(feature = "papi")]
        stop_papi(&mut event_set, "to_mont");
        #[cfg(feature = "papi")]
        event_set.start().unwrap();

        // start rounds
        let zeros = _mm256_set_epi64x(0, 0, 0, 0);
        let mut state: [__m256i; 8] = [zeros, zeros, zeros, zeros, inp[0], inp[1], inp[2], inp[3]];

        ark(&mut state, C, 0);

        #[cfg(feature = "papi")]
        stop_papi(&mut event_set, "first ark");

        /*
        let mut z = [0u64; 4];
        let z1 = [3650884469251175381u64, 0, 0, 0];
        _mm256_storeu_si256(z.as_mut_ptr().cast::<__m256i>(), state[0]);
        assert_eq!(z1, z);
        let z2 = [4312995929451917048u64, 0, 0, 0];
        _mm256_storeu_si256(z.as_mut_ptr().cast::<__m256i>(), state[1]);
        assert_eq!(z2, z);
        let z3 = [14528712943685515188u64, 0, 0, 0];
        _mm256_storeu_si256(z.as_mut_ptr().cast::<__m256i>(), state[2]);
        assert_eq!(z3, z);
        let z4 = [804645480652767018u64, 0, 0, 0];
        _mm256_storeu_si256(z.as_mut_ptr().cast::<__m256i>(), state[3]);
        assert_eq!(z4, z);
        let z5 = [14462745598712311877u64, 9965481966597437291u64, 5916123222076323011u64, 14423924459958803780u64];
        _mm256_storeu_si256(z.as_mut_ptr().cast::<__m256i>(), state[4]);
        assert_eq!(z5, z);
        let z6 = [7570161332469838584u64, 18440159137561926521u64, 7248986691198917743u64, 16755156072218033775u64];
        _mm256_storeu_si256(z.as_mut_ptr().cast::<__m256i>(), state[5]);
        assert_eq!(z6, z);
        let z7 = [12421518753342417017u64, 966430971004801851u64, 13841309536587625009u64, 14460935863064733763u64];
        _mm256_storeu_si256(z.as_mut_ptr().cast::<__m256i>(), state[6]);
        assert_eq!(z7, z);
        let z8 = [1527580982755995308u64, 1452659775731263630u64, 3308699589782081186u64, 2575827241589250587u64];
        _mm256_storeu_si256(z.as_mut_ptr().cast::<__m256i>(), state[7]);
        assert_eq!(z8, z);
        */

        #[cfg(feature = "papi")]
        event_set.start().unwrap();
        for i in 0..(N_ROUNDS_F / 2 - 1) {
            exp5state(&mut state);
            ark(&mut state, C, (i + 1) * CT);
            mix(&mut state, M);
        }
        #[cfg(feature = "papi")]
        stop_papi(&mut event_set, "half full rounds");

        #[cfg(feature = "papi")]
        event_set.start().unwrap();
        exp5state(&mut state);
        #[cfg(feature = "papi")]
        stop_papi(&mut event_set, "exp5state");

        #[cfg(feature = "papi")]
        event_set.start().unwrap();
        ark(&mut state, C, (N_ROUNDS_F / 2) * CT);
        #[cfg(feature = "papi")]
        stop_papi(&mut event_set, "ark");

        #[cfg(feature = "papi")]
        event_set.start().unwrap();
        mix(&mut state, P);
        #[cfg(feature = "papi")]
        stop_papi(&mut event_set, "mix");

        // println!("After 1st rounds:");
        // print_state8(&state);

        #[cfg(feature = "papi")]
        event_set.start().unwrap();
        // switch to classic representation
        let mut cstate = [ElementBN128::zero(); 5];
        let mut tmps = [[0u64; 4]; 4];
        let mut tmpv = [0u64; 4];
        _mm256_storeu_si256(tmpv.as_mut_ptr().cast::<__m256i>(), state[0]);
        tmps[0][0] = tmpv[0];
        _mm256_storeu_si256(tmpv.as_mut_ptr().cast::<__m256i>(), state[1]);
        tmps[0][1] = tmpv[0];
        _mm256_storeu_si256(tmpv.as_mut_ptr().cast::<__m256i>(), state[2]);
        tmps[0][2] = tmpv[0];
        _mm256_storeu_si256(tmpv.as_mut_ptr().cast::<__m256i>(), state[3]);
        tmps[0][3] = tmpv[0];
        cstate[0] = ElementBN128::new(tmps[0]);
        _mm256_storeu_si256(tmpv.as_mut_ptr().cast::<__m256i>(), state[4]);
        tmps[0][0] = tmpv[0];
        tmps[1][0] = tmpv[1];
        tmps[2][0] = tmpv[2];
        tmps[3][0] = tmpv[3];
        _mm256_storeu_si256(tmpv.as_mut_ptr().cast::<__m256i>(), state[5]);
        tmps[0][1] = tmpv[0];
        tmps[1][1] = tmpv[1];
        tmps[2][1] = tmpv[2];
        tmps[3][1] = tmpv[3];
        _mm256_storeu_si256(tmpv.as_mut_ptr().cast::<__m256i>(), state[6]);
        tmps[0][2] = tmpv[0];
        tmps[1][2] = tmpv[1];
        tmps[2][2] = tmpv[2];
        tmps[3][2] = tmpv[3];
        _mm256_storeu_si256(tmpv.as_mut_ptr().cast::<__m256i>(), state[7]);
        tmps[0][3] = tmpv[0];
        tmps[1][3] = tmpv[1];
        tmps[2][3] = tmpv[2];
        tmps[3][3] = tmpv[3];
        cstate[1] = ElementBN128::new(tmps[0]);
        cstate[2] = ElementBN128::new(tmps[1]);
        cstate[3] = ElementBN128::new(tmps[2]);
        cstate[4] = ElementBN128::new(tmps[3]);

        // println!("After 1st rounds:");
        // print_state(&cstate);

        for i in 0..N_ROUNDS_P {
            cstate[0].exp5();
            let cc = ElementBN128::new(C[(N_ROUNDS_F / 2 + 1) * CT + i]);
            cstate[0].add(cstate[0], cc);

            let mut mul = ElementBN128::zero();
            let mut new_state0 = ElementBN128::zero();
            for j in 0..CT {
                let ss = ElementBN128::new(S[(CT * 2 - 1) * i + j]);
                mul.mul(ss, cstate[j]);
                new_state0.add(new_state0, mul);
            }

            for k in 1..CT {
                let ss = ElementBN128::new(S[(CT * 2 - 1) * i + CT + k - 1]);
                mul.set_zero();
                mul.mul(cstate[0], ss);
                cstate[k].add(cstate[k], mul);
            }
            cstate[0] = new_state0;
        }
        #[cfg(feature = "papi")]
        stop_papi(&mut event_set, "partial rounds");
        // println!("After middle rounds:");
        // print_state(&cstate);

        #[cfg(feature = "papi")]
        event_set.start().unwrap();
        // switch to AVX
        state = [
            _mm256_set_epi64x(0i64, 0i64, 0i64, cstate[0].z[0] as i64),
            _mm256_set_epi64x(0i64, 0i64, 0i64, cstate[0].z[1] as i64),
            _mm256_set_epi64x(0i64, 0i64, 0i64, cstate[0].z[2] as i64),
            _mm256_set_epi64x(0i64, 0i64, 0i64, cstate[0].z[3] as i64),
            _mm256_set_epi64x(
                cstate[4].z[0] as i64,
                cstate[3].z[0] as i64,
                cstate[2].z[0] as i64,
                cstate[1].z[0] as i64,
            ),
            _mm256_set_epi64x(
                cstate[4].z[1] as i64,
                cstate[3].z[1] as i64,
                cstate[2].z[1] as i64,
                cstate[1].z[1] as i64,
            ),
            _mm256_set_epi64x(
                cstate[4].z[2] as i64,
                cstate[3].z[2] as i64,
                cstate[2].z[2] as i64,
                cstate[1].z[2] as i64,
            ),
            _mm256_set_epi64x(
                cstate[4].z[3] as i64,
                cstate[3].z[3] as i64,
                cstate[2].z[3] as i64,
                cstate[1].z[3] as i64,
            ),
        ];

        // println!("After middle rounds:");
        // print_state8(&state);

        for i in 0..(N_ROUNDS_F / 2 - 1) {
            exp5state(&mut state);
            ark(
                &mut state,
                C,
                (N_ROUNDS_F / 2 + 1) * CT + N_ROUNDS_P + i * CT,
            );
            mix(&mut state, M);
        }
        exp5state(&mut state);
        mix(&mut state, M);
        #[cfg(feature = "papi")]
        stop_papi(&mut event_set, "half full rounds");

        // println!("After all rounds:");
        // print_state8(&state);

        #[cfg(feature = "papi")]
        event_set.start().unwrap();

        let ss0 = from_mont(state[0..4].try_into().unwrap());
        let ss1 = from_mont(state[4..8].try_into().unwrap());

        // println!("After from_mont rounds:");
        // print_state4(&ss0);
        // print_state4(&ss1);

        let mut out: [u64; 12] = [0; 12];
        _mm256_storeu_si256(tmpv.as_mut_ptr().cast::<__m256i>(), ss0[0]);
        out[2] = tmpv[0];
        _mm256_storeu_si256(tmpv.as_mut_ptr().cast::<__m256i>(), ss0[1]);
        out[1] = tmpv[0];
        _mm256_storeu_si256(tmpv.as_mut_ptr().cast::<__m256i>(), ss0[2]);
        out[0] = tmpv[0];
        _mm256_storeu_si256(tmpv.as_mut_ptr().cast::<__m256i>(), ss1[0]);
        out[5] = tmpv[0];
        out[8] = tmpv[1];
        out[11] = tmpv[2];
        _mm256_storeu_si256(tmpv.as_mut_ptr().cast::<__m256i>(), ss1[1]);
        out[4] = tmpv[0];
        out[7] = tmpv[1];
        out[10] = tmpv[2];
        _mm256_storeu_si256(tmpv.as_mut_ptr().cast::<__m256i>(), ss1[2]);
        out[3] = tmpv[0];
        out[6] = tmpv[1];
        out[9] = tmpv[2];
        for i in 0..12 {
            if out[i] >= 0xFFFFFFFF00000001u64 {
                out[i] = out[i] - 0xFFFFFFFF00000001u64;
            }
        }
        #[cfg(feature = "papi")]
        stop_papi(&mut event_set, "from_mont");

        out
    }
}

#[cfg(test)]
mod tests {
    use core::arch::x86_64::*;

    use anyhow::Result;

    use super::{add64, sub64};

    #[test]
    fn test_bn128_avx() -> Result<()> {
        unsafe {
            let ct1 = _mm256_set_epi64x(
                2896914383306846353i64,
                2896914383306846353i64,
                2896914383306846353i64,
                2896914383306846353i64,
            );
            let ct2 = _mm256_set_epi64x(
                13281191951274694749u64 as i64,
                13281191951274694749u64 as i64,
                13281191951274694749u64 as i64,
                13281191951274694749u64 as i64,
            );
            let exp: [u64; 4] = [
                0xE0842DFEFB3AC8EEu64,
                0xE0842DFEFB3AC8EEu64,
                0xE0842DFEFB3AC8EEu64,
                0xE0842DFEFB3AC8EEu64,
            ];

            let r = _mm256_add_epi64(ct1, ct2);
            let mut vr: [u64; 4] = [0; 4];
            _mm256_storeu_si256(vr.as_mut_ptr().cast::<__m256i>(), r);
            println!("{:X?}", vr);
            assert_eq!(vr, exp);
        }
        Ok(())
    }

    #[test]
    fn test_bn128_add64() -> Result<()> {
        unsafe {
            let a = _mm256_set_epi64x(
                0xFFFFFFFFFFFFFFFFu64 as i64,
                0x0FFFFFFFFFFFFFFF as i64,
                0xFFFFFFFFFFFFFFFFu64 as i64,
                0xFFFFFFFFFFFFFFFFu64 as i64,
            );
            let b = _mm256_set_epi64x(
                0xFFFFFFFFFFFFFFFFu64 as i64,
                0x0FFFFFFFFFFFFFFF as i64,
                0x0i64,
                0x1i64,
            );
            let cin = _mm256_set_epi64x(0, 0, 0, 0);
            let res = [
                0u64,
                0xFFFFFFFFFFFFFFFFu64,
                0x1FFFFFFFFFFFFFFEu64,
                0xFFFFFFFFFFFFFFFEu64,
            ];

            let cout = [1u64, 0u64, 0u64, 1u64];

            let mut v: [u64; 4] = [0; 4];
            let (r, c) = add64(&a, &b, &cin);

            _mm256_storeu_si256(v.as_mut_ptr().cast::<__m256i>(), r);
            println!(" Res: {:X?}", v);
            assert_eq!(v, res);
            _mm256_storeu_si256(v.as_mut_ptr().cast::<__m256i>(), c);
            println!("Cout: {:X?}", v);
            assert_eq!(v, cout);
        }
        Ok(())
    }

    #[test]
    fn test_bn128_sub64() -> Result<()> {
        unsafe {
            let a = _mm256_set_epi64x(4i64, 7i64, 0xFFFFFFFFFFFFFFFFu64 as i64, 0x0u64 as i64);
            let b = _mm256_set_epi64x(7i64, 4i64, 0x0i64, 0xFFFFFFFFFFFFFFFFu64 as i64);
            let bin = _mm256_set_epi64x(0, 0, 0, 0);

            let res = [0x1u64, 0xFFFFFFFFFFFFFFFFu64, 3u64, 0xFFFFFFFFFFFFFFFDu64];

            let bout = [1u64, 0u64, 0u64, 1u64];

            let mut v: [u64; 4] = [0; 4];
            let (c1, c2) = sub64(&a, &b, &bin);
            _mm256_storeu_si256(v.as_mut_ptr().cast::<__m256i>(), c1);
            println!("Res: {:X?}", v);
            println!("Exp: {:X?}", res);
            assert_eq!(v, res);
            _mm256_storeu_si256(v.as_mut_ptr().cast::<__m256i>(), c2);
            println!("Cout: {:X?}", v);
            assert_eq!(v, bout);
        }
        Ok(())
    }
}
