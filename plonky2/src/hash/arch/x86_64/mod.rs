// // Requires:
// // - AVX2
// // - BMI2 (for MULX and SHRX)
// #[cfg(all(target_feature = "avx2", target_feature = "bmi2"))]
#[cfg(target_feature = "avx2")]
pub mod goldilocks_avx2;
#[cfg(target_feature = "avx512dq")]
pub mod goldilocks_avx512;
#[cfg(target_feature = "avx2")]
pub mod poseidon2_goldilocks_avx2;
#[cfg(target_feature = "avx2")]
pub mod poseidon_bn128_avx2;
#[cfg(target_feature = "avx2")]
pub mod poseidon_goldilocks_avx2;
#[cfg(all(target_feature = "avx2", target_feature = "avx512dq"))]
pub mod poseidon_goldilocks_avx512;
