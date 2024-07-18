#[cfg(target_feature = "neon")]
pub(crate) mod poseidon_goldilocks_neon;

#[cfg(target_feature = "sve")]
pub(crate) mod poseidon_goldilocks_sve;
