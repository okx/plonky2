#![allow(incomplete_features)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::needless_range_loop)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![feature(specialization)]
#![cfg_attr(not(test), no_std)]
#![cfg(not(test))]
extern crate alloc;

pub(crate) mod arch;

pub mod batch_util;
pub mod cosets;
pub mod extension;
pub mod fft;
pub mod goldilocks_extensions;
pub mod goldilocks_field;
pub mod interpolation;
pub mod ops;
pub mod packable;
pub mod packed;
pub mod polynomial;
pub mod secp256k1_base;
pub mod secp256k1_scalar;
pub mod types;
pub mod zero_poly_coset;

#[cfg(test)]
mod field_testing;

#[cfg(test)]
mod prime_field_testing;

#[cfg(feature = "precompile")]
include!(concat!(env!("OUT_DIR"), "/goldilock_root_of_unity.rs"));

#[cfg(feature = "precompile")]
use std::collections::HashMap;

#[cfg(feature = "precompile")]
use fft::FftRootTable;
#[cfg(feature = "precompile")]
use goldilocks_field::GoldilocksField;
#[cfg(feature = "precompile")]
use lazy_static::lazy_static;
#[cfg(feature = "precompile")]
use plonky2_util::pre_compute::{PRE_COMPUTE_END, PRE_COMPUTE_START};

#[cfg(feature = "precompile")]
lazy_static! {
    pub static ref PRE_COMPUTE_ROOT_TABLES: HashMap<usize, FftRootTable<GoldilocksField>> = {
        let mut map = HashMap::new();

        let mut offset =  0;
        for lg_n in PRE_COMPUTE_START..=PRE_COMPUTE_END {

            // let offset_lgn = lg_n -  PRE_COMPUTE_START;
            let mut root_table = Vec::with_capacity(lg_n);
            for lg_m in 1..=lg_n {
                let half_m = 1 << (lg_m - 1);
                let  size_to_take =
                if half_m > 1 {
                    half_m
                } else {
                    2
                };
                let tables = PRE_COMPILED[offset..(offset+size_to_take)]
                .iter()
                .map(|x: &u64| GoldilocksField(*x)).collect::<Vec<GoldilocksField>>();
                root_table.push(tables);
                offset+= size_to_take;

            }
            map.insert(lg_n, root_table);
        }
        map
    };
}

#[cfg(test)]
mod test {

    #[cfg(feature = "precompile")]
    #[test]
    fn test_pre_compute() {
        for lgn_size in (PRE_COMPUTE_START..=PRE_COMPUTE_END) {
            let ret = PRE_COMPUTE_ROOT_TABLES.get(&lgn_size).unwrap();
            assert_eq!(ret.len(), lgn_size);
            for i in (0..ret.len()) {
                let sub = &ret[i];
                assert_eq!(sub[0].0, 0);
                assert_eq!(sub.len(), if i == 0 { 2 } else { 1 << i })
            }
        }
    }
}
