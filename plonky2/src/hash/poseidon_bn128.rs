#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
// include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use super::hash_types::HashOutTarget;
use super::poseidon::PoseidonPermutation;
use crate::field::extension::quadratic::QuadraticExtension;
use crate::field::extension::Extendable;
use crate::field::goldilocks_field::GoldilocksField;
#[cfg(target_feature = "avx2")]
use crate::hash::arch::x86_64::poseidon_bn128_avx2::permute_bn128_avx;
use crate::hash::hash_types::{HashOut, RichField};
use crate::hash::hashing::{compress, hash_n_to_hash_no_pad, PlonkyPermutation};
use crate::hash::poseidon::{PoseidonHash, SPONGE_RATE, SPONGE_WIDTH};
#[cfg(not(target_feature = "avx2"))]
use crate::hash::poseidon_bn128_ops::PoseidonBN128NativePermutation;
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::{AlgebraicHasher, GenericConfig, Hasher, HasherType};

#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct PoseidonBN128Permutation<F> {
    state: [F; SPONGE_WIDTH],
}

impl<F: RichField> Eq for PoseidonBN128Permutation<F> {}

impl<F: RichField> AsRef<[F]> for PoseidonBN128Permutation<F> {
    fn as_ref(&self) -> &[F] {
        &self.state
    }
}

impl<F: RichField> PlonkyPermutation<F> for PoseidonBN128Permutation<F> {
    const RATE: usize = SPONGE_RATE;
    const WIDTH: usize = SPONGE_WIDTH;

    fn new<I: IntoIterator<Item = F>>(elts: I) -> Self {
        let mut perm = Self {
            state: [F::default(); SPONGE_WIDTH],
        };
        perm.set_from_iter(elts, 0);
        perm
    }

    fn set_elt(&mut self, elt: F, idx: usize) {
        self.state[idx] = elt;
    }

    fn set_from_slice(&mut self, elts: &[F], start_idx: usize) {
        let begin = start_idx;
        let end = start_idx + elts.len();
        self.state[begin..end].copy_from_slice(elts);
    }

    fn set_from_iter<I: IntoIterator<Item = F>>(&mut self, elts: I, start_idx: usize) {
        for (s, e) in self.state[start_idx..].iter_mut().zip(elts) {
            *s = e;
        }
    }

    /*
    // Go Wrapper - 33% slower than Rust version below
    fn permute(&mut self) {
        assert_eq!(SPONGE_WIDTH, 12);
        // println!("start permute............");
        unsafe {
            let h = permute(
                self.state[0].to_canonical_u64(),
                self.state[1].to_canonical_u64(),
                self.state[2].to_canonical_u64(),
                self.state[3].to_canonical_u64(),
                self.state[4].to_canonical_u64(),
                self.state[5].to_canonical_u64(),
                self.state[6].to_canonical_u64(),
                self.state[7].to_canonical_u64(),
                self.state[8].to_canonical_u64(),
                self.state[9].to_canonical_u64(),
                self.state[10].to_canonical_u64(),
                self.state[11].to_canonical_u64(),
            );

            let permute_output = [
                F::from_canonical_u64(if h.r0 >= F::ORDER {
                    h.r0 - F::ORDER
                } else {
                    h.r0
                }),
                F::from_canonical_u64(if h.r1 >= F::ORDER {
                    h.r1 - F::ORDER
                } else {
                    h.r1
                }),
                F::from_canonical_u64(if h.r2 >= F::ORDER {
                    h.r2 - F::ORDER
                } else {
                    h.r2
                }),
                F::from_canonical_u64(if h.r3 >= F::ORDER {
                    h.r3 - F::ORDER
                } else {
                    h.r3
                }),
                F::from_canonical_u64(if h.r4 >= F::ORDER {
                    h.r4 - F::ORDER
                } else {
                    h.r4
                }),
                F::from_canonical_u64(if h.r5 >= F::ORDER {
                    h.r5 - F::ORDER
                } else {
                    h.r5
                }),
                F::from_canonical_u64(if h.r6 >= F::ORDER {
                    h.r6 - F::ORDER
                } else {
                    h.r6
                }),
                F::from_canonical_u64(if h.r7 >= F::ORDER {
                    h.r7 - F::ORDER
                } else {
                    h.r7
                }),
                F::from_canonical_u64(if h.r8 >= F::ORDER {
                    h.r8 - F::ORDER
                } else {
                    h.r8
                }),
                F::from_canonical_u64(if h.r9 >= F::ORDER {
                    h.r9 - F::ORDER
                } else {
                    h.r9
                }),
                F::from_canonical_u64(if h.r10 >= F::ORDER {
                    h.r10 - F::ORDER
                } else {
                    h.r10
                }),
                F::from_canonical_u64(if h.r11 >= F::ORDER {
                    h.r11 - F::ORDER
                } else {
                    h.r11
                }),
            ];
            self.set_from_slice(&permute_output, 0)
        }
    }
    */

    fn permute(&mut self) {
        assert_eq!(SPONGE_WIDTH, 12);
        let su64: [u64; 12] = [
            self.state[0].to_canonical_u64(),
            self.state[1].to_canonical_u64(),
            self.state[2].to_canonical_u64(),
            self.state[3].to_canonical_u64(),
            self.state[4].to_canonical_u64(),
            self.state[5].to_canonical_u64(),
            self.state[6].to_canonical_u64(),
            self.state[7].to_canonical_u64(),
            self.state[8].to_canonical_u64(),
            self.state[9].to_canonical_u64(),
            self.state[10].to_canonical_u64(),
            self.state[11].to_canonical_u64(),
        ];

        #[cfg(not(target_feature = "avx2"))]
        let p: PoseidonBN128NativePermutation<F> = Default::default();
        #[cfg(not(target_feature = "avx2"))]
        let out = p.permute_fn(su64);
        #[cfg(target_feature = "avx2")]
        let out = permute_bn128_avx(su64);

        let permute_output = [
            F::from_canonical_u64(out[0]),
            F::from_canonical_u64(out[1]),
            F::from_canonical_u64(out[2]),
            F::from_canonical_u64(out[3]),
            F::from_canonical_u64(out[4]),
            F::from_canonical_u64(out[5]),
            F::from_canonical_u64(out[6]),
            F::from_canonical_u64(out[7]),
            F::from_canonical_u64(out[8]),
            F::from_canonical_u64(out[9]),
            F::from_canonical_u64(out[10]),
            F::from_canonical_u64(out[11]),
        ];

        self.set_from_slice(&permute_output, 0)
    }

    fn squeeze(&self) -> &[F] {
        &self.state[..Self::RATE]
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PoseidonBN128Hash;
impl<F: RichField> Hasher<F> for PoseidonBN128Hash {
    const HASHER_TYPE: HasherType = HasherType::PoseidonBN128;
    const HASH_SIZE: usize = 4 * 8;
    type Hash = HashOut<F>;
    type Permutation = PoseidonBN128Permutation<F>;

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        // println!("PoseidonBN128Hash hash_no_pad");
        hash_n_to_hash_no_pad::<F, Self::Permutation>(input)
    }

    fn hash_public_inputs(input: &[F]) -> Self::Hash {
        println!("PoseidonBN128Hash hash public inputs");
        PoseidonHash::hash_no_pad(input)
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        // println!("PoseidonBN128Hash two_to_one");
        compress::<F, Self::Permutation>(left, right)
    }
}

// TODO: this is a work around. Still use Goldilocks based Poseidon for algebraic PoseidonBN128Hash.
impl<F: RichField> AlgebraicHasher<F> for PoseidonBN128Hash {
    type AlgebraicPermutation = PoseidonPermutation<Target>;

    fn permute_swapped<const D: usize>(
        inputs: Self::AlgebraicPermutation,
        swap: BoolTarget,
        builder: &mut CircuitBuilder<F, D>,
    ) -> Self::AlgebraicPermutation
    where
        F: RichField + Extendable<D>,
    {
        PoseidonHash::permute_swapped(inputs, swap, builder)
    }

    fn public_inputs_hash<const D: usize>(
        inputs: Vec<Target>,
        builder: &mut CircuitBuilder<F, D>,
    ) -> HashOutTarget
    where
        F: RichField + Extendable<D>,
    {
        PoseidonHash::public_inputs_hash(inputs, builder)
    }
}

/// Configuration using Poseidon over the Goldilocks field.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct PoseidonBN128GoldilocksConfig;

impl GenericConfig<2> for PoseidonBN128GoldilocksConfig {
    type F = GoldilocksField;
    type FE = QuadraticExtension<Self::F>;
    type Hasher = PoseidonBN128Hash;
    type InnerHasher = PoseidonBN128Hash;
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use plonky2_field::types::Field;

    use super::PoseidonBN128Hash;
    use crate::hash::poseidon::PoseidonHash;
    use crate::plonk::config::{GenericConfig, GenericHashOut, Hasher, PoseidonGoldilocksConfig};

    #[test]
    fn test_poseidon_bn128_hash_no_pad() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let mut v = Vec::new();
        v.push(F::from_canonical_u64(8917524657281059100u64));
        v.push(F::from_canonical_u64(13029010200779371910u64));
        v.push(F::from_canonical_u64(16138660518493481604u64));
        v.push(F::from_canonical_u64(17277322750214136960u64));
        v.push(F::from_canonical_u64(1441151880423231822u64));
        let h = PoseidonBN128Hash::hash_no_pad(&v);
        assert_eq!(h.elements[0].0, 16736853722845225729u64);
        assert_eq!(h.elements[1].0, 1446699130810517790u64);
        assert_eq!(h.elements[2].0, 15445626857806971868u64);
        assert_eq!(h.elements[3].0, 6331160477881736675u64);

        Ok(())
    }

    #[test]
    fn test_poseidon_bn128_two_to_one() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let left: [u8; 32] = [
            1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5,
            6, 7, 8,
        ];
        let right: [u8; 32] = [
            8, 9, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5,
            6, 7, 1,
        ];

        let h = PoseidonBN128Hash::two_to_one(
            GenericHashOut::<F>::from_bytes(&left),
            GenericHashOut::<F>::from_bytes(&right),
        );

        assert_eq!(h.elements[0].0, 5894400909438531414u64);
        assert_eq!(h.elements[1].0, 4814851992117646301u64);
        assert_eq!(h.elements[2].0, 17814584260098324190u64);
        assert_eq!(h.elements[3].0, 15859500576163309036u64);

        Ok(())
    }

    #[test]
    fn test_poseidon_bn128_hash_public_inputs_same_as_poseidon_hash() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let mut v = Vec::new();
        v.push(F::from_canonical_u64(8917524657281059100u64));
        v.push(F::from_canonical_u64(13029010200779351910u64));
        v.push(F::from_canonical_u64(16138660518493481604u64));
        v.push(F::from_canonical_u64(17277322750214136960u64));
        v.push(F::from_canonical_u64(1441151880423231811u64));

        let h = PoseidonBN128Hash::hash_public_inputs(&v);

        assert_eq!(h.elements[0].0, 2325439551141788444);
        assert_eq!(h.elements[1].0, 15244397589056680708);
        assert_eq!(h.elements[2].0, 5900587506047513594);
        assert_eq!(h.elements[3].0, 7217031981798124005);

        // let inputs = inputs.iter().map(|x| F::from_canonical_u64(*x)).collect::<Vec<F>>();
        let poseidon_hash = PoseidonHash::hash_no_pad(&v);
        assert_eq!(h, poseidon_hash);
        Ok(())
    }
}
