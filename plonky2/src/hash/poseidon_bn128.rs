
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use crate::field::extension::quadratic::QuadraticExtension;
use crate::field::extension::Extendable;
use crate::field::goldilocks_field::GoldilocksField;

use crate::hash::hash_types::{HashOut, RichField};
use crate::hash::hashing::{compress, hash_n_to_hash_no_pad, PlonkyPermutation};
use crate::hash::poseidon::{PoseidonHash, SPONGE_RATE, SPONGE_WIDTH};
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::{AlgebraicHasher, GenericConfig, Hasher, HasherType};

use super::poseidon::PoseidonPermutation;

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

    fn permute(&mut self) {
        assert_eq!(SPONGE_WIDTH, 12);
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
        hash_n_to_hash_no_pad::<F, Self::Permutation>(input)
    }

    // fn hash_public_inputs(input: &[F]) -> Self::Hash {
    //     PoseidonHash::hash_no_pad(input)
    // }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        compress::<F, Self::Permutation>(left, right)
    }
}

// impl<F: RichField> AlgebraicHasher<F> for PoseidonBN128Hash {
//     type AlgebraicPermutation = PoseidonBN128Permutation<Target>;

//     fn permute_swapped<const D: usize>(
//         inputs: Self::AlgebraicPermutation,
//         swap: BoolTarget,
//         builder: &mut CircuitBuilder<F, D>,
//     ) -> Self::AlgebraicPermutation
//     where
//         F: RichField + Extendable<D>,
//     {
//         let gate_type = PoseidonGate::<F, D>::new();
//         let gate = builder.add_gate(gate_type, vec![]);

//         let swap_wire = PoseidonGate::<F, D>::WIRE_SWAP;
//         let swap_wire = Target::wire(gate, swap_wire);
//         builder.connect(swap.target, swap_wire);

//         // Route input wires.
//         let inputs = inputs.as_ref();
//         for i in 0..SPONGE_WIDTH {
//             let in_wire = PoseidonGate::<F, D>::wire_input(i);
//             let in_wire = Target::wire(gate, in_wire);
//             builder.connect(inputs[i], in_wire);
//         }

//         // Collect output wires.
//         Self::AlgebraicPermutation::new(
//             (0..SPONGE_WIDTH).map(|i| Target::wire(gate, PoseidonGate::<F, D>::wire_output(i))),
//         )
//     }
// }


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
    // fn public_inputs_hash<const D: usize>(
    //     inputs: Vec<Target>,
    //     builder: &mut CircuitBuilder<F, D>,
    // ) -> HashOutTarget
    // where
    //     F: RichField + Extendable<D>,
    // {
    //     PoseidonHash::public_inputs_hash(inputs, builder)
    // }
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
    use crate::plonk::config::{GenericConfig, Hasher, PoseidonGoldilocksConfig};
    use crate::hash::poseidon_bn128::PoseidonBN128Hash;

    #[test]
    fn test_poseidon_bn128() -> Result<()> {
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
}