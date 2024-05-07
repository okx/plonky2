use std::fmt::Debug;

use unroll::unroll_for_loops;

use super::hash_types::{HashOut, RichField};
use super::hashing::{compress, hash_n_to_hash_no_pad, PlonkyPermutation};
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::PrimeField64;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::{Hasher, HasherType};

// change these values and disable `default-sponge-params` feature if it is needed to change the
// default sponge parameters
#[cfg(not(feature = "default-sponge-params"))]
const CUSTOM_SPONGE_RATE: usize = 8;
#[cfg(not(feature = "default-sponge-params"))]
const CUSTOM_SPONGE_CAPACITY: usize = 4;

/// This constant describes the number of elements in the outer part of the cryptographic sponge
/// function.
#[cfg(feature = "default-sponge-params")]
pub const SPONGE_RATE: usize = 8;
/// This constant describes the number of elements in the inner part of the cryptographic sponge
/// function.
#[cfg(feature = "default-sponge-params")]
pub const SPONGE_CAPACITY: usize = 4;

#[cfg(not(feature = "default-sponge-params"))]
pub const SPONGE_RATE: usize = CUSTOM_SPONGE_RATE;
#[cfg(not(feature = "default-sponge-params"))]
pub const SPONGE_CAPACITY: usize = CUSTOM_SPONGE_CAPACITY;

/// This is the number of elements which constitute the state of the internal permutation and the
/// cryptographic sponge function built from this permutation.
pub const SPONGE_WIDTH: usize = SPONGE_RATE + SPONGE_CAPACITY;
/// Number of state elements involved in the `Bars` layer
pub const NUM_BARS: usize = 4;

// The number of full rounds and partial rounds is given by the
// calc_round_numbers.py script. They happen to be the same for both
// width 8 and width 12 with s-box x^7.
//
// NB: Changing any of these values will require regenerating all of
// the precomputed constant arrays in this file.
/// Number of rounds in Monolith permutations
pub const N_ROUNDS: usize = 6;
/// Bit-size of the domain of the lookup function applied in the `Bars` layer: a state element is
/// split in limbs of `LOOKUP_BITS` bits, and the lookup function is applied to each limb.
pub const LOOKUP_BITS: usize = 8;
/// Size of the domain of the lookup function applied in the `Bars` layer
pub const LOOKUP_SIZE: usize = 1 << LOOKUP_BITS;
/// Number of limbs necessary to represent a 64-bit state element
pub const LOOKUP_NUM_LIMBS: usize = 64 / LOOKUP_BITS;

#[inline]
pub(crate) fn split(x: u128) -> (u64, u32) {
    (x as u64, (x >> 64) as u32)
}

// helper function to compute concrete layer. The function requires to provide a buffer with
// `SPONGE_WIDTH` elements initialized to 0 to compute the outcome of the layer
#[inline(always)]
#[unroll_for_loops]
fn concrete_u128_with_tmp_buffer<M: Monolith>(
    state_u128: &[u128; SPONGE_WIDTH],
    round_constants: &[u64; SPONGE_WIDTH],
    res: &mut [u128; SPONGE_WIDTH],
) {
    for row in 0..SPONGE_WIDTH {
        for (column, input) in state_u128.iter().enumerate() {
            res[row] += *input * (M::MAT_12[row][column] as u128);
        }
        res[row] += round_constants[row] as u128;
        res[row] = M::from_noncanonical_u96(split(res[row])).to_noncanonical_u64() as u128;
    }
}
/// `Monolith` trait provides all the functions necessary to perform a Monolith permutation
pub trait Monolith: PrimeField64 {
    // Static data
    /// Number of round constants employed in a full Monolith permutation
    const N_ROUND_CONSTANTS: usize = SPONGE_WIDTH * (N_ROUNDS + 1);
    /// All the round constants employed in a full Monolith permutation
    const ROUND_CONSTANTS: [[u64; SPONGE_WIDTH]; N_ROUNDS + 1];
    /// This constant contains the first row of a circulant `SPONGE_WIDTH x SPONGE_WIDTH` MDS matrix
    /// M. All of the remaining rows of M are rotations of this constant vector. A multiplication
    /// by M is used in the affine layer of Monolith.
    const MAT_12: [[u64; SPONGE_WIDTH]; SPONGE_WIDTH];

    /// Compute the "Bar" component
    /// element is split in (16-bit lookups, analogous for 8-bit lookups):
    /// [x_3 || x_2 || x_1 || x_0], where x_i is 16 bits large
    /// element = 2^48 * x_3 + 2^32 * x_2 + 2^16 * x_1 + x_0
    /// Use lookups on x_3, x_2, x_1, x_0 and obtain y_3, y_2, y_1, y_0
    /// [y_3 || y_2 || y_1 || y_0], where y_i is 16 bits large
    /// Output y is set such that y = 2^48 * x_3 + 2^32 * x_2 + 2^16 * x_1 + x_0
    #[inline(always)]
    fn bar_64(limb: u64) -> u64 {
        match LOOKUP_BITS {
            8 => {
                let limbl1 =
                    ((!limb & 0x8080808080808080) >> 7) | ((!limb & 0x7F7F7F7F7F7F7F7F) << 1); // Left rotation by 1
                let limbl2 =
                    ((limb & 0xC0C0C0C0C0C0C0C0) >> 6) | ((limb & 0x3F3F3F3F3F3F3F3F) << 2); // Left rotation by 2
                let limbl3 =
                    ((limb & 0xE0E0E0E0E0E0E0E0) >> 5) | ((limb & 0x1F1F1F1F1F1F1F1F) << 3); // Left rotation by 3

                // y_i = x_i + (1 + x_{i+1}) * x_{i+2} * x_{i+3}
                let tmp = limb ^ limbl1 & limbl2 & limbl3;
                ((tmp & 0x8080808080808080) >> 7) | ((tmp & 0x7F7F7F7F7F7F7F7F) << 1)
            }
            16 => {
                let limbl1 =
                    ((!limb & 0x8000800080008000) >> 15) | ((!limb & 0x7FFF7FFF7FFF7FFF) << 1); // Left rotation by 1
                let limbl2 =
                    ((limb & 0xC000C000C000C000) >> 14) | ((limb & 0x3FFF3FFF3FFF3FFF) << 2); // Left rotation by 2
                let limbl3 =
                    ((limb & 0xE000E000E000E000) >> 13) | ((limb & 0x1FFF1FFF1FFF1FFF) << 3); // Left rotation by 3

                // y_i = x_i + (1 + x_{i+1}) * x_{i+2} * x_{i+3}
                let tmp = limb ^ limbl1 & limbl2 & limbl3;
                ((tmp & 0x8000800080008000) >> 15) | ((tmp & 0x7FFF7FFF7FFF7FFF) << 1)
                // Final rotation
            }
            _ => {
                panic!("Unsupported lookup size");
            }
        }
    }

    /// Same as `bar` optimized for u128
    #[inline(always)]
    fn bar_u128(el: &mut u128) {
        let limb = *el as u64;
        *el = match LOOKUP_BITS {
            8 => {
                let limbl1 =
                    ((!limb & 0x8080808080808080) >> 7) | ((!limb & 0x7F7F7F7F7F7F7F7F) << 1); // Left rotation by 1
                let limbl2 =
                    ((limb & 0xC0C0C0C0C0C0C0C0) >> 6) | ((limb & 0x3F3F3F3F3F3F3F3F) << 2); // Left rotation by 2
                let limbl3 =
                    ((limb & 0xE0E0E0E0E0E0E0E0) >> 5) | ((limb & 0x1F1F1F1F1F1F1F1F) << 3); // Left rotation by 3

                // y_i = x_i + (1 + x_{i+1}) * x_{i+2} * x_{i+3}
                let tmp = limb ^ limbl1 & limbl2 & limbl3;
                ((tmp & 0x8080808080808080) >> 7) | ((tmp & 0x7F7F7F7F7F7F7F7F) << 1)
            }
            16 => {
                let limbl1 =
                    ((!limb & 0x8000800080008000) >> 15) | ((!limb & 0x7FFF7FFF7FFF7FFF) << 1); // Left rotation by 1
                let limbl2 =
                    ((limb & 0xC000C000C000C000) >> 14) | ((limb & 0x3FFF3FFF3FFF3FFF) << 2); // Left rotation by 2
                let limbl3 =
                    ((limb & 0xE000E000E000E000) >> 13) | ((limb & 0x1FFF1FFF1FFF1FFF) << 3); // Left rotation by 3

                // y_i = x_i + (1 + x_{i+1}) * x_{i+2} * x_{i+3}
                let tmp = limb ^ limbl1 & limbl2 & limbl3;
                ((tmp & 0x8000800080008000) >> 15) | ((tmp & 0x7FFF7FFF7FFF7FFF) << 1)
                // Final rotation
            }
            _ => {
                panic!("Unsupported lookup size");
            }
        } as u128;
    }

    /// Same as `bars` optimized for u128
    fn bars_u128(state_u128: &mut [u128; SPONGE_WIDTH]) {
        Self::bar_u128(&mut state_u128[0]);
        Self::bar_u128(&mut state_u128[1]);
        Self::bar_u128(&mut state_u128[2]);
        Self::bar_u128(&mut state_u128[3]);
    }

    /// Compute the "Bricks" component
    #[inline(always)]
    #[unroll_for_loops]
    fn bricks(state: &mut [Self; SPONGE_WIDTH]) {
        // Feistel Type-3
        for i in (1..SPONGE_WIDTH).rev() {
            let prev = state[i - 1];
            let tmp_square = prev * prev;
            state[i] += tmp_square;
        }
    }

    /// Same as `bricks` optimized for u128
    /// Result is not reduced!
    #[unroll_for_loops]
    fn bricks_u128(state_u128: &mut [u128; SPONGE_WIDTH]) {
        // Feistel Type-3
        // Use "& 0xFFFFFFFFFFFFFFFF" to tell the compiler it is dealing with 64-bit values (save
        // some instructions for upper half)
        for i in (1..SPONGE_WIDTH).rev() {
            let prev = state_u128[i - 1];
            let mut tmp_square =
                (prev & 0xFFFFFFFFFFFFFFFF_u128) * (prev & 0xFFFFFFFFFFFFFFFF_u128);
            tmp_square = Self::from_noncanonical_u128(tmp_square).to_noncanonical_u64() as u128;
            state_u128[i] =
                (state_u128[i] & 0xFFFFFFFFFFFFFFFF_u128) + (tmp_square & 0xFFFFFFFFFFFFFFFF_u128);
        }
    }

    /// Same as `bricks` for field extensions of `Self`.
    #[inline(always)]
    #[unroll_for_loops]
    fn bricks_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; SPONGE_WIDTH],
    ) {
        // Feistel Type-3
        // Feistel Type-3
        for i in (1..SPONGE_WIDTH).rev() {
            let prev = state[i - 1];
            let tmp_square = prev * prev;
            state[i] += tmp_square;
        }
    }

    /// Recursive version of `bricks`.
    #[inline(always)]
    #[unroll_for_loops]
    fn bricks_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &mut [ExtensionTarget<D>; SPONGE_WIDTH],
    ) where
        Self: RichField + Extendable<D>,
    {
        // Feistel Type-3
        for i in (1..SPONGE_WIDTH).rev() {
            let prev = state[i - 1];
            state[i] = builder.mul_add_extension(prev, prev, state[i])
        }
    }

    /// Compute the "Concrete" component
    #[inline(always)]
    #[unroll_for_loops]
    fn concrete(state: &mut [Self; SPONGE_WIDTH], round_constants: &[u64; SPONGE_WIDTH]) {
        let mut state_tmp = [0u128; SPONGE_WIDTH];
        let mut state_u128 = [0u128; SPONGE_WIDTH];
        for (dst, src) in state_u128.iter_mut().zip(state.iter()) {
            *dst = src.to_noncanonical_u64() as u128;
        }
        concrete_u128_with_tmp_buffer::<Self>(&state_u128, round_constants, &mut state_tmp);
        for (dst, src) in state.iter_mut().zip(state_tmp.iter()) {
            *dst = Self::from_noncanonical_u64(*src as u64)
        }
    }

    /// Same as `concrete` optimized for u128
    fn concrete_u128(state_u128: &mut [u128; SPONGE_WIDTH], round_constants: &[u64; SPONGE_WIDTH]) {
        let mut state_tmp = [0_u128; SPONGE_WIDTH];
        concrete_u128_with_tmp_buffer::<Self>(state_u128, round_constants, &mut state_tmp);
        state_u128.copy_from_slice(&state_tmp);
    }

    /// Same as `concrete` for field extensions of `Self`.
    #[inline(always)]
    #[unroll_for_loops]
    fn concrete_field<F: FieldExtension<D, BaseField = Self>, const D: usize>(
        state: &mut [F; SPONGE_WIDTH],
        round_constants: &[u64; SPONGE_WIDTH],
    ) {
        let mut state_tmp = vec![F::ZERO; SPONGE_WIDTH];
        for row in 0..SPONGE_WIDTH {
            for (column, input) in state.iter().enumerate() {
                state_tmp[row] += *input * F::from_canonical_u64(Self::MAT_12[row][column]);
            }
            state_tmp[row] += F::from_canonical_u64(round_constants[row]);
        }
        state.copy_from_slice(&state_tmp);
    }

    /// Recursive version of `concrete`.
    #[inline(always)]
    #[unroll_for_loops]
    fn concrete_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &mut [ExtensionTarget<D>; SPONGE_WIDTH],
        round_constants: &[u64; SPONGE_WIDTH],
    ) where
        Self: RichField + Extendable<D>,
    {
        let mut state_tmp = vec![builder.zero_extension(); SPONGE_WIDTH];
        for row in 0..SPONGE_WIDTH {
            for (column, input) in state.iter().enumerate() {
                state_tmp[row] = builder.mul_const_add_extension(
                    Self::from_canonical_u64(Self::MAT_12[row][column]),
                    *input,
                    state_tmp[row],
                );
            }
            state_tmp[row] = builder.add_const_extension(
                state_tmp[row],
                Self::from_canonical_u64(round_constants[row]),
            );
        }
        state.copy_from_slice(&state_tmp);
    }

    /// Full Monolith permutation
    #[cfg(not(target_feature = "avx2"))]
    #[inline]
    fn monolith(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        let mut state_u128 = [0; SPONGE_WIDTH];
        for (out, inp) in state_u128.iter_mut().zip(input.iter()) {
            *out = inp.to_noncanonical_u64() as u128;
        }

        Self::concrete_u128(&mut state_u128, &Self::ROUND_CONSTANTS[0]);
        for rc in Self::ROUND_CONSTANTS.iter().skip(1) {
            Self::bars_u128(&mut state_u128);
            Self::bricks_u128(&mut state_u128);
            Self::concrete_u128(&mut state_u128, rc);
        }

        // Convert back
        let mut state_f = [Self::ZERO; SPONGE_WIDTH];
        for (out, inp) in state_f.iter_mut().zip(state_u128.iter()) {
            *out = Self::from_canonical_u64(*inp as u64);
        }
        state_f
    }

    #[cfg(target_feature = "avx2")]
    #[inline]
    fn monolith(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        use super::arch::x86_64::monolith_avx2::monolith_avx;

        let mut state_u64 = [0u64; SPONGE_WIDTH];
        for (out, inp) in state_u64.iter_mut().zip(input.iter()) {
            *out = inp.to_noncanonical_u64();
        }

        monolith_avx(&mut state_u64);

        let mut state_f = [Self::ZERO; SPONGE_WIDTH];
        for (out, inp) in state_f.iter_mut().zip(state_u64.iter()) {
            *out = Self::from_canonical_u64(*inp as u64);
        }
        state_f
    }
}

/// Implementor of Plonky2 `PlonkyPermutation` trait for Monolith
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct MonolithPermutation<T> {
    state: [T; SPONGE_WIDTH],
}

impl<T: Eq> Eq for MonolithPermutation<T> {}

impl<T> AsRef<[T]> for MonolithPermutation<T> {
    fn as_ref(&self) -> &[T] {
        &self.state
    }
}

trait Permuter: Sized {
    fn permute(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH];
}

impl<F: Monolith> Permuter for F {
    fn permute(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        <F as Monolith>::monolith(input)
    }
}

impl Permuter for Target {
    fn permute(_input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        panic!("Call `permute_swapped()` instead of `permute()`");
    }
}

impl<T: Copy + Debug + Default + Eq + Permuter + Send + Sync> PlonkyPermutation<T>
    for MonolithPermutation<T>
{
    const RATE: usize = SPONGE_RATE;
    const WIDTH: usize = SPONGE_WIDTH;

    fn new<I: IntoIterator<Item = T>>(elts: I) -> Self {
        let mut perm = Self {
            state: [T::default(); SPONGE_WIDTH],
        };
        perm.set_from_iter(elts, 0);
        perm
    }

    fn set_elt(&mut self, elt: T, idx: usize) {
        self.state[idx] = elt;
    }

    fn set_from_slice(&mut self, elts: &[T], start_idx: usize) {
        let begin = start_idx;
        let end = start_idx + elts.len();
        self.state[begin..end].copy_from_slice(elts);
    }

    fn set_from_iter<I: IntoIterator<Item = T>>(&mut self, elts: I, start_idx: usize) {
        for (s, e) in self.state[start_idx..].iter_mut().zip(elts) {
            *s = e;
        }
    }

    fn permute(&mut self) {
        self.state = T::permute(self.state);
    }

    fn squeeze(&self) -> &[T] {
        &self.state[..Self::RATE]
    }
}

/// Implementor of Plonky2 `Hasher` trait for Monolith
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MonolithHash;
impl<F: RichField + Monolith> Hasher<F> for MonolithHash {
    const HASH_SIZE: usize = 4 * 8;
    type Hash = HashOut<F>;
    type Permutation = MonolithPermutation<F>;

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        hash_n_to_hash_no_pad::<F, Self::Permutation>(input)
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        compress::<F, Self::Permutation>(left, right)
    }

    const HASHER_TYPE: HasherType = HasherType::Monolith;

    fn hash_public_inputs(input: &[F]) -> Self::Hash {
        Self::hash_no_pad(input)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use core::ops::Mul;

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::{Field, Sample};
    use crate::gates::monolith::generate_config_for_monolith_gate;
    use crate::hash::hash_types::NUM_HASH_OUT_ELTS;
    use crate::hash::monolith_goldilocks::MonolithGoldilocksConfig;
    use crate::iop::witness::{PartialWitness, WitnessWrite};

    pub(crate) fn check_test_vectors<F: Field>(
        test_vectors: Vec<([u64; SPONGE_WIDTH], [u64; SPONGE_WIDTH])>,
    ) where
        F: Monolith,
    {
        for (input_, expected_output_) in test_vectors.into_iter() {
            let mut input = [F::ZERO; SPONGE_WIDTH];
            for i in 0..SPONGE_WIDTH {
                input[i] = F::from_canonical_u64(input_[i]);
            }
            let output = F::monolith(input);
            for i in 0..SPONGE_WIDTH {
                let ex_output = F::from_canonical_u64(expected_output_[i]);
                assert_eq!(output[i], ex_output);
            }
        }
    }

    #[test]
    fn test_circuit_with_monolith() {
        type F = GoldilocksField;
        const D: usize = 2;
        type H = MonolithHash;
        type C = MonolithGoldilocksConfig;
        const NUM_OPS: usize = 1024;
        let config = generate_config_for_monolith_gate::<F, D>();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let init_t = builder.add_virtual_public_input();
        let mut res_t = builder.add_virtual_target();
        builder.connect(init_t, res_t);
        let hash_targets = (0..NUM_HASH_OUT_ELTS - 1)
            .map(|_| builder.add_virtual_target())
            .collect::<Vec<_>>();
        for _ in 0..NUM_OPS {
            res_t = builder.mul(res_t, res_t);
            let mut to_be_hashed_elements = vec![res_t];
            to_be_hashed_elements.extend_from_slice(hash_targets.as_slice());
            res_t = builder.hash_or_noop::<H>(to_be_hashed_elements).elements[0]
        }
        let out_t = builder.add_virtual_public_input();
        let is_eq_t = builder.is_equal(out_t, res_t);
        builder.assert_one(is_eq_t.target);

        let data = builder.build::<C>();

        let mut pw = PartialWitness::<F>::new();
        let input = F::rand();
        pw.set_target(init_t, input);

        let input_hash_elements = hash_targets
            .iter()
            .map(|&hash_t| {
                let elem = F::rand();
                pw.set_target(hash_t, elem);
                elem
            })
            .collect::<Vec<_>>();

        let mut res = input;
        for _ in 0..NUM_OPS {
            res = res.mul(res);
            let mut to_be_hashed_elements = vec![res];
            to_be_hashed_elements.extend_from_slice(input_hash_elements.as_slice());
            res = H::hash_or_noop(to_be_hashed_elements.as_slice()).elements[0]
        }

        pw.set_target(out_t, res);

        let proof = data.prove(pw).unwrap();

        assert_eq!(proof.public_inputs[0], input);
        assert_eq!(proof.public_inputs[1], res);

        data.verify(proof).unwrap();
    }
}
