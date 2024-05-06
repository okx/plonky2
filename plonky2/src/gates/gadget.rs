use crate::gates::base_sum_custom::{BaseSplitGenerator, BaseSumCustomGate};
use crate::gates::monolith::MonolithGate;
use crate::hash::monolith::{
    Monolith, MonolithHash, MonolithPermutation, LOOKUP_BITS, LOOKUP_NUM_LIMBS, LOOKUP_SIZE,
    NUM_BARS, N_ROUNDS, SPONGE_WIDTH,
};
use crate::field::extension::Extendable;
use crate::gates::lookup_table::LookupTable;
use crate::hash::hash_types::{HashOutTarget, RichField};
use crate::hash::hashing::PlonkyPermutation;
use crate::iop::generator::{GeneratedValues, SimpleGenerator};
use crate::iop::target::{BoolTarget, Target};
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::config::AlgebraicHasher;
use crate::util::serialization::{Buffer, IoResult};
use std::sync::Arc;

/// `SplitAndLookup` provides a method to perform the following operation in a Plonky2 circuit:
/// 1) Split the input element into a list of targets, where each one represents a
/// base B limb of the element, with little-endian ordering
/// 2) Applies a lookup table (which should be defined only over base B input values) to each element
/// 3) Composes the final target using the outputs of the lookup table
pub trait SplitAndLookup<const B: usize> {
    /// Split and lookup functionality: `x` is the input element, `num_limbs` the number of base `B`
    /// limbs `x` has to be split into, and `lut_index` identifies the lookup table to be applied,
    /// which is assumed to have been already added to the set of lookup tables of the circuit
    fn split_le_lookup(&mut self, x: Target, num_limbs: usize, lut_index: usize) -> Target;
}

impl<F: RichField + Extendable<D>, const D: usize, const B: usize> SplitAndLookup<B>
    for CircuitBuilder<F, D>
{
    fn split_le_lookup(&mut self, x: Target, num_limbs: usize, lut_index: usize) -> Target {
        // Split into individual targets (decompose)
        let gate_type = BaseSumCustomGate::<B>::new(num_limbs, &self.config);
        let (gate, i) = self.find_slot(gate_type, &[F::from_canonical_usize(num_limbs)], &[]);
        let sum = Target::wire(gate, gate_type.ith_wire_sum(i));
        self.connect(x, sum);

        let split_targets_in = Target::wires_from_range(gate, gate_type.ith_limbs(i));

        // Apply lookups
        let mut split_targets_out = vec![];
        for i in 0..num_limbs {
            split_targets_out.push(self.add_lookup_from_index(split_targets_in[i], lut_index));
        }

        // Get final output target (compose)
        let limbs = split_targets_out;

        let (row, i) = self.find_slot(gate_type, &[F::from_canonical_usize(num_limbs)], &[]);
        for (limb, wire) in limbs.iter().zip(gate_type.ith_limbs(i)) {
            self.connect(*limb, Target::wire(row, wire));
        }

        self.add_simple_generator(BaseSumCustomRestrictGenerator::<B>(
            BaseSplitGenerator::new(row, num_limbs, i),
        ));

        Target::wire(row, gate_type.ith_wire_sum(i))
    }
}

#[derive(Debug, Default, Clone)]
struct BaseSumCustomRestrictGenerator<const B: usize>(BaseSplitGenerator<B>);

impl<F: RichField + Extendable<D>, const B: usize, const D: usize> SimpleGenerator<F, D>
    for BaseSumCustomRestrictGenerator<B>
{
    fn id(&self) -> String {
        "BaseSumCustomRestrictGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        self.0.limbs_wires()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let sum = self
            .0
            .limbs_wires()
            .iter()
            .map(|&t| witness.get_target(t))
            .rev()
            .fold(F::ZERO, |acc, limb| acc * F::from_canonical_usize(B) + limb);

        out_buffer.set_target(self.0.wire_sum(), sum);
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        self.0.serialize(dst, _common_data)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let gen = BaseSplitGenerator::deserialize(src, _common_data)?;
        Ok(BaseSumCustomRestrictGenerator::<B>(gen))
    }
}

impl<F: RichField + Monolith> AlgebraicHasher<F> for MonolithHash {
    type AlgebraicPermutation = MonolithPermutation<Target>;

    fn permute_swapped<const D: usize>(
        inputs: Self::AlgebraicPermutation,
        swap: BoolTarget,
        builder: &mut CircuitBuilder<F, D>,
    ) -> Self::AlgebraicPermutation
    where
        F: RichField + Extendable<D>,
    {
        let lut_index = add_monolith_lookup_table(builder);
        let gate_type = MonolithGate::<F, D>::new();
        let gate = builder.add_gate(gate_type, vec![]);

        let swap_wire = MonolithGate::<F, D>::WIRE_SWAP;
        let swap_wire = Target::wire(gate, swap_wire);
        builder.connect(swap.target, swap_wire);

        // Route input wires.
        let inputs = inputs.as_ref();
        for i in 0..SPONGE_WIDTH {
            let in_wire = MonolithGate::<F, D>::wire_input(i);
            let in_wire = Target::wire(gate, in_wire);
            builder.connect(inputs[i], in_wire);
        }

        // Route lookup wires
        for round_ctr in 0..N_ROUNDS {
            for i in 0..NUM_BARS {
                let target_input: Target =
                    Target::wire(gate, MonolithGate::<F, D>::wire_concrete_out(round_ctr, i));
                let target_output =
                    Target::wire(gate, MonolithGate::<F, D>::wire_bars_out(round_ctr, i));
                let target_should = SplitAndLookup::<LOOKUP_SIZE>::split_le_lookup(
                    builder,
                    target_input,
                    LOOKUP_NUM_LIMBS,
                    lut_index,
                ); // Assumes a single lookup table
                builder.connect(target_output, target_should);
            }
        }

        // Collect output wires.
        Self::AlgebraicPermutation::new(
            (0..SPONGE_WIDTH).map(|i| Target::wire(gate, MonolithGate::<F, D>::wire_output(i))),
        )
    }

    fn public_inputs_hash<const D: usize>(
        inputs: Vec<Target>,
        builder: &mut CircuitBuilder<F, D>,
    ) -> crate::hash::hash_types::HashOutTarget
    where
        F: RichField + Extendable<D> {
            HashOutTarget::from_vec(builder.hash_n_to_m_no_pad::<MonolithHash>(inputs, 4))
    }
}

pub(crate) fn add_monolith_lookup_table<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
) -> usize {
    // Add lookup table for Monolith. To ensure that the big lookup-table of Monolith is computed
    // and added to the builder only the first time this function is called, we employ a fake small
    // lookup-table to the circuit builder: if such a fake table is not available, then we compute
    // and add the big Monolith table; otherwise, we skip the computation of the Monolith table and
    // we simply return its index
    let fake_table: LookupTable = Arc::new(vec![(0u16, 0u16)]);
    if let Some(idx) = builder.is_stored(fake_table.clone()) {
        idx + 1
    } else {
        let fake_idx = builder.add_lookup_table_from_pairs(fake_table);
        // use fake lut in order to avoid errors when generating constraints
        let zero = builder.zero();
        builder.add_lookup_from_index(zero, fake_idx);
        let inp_table: [u16; LOOKUP_SIZE] = core::array::from_fn(|i| i as u16);
        let idx = builder.add_lookup_table_from_fn(
            |i| {
                let limb = i;
                match LOOKUP_BITS {
                    8 => {
                        let limbl1 = ((!limb & 0x80) >> 7) | ((!limb & 0x7F) << 1); // Left rotation by 1
                        let limbl2 = ((limb & 0xC0) >> 6) | ((limb & 0x3F) << 2); // Left rotation by 2
                        let limbl3 = ((limb & 0xE0) >> 5) | ((limb & 0x1F) << 3); // Left rotation by 3

                        // y_i = x_i + (1 + x_{i+1}) * x_{i+2} * x_{i+3}
                        let tmp = limb ^ limbl1 & limbl2 & limbl3;
                        ((tmp & 0x80) >> 7) | ((tmp & 0x7F) << 1)
                    }
                    16 => {
                        let limbl1 = ((!limb & 0x8000) >> 15) | ((!limb & 0x7FFF) << 1); // Left rotation by 1
                        let limbl2 = ((limb & 0xC000) >> 14) | ((limb & 0x3FFF) << 2); // Left rotation by 2
                        let limbl3 = ((limb & 0xE000) >> 13) | ((limb & 0x1FFF) << 3); // Left rotation by 3

                        // y_i = x_i + (1 + x_{i+1}) * x_{i+2} * x_{i+3}
                        let tmp = limb ^ limbl1 & limbl2 & limbl3;
                        ((tmp & 0x8000) >> 15) | ((tmp & 0x7FFF) << 1) // Final rotation
                    }
                    _ => {
                        panic!("Unsupported lookup size");
                    }
                }
            },
            &inp_table,
        );
        assert_eq!(fake_idx + 1, idx);
        idx
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::hash::monolith::{Monolith, MonolithHash, MonolithPermutation, SPONGE_WIDTH};
    use anyhow::Result;
    use log::{info, Level};
    use crate::field::extension::Extendable;
    use crate::hash::hash_types::RichField;
    use crate::hash::hashing::PlonkyPermutation;
    use crate::iop::target::Target;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::{CircuitConfig, CircuitData};
    use crate::plonk::config::{AlgebraicHasher, GenericConfig, Hasher};
    use crate::plonk::proof::ProofWithPublicInputs;
    use crate::plonk::prover::prove;
    use crate::util::timing::TimingTree;

    pub(crate) fn test_monolith_hash_circuit<
        F: RichField + Extendable<D> + Monolith,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        config: CircuitConfig,
    ) {
        let mut builder = CircuitBuilder::new(config);

        let inp_targets_array = builder.add_virtual_target_arr::<SPONGE_WIDTH>();
        let inp_targets = MonolithPermutation::<Target>::new(inp_targets_array);

        let out_targets =
            MonolithHash::permute_swapped(inp_targets, builder._false(), &mut builder);
        builder.register_public_inputs(out_targets.as_ref());
        builder.print_gate_counts(0);

        println!("Num wires: {}", builder.config.num_wires);
        println!("Num routed wires: {}", builder.config.num_routed_wires);

        let now = std::time::Instant::now();
        let circuit = builder.build::<C>();
        println!("[Build time] {:.4} s", now.elapsed().as_secs_f64());
        println!("Circuit degree bits: {}", circuit.common.degree_bits());

        let permutation_inputs = (0..SPONGE_WIDTH)
            .map(F::from_canonical_usize)
            .collect::<Vec<_>>();

        let mut inputs = PartialWitness::new();
        inp_targets
            .as_ref()
            .iter()
            .zip(permutation_inputs.iter())
            .for_each(|(t, val)| inputs.set_target(*t, *val));

        let now = std::time::Instant::now();
        let proof = circuit.prove(inputs).unwrap();
        println!("[Prove time] {:.4} s", now.elapsed().as_secs_f64());
        println!("Proof size (bytes): {}", proof.to_bytes().len());

        let expected_outputs: [F; SPONGE_WIDTH] =
            F::monolith(permutation_inputs.try_into().unwrap());

        proof
            .public_inputs
            .iter()
            .zip(expected_outputs.iter())
            .for_each(|(v, out)| assert_eq!(*v, *out));

        let now = std::time::Instant::now();
        circuit.verify(proof).unwrap();
        println!("[Verify time] {:.4} s", now.elapsed().as_secs_f64());
    }

    pub(crate) fn prove_circuit_with_hash<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
        H: Hasher<F> + AlgebraicHasher<F>,
    >(
        config: CircuitConfig,
        num_ops: usize,
        print_timing: bool,
    ) -> Result<(CircuitData<F, C, D>, ProofWithPublicInputs<F, C, D>)> {
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let init_t = builder.add_virtual_public_input();
        let mut res_t = builder.add_virtual_target();
        builder.connect(init_t, res_t);
        let hash_targets = (0..SPONGE_WIDTH - 1)
            .map(|_| builder.add_virtual_target())
            .collect::<Vec<_>>();
        for _ in 0..num_ops {
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
        for _ in 0..num_ops {
            res = res.mul(res);
            let mut to_be_hashed_elements = vec![res];
            to_be_hashed_elements.extend_from_slice(input_hash_elements.as_slice());
            res = H::hash_no_pad(to_be_hashed_elements.as_slice()).elements[0]
        }

        pw.set_target(out_t, res);

        let proof = if print_timing {
            let mut timing = TimingTree::new("prove", Level::Debug);
            let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
            timing.print();
            info!("proof size: {}", proof.to_bytes().len());
            proof
        } else {
            data.prove(pw)?
        };

        assert_eq!(proof.public_inputs[0], input);
        assert_eq!(proof.public_inputs[1], res);

        Ok((data, proof))
    }

    pub(crate) fn recursive_proof<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        InnerC: GenericConfig<D, F = F>,
        const D: usize,
    >(
        inner_proof: ProofWithPublicInputs<F, InnerC, D>,
        inner_cd: &CircuitData<F, InnerC, D>,
        config: &CircuitConfig,
    ) -> Result<(CircuitData<F, C, D>, ProofWithPublicInputs<F, C, D>)>
    where
        C::Hasher: AlgebraicHasher<F>,
        InnerC::Hasher: AlgebraicHasher<F>,
    {
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let mut pw = PartialWitness::new();
        let pt = builder.add_virtual_proof_with_pis::<InnerC>(&inner_cd.common);
        pw.set_proof_with_pis_target(&pt, &inner_proof);

        let inner_data =
            builder.add_virtual_verifier_data(inner_cd.common.config.fri_config.cap_height);
        pw.set_cap_target(
            &inner_data.constants_sigmas_cap,
            &inner_cd.verifier_only.constants_sigmas_cap,
        );
        pw.set_hash_target(
            inner_data.circuit_digest,
            inner_cd.verifier_only.circuit_digest,
        );

        for &pi_t in pt.public_inputs.iter() {
            let t = builder.add_virtual_public_input();
            builder.connect(pi_t, t);
        }
        builder.verify_proof::<InnerC>(&pt, &inner_data, &inner_cd.common);
        let data = builder.build::<C>();

        let proof = data.prove(pw)?;

        Ok((data, proof))
    }
}
