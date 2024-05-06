use crate::hash::monolith::{Monolith, NUM_BARS, N_ROUNDS, SPONGE_WIDTH};
use itertools::Itertools;
use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, WitnessGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use crate::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};
use crate::util::serialization::{Buffer, IoResult, Read, Write};
use std::cmp;
use std::marker::PhantomData;

/// Evaluates a full Monolith permutation with 12 state elements.
///
/// This also has some extra features to make it suitable for efficiently verifying Merkle proofs.
/// It has a flag which can be used to swap the first four inputs with the next four, for ordering
/// sibling digests.
#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub struct MonolithGate<F: RichField + Extendable<D>, const D: usize>(PhantomData<F>);

impl<F: RichField + Extendable<D>, const D: usize> MonolithGate<F, D> {
    /// Instantiate a new `MonolithGate`
    pub fn new() -> Self {
        Self(PhantomData)
    }

    /// The wire index for the `i`th input to the permutation.
    pub fn wire_input(i: usize) -> usize {
        i
    }

    /// The wire index for the `i`th output to the permutation.
    pub fn wire_output(i: usize) -> usize {
        SPONGE_WIDTH + i
    }

    /// If this is set to 1, the first four inputs will be swapped with the next four inputs. This
    /// is useful for ordering hashes in Merkle proofs. Otherwise, this should be set to 0.
    pub const WIRE_SWAP: usize = 2 * SPONGE_WIDTH;

    const START_DELTA: usize = 2 * SPONGE_WIDTH + 1;

    /// A wire which stores `swap * (input[i + 4] - input[i])`; used to compute the swapped inputs.
    fn wire_delta(i: usize) -> usize {
        assert!(i < 4);
        Self::START_DELTA + i
    }

    const START_PERM: usize = Self::START_DELTA + 4;

    /// A wire which stores the output of the `i`-th Concrete of the `round`-th round
    pub fn wire_concrete_out(round: usize, i: usize) -> usize {
        // Configuration:
        // 1 Concrete_out for each state element
        // 1 Bar_out for each state element which goes through Bars
        // = STATE_SIZE + NUM_BARS cells for each round
        match round {
            0 => {
                debug_assert!(round == 0);
                debug_assert!(i < NUM_BARS);
                Self::START_PERM + i
            }
            _ => {
                debug_assert!(round > 0);
                debug_assert!(i < SPONGE_WIDTH);
                Self::START_PERM + (NUM_BARS * 2) + (SPONGE_WIDTH + NUM_BARS) * (round - 1) + i
            }
        }
    }

    /// A wire which stores the output of the `i`-th Bar of the `round`-th round
    pub fn wire_bars_out(round: usize, i: usize) -> usize {
        debug_assert!(i < NUM_BARS);
        Self::START_PERM + NUM_BARS + (SPONGE_WIDTH + NUM_BARS) * round + i
    }

    /// End of wire indices, exclusive.
    fn end() -> usize {
        Self::START_PERM + (NUM_BARS * 2) + (SPONGE_WIDTH + NUM_BARS) * (N_ROUNDS - 1)
    }
}

impl<F: RichField + Extendable<D> + Monolith, const D: usize> Gate<F, D> for MonolithGate<F, D> {
    fn id(&self) -> String {
        format!("{self:?}<WIDTH={SPONGE_WIDTH}>")
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<()> {
        Ok(())
    }

    fn deserialize(_src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(MonolithGate::new())
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        constraints.push(swap * (swap - F::Extension::ONE));

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            constraints.push(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        let mut state = [F::Extension::ZERO; SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = Self::wire_input(i);
            let input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_wires[input_lhs] + delta_i;
            state[i + 4] = vars.local_wires[input_rhs] - delta_i;
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        // Permutation
        <F as Monolith>::concrete_field(&mut state, &<F as Monolith>::ROUND_CONSTANTS[0]);
        for (round_ctr, rc) in <F as Monolith>::ROUND_CONSTANTS.iter().skip(1).enumerate() {
            // Check values after Concrete and set new state after applying bars
            let loop_end = match round_ctr {
                0 => NUM_BARS,
                _ => SPONGE_WIDTH,
            };
            for i in 0..loop_end {
                let concrete_out = vars.local_wires[Self::wire_concrete_out(round_ctr, i)];
                constraints.push(state[i] - concrete_out);
                // Get values after Bars (this assumes lookups have already been applied, i.e., these are the outputs of Bars)
                if i < NUM_BARS {
                    state[i] = vars.local_wires[Self::wire_bars_out(round_ctr, i)];
                } else {
                    state[i] = concrete_out;
                }
            }

            // Bricks + Concrete
            <F as Monolith>::bricks_field(&mut state);
            <F as Monolith>::concrete_field(&mut state, rc);
        }

        // Final
        for i in 0..SPONGE_WIDTH {
            constraints.push(state[i] - vars.local_wires[Self::wire_output(i)]);
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        yield_constr.one(swap * swap.sub_one());

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            yield_constr.one(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        let mut state = [F::ZERO; SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = Self::wire_input(i);
            let input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_wires[input_lhs] + delta_i;
            state[i + 4] = vars.local_wires[input_rhs] - delta_i;
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        // Permutation
        <F as Monolith>::concrete(&mut state, &<F as Monolith>::ROUND_CONSTANTS[0]);
        for (round_ctr, rc) in <F as Monolith>::ROUND_CONSTANTS.iter().skip(1).enumerate() {
            // Check values after Concrete and set new state after applying bars
            let loop_end = match round_ctr {
                0 => NUM_BARS,
                _ => SPONGE_WIDTH,
            };
            for i in 0..loop_end {
                let concrete_out = vars.local_wires[Self::wire_concrete_out(round_ctr, i)];
                yield_constr.one(state[i] - concrete_out);
                // Get values after Bars (this assumes lookups have already been applied, i.e., these are the outputs of Bars)
                if i < NUM_BARS {
                    state[i] = vars.local_wires[Self::wire_bars_out(round_ctr, i)];
                } else {
                    state[i] = concrete_out;
                }
            }

            // Bricks + Concrete
            <F as Monolith>::bricks(&mut state);
            <F as Monolith>::concrete(&mut state, rc);
        }

        // Final
        for i in 0..SPONGE_WIDTH {
            yield_constr.one(state[i] - vars.local_wires[Self::wire_output(i)]);
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        constraints.push(builder.mul_sub_extension(swap, swap, swap));

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let diff = builder.sub_extension(input_rhs, input_lhs);
            constraints.push(builder.mul_sub_extension(swap, diff, delta_i));
        }

        // Compute the possibly-swapped input layer.
        let mut state = [builder.zero_extension(); SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            state[i] = builder.add_extension(input_lhs, delta_i);
            state[i + 4] = builder.sub_extension(input_rhs, delta_i);
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        // Permutation
        <F as Monolith>::concrete_circuit(
            builder,
            &mut state,
            &<F as Monolith>::ROUND_CONSTANTS[0],
        );
        for (round_ctr, rc) in <F as Monolith>::ROUND_CONSTANTS.iter().skip(1).enumerate() {
            // Check values after Concrete and set new state after applying bars
            let loop_end = match round_ctr {
                0 => NUM_BARS,
                _ => SPONGE_WIDTH,
            };
            for i in 0..loop_end {
                let concrete_out = vars.local_wires[Self::wire_concrete_out(round_ctr, i)];
                constraints.push(builder.sub_extension(state[i], concrete_out));
                // Get values after Bars (this assumes lookups have already been applied, i.e., these are the outputs of Bars)
                if i < NUM_BARS {
                    state[i] = vars.local_wires[Self::wire_bars_out(round_ctr, i)];
                } else {
                    state[i] = concrete_out;
                }
            }

            // Get values after Bars (this assumes lookups have already been applied, i.e., these are the outputs of Bars)
            for i in 0..NUM_BARS {
                state[i] = vars.local_wires[Self::wire_bars_out(round_ctr, i)];
            }

            // Bricks + Concrete
            <F as Monolith>::bricks_circuit(builder, &mut state);
            <F as Monolith>::concrete_circuit(builder, &mut state, rc);
        }

        // Final
        for i in 0..SPONGE_WIDTH {
            constraints
                .push(builder.sub_extension(state[i], vars.local_wires[Self::wire_output(i)]));
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        let gen = MonolithGenerator::<F, D> {
            row,
            _phantom: PhantomData,
        };
        vec![WitnessGeneratorRef::new(gen)]
    }

    fn num_wires(&self) -> usize {
        Self::end()
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        NUM_BARS + SPONGE_WIDTH * (N_ROUNDS - 1) + SPONGE_WIDTH + 1 + 4
    }

    fn export_circom_verification_code(&self) -> String {
        todo!()
    }

    fn export_solidity_verification_code(&self) -> String {
        todo!()
    }
}

/// Generator for `MonolithGate` wires
#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub struct MonolithGenerator<F: RichField + Extendable<D> + Monolith, const D: usize> {
    row: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D> + Monolith, const D: usize> WitnessGenerator<F, D>
    for MonolithGenerator<F, D>
{
    fn id(&self) -> String {
        "MonolithGenerator".to_string()
    }

    fn watch_list(&self) -> Vec<Target> {
        (0..SPONGE_WIDTH)
            .map(|i| MonolithGate::<F, D>::wire_input(i))
            .chain(Some(MonolithGate::<F, D>::WIRE_SWAP))
            .chain(
                (0..N_ROUNDS)
                    .cartesian_product(0..NUM_BARS)
                    .map(|(round, i)| MonolithGate::<F, D>::wire_bars_out(round, i)),
            )
            .map(|column| Target::wire(self.row, column))
            .collect()
    }

    fn run(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) -> bool {
        let local_wire = |column| Wire {
            row: self.row,
            column,
        };

        let mut state = (0..SPONGE_WIDTH)
            .map_while(|i| witness.try_get_wire(local_wire(MonolithGate::<F, D>::wire_input(i))))
            .collect::<Vec<_>>();
        // exit if some of the input wires have not been already computed
        if state.len() < SPONGE_WIDTH {
            return false;
        }

        let swap_value =
            if let Some(wire) = witness.try_get_wire(local_wire(MonolithGate::<F, D>::WIRE_SWAP)) {
                wire
            } else {
                return false;
            };
        debug_assert!(swap_value == F::ZERO || swap_value == F::ONE);

        for i in 0..4 {
            let delta_i = swap_value * (state[i + 4] - state[i]);
            out_buffer.set_wire(local_wire(MonolithGate::<F, D>::wire_delta(i)), delta_i);
        }

        if swap_value == F::ONE {
            for i in 0..4 {
                state.swap(i, 4 + i);
            }
        }

        let mut state: [F; SPONGE_WIDTH] = state.try_into().unwrap();

        // Permutation
        <F as Monolith>::concrete_field(&mut state, &<F as Monolith>::ROUND_CONSTANTS[0]);
        for (round_ctr, rc) in <F as Monolith>::ROUND_CONSTANTS.iter().skip(1).enumerate() {
            // Set values after Concrete
            let loop_end = match round_ctr {
                0 => NUM_BARS,
                _ => SPONGE_WIDTH,
            };
            for i in 0..loop_end {
                out_buffer.set_wire(
                    local_wire(MonolithGate::<F, D>::wire_concrete_out(round_ctr, i)),
                    state[i],
                );
            }

            // Get values after Bars (this assumes lookups have already been applied, i.e., these are the outputs of Bars)
            for i in 0..NUM_BARS {
                state[i] = match witness.try_get_wire(local_wire(
                    MonolithGate::<F, D>::wire_bars_out(round_ctr, i),
                )) {
                    Some(value) => value,
                    None => return false,
                };
            }

            // Bricks + Concrete
            <F as Monolith>::bricks_field(&mut state);
            <F as Monolith>::concrete_field(&mut state, rc);
        }

        // Final
        for i in 0..SPONGE_WIDTH {
            out_buffer.set_wire(local_wire(MonolithGate::<F, D>::wire_output(i)), state[i]);
        }

        true
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        Ok(Self {
            row,
            _phantom: PhantomData,
        })
    }
}

pub fn generate_config_for_monolith_gate<
    F: RichField + Extendable<D> + Monolith,
    const D: usize,
>() -> CircuitConfig {
    let needed_wires = cmp::max(
        MonolithGate::<F, D>::new().num_wires(),
        CircuitConfig::standard_recursion_config().num_wires,
    );
    CircuitConfig {
        num_wires: needed_wires,
        num_routed_wires: needed_wires,
        ..CircuitConfig::standard_recursion_config()
    }
}

#[cfg(test)]
mod tests {
    use crate::gates::monolith::MonolithGate;
    use crate::hash::monolith_goldilocks::MonolithGoldilocksConfig;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::plonk::config::GenericConfig;

    #[test]
    fn wire_indices() {
        type F = GoldilocksField;
        type Gate = MonolithGate<F, 4>;

        assert_eq!(Gate::wire_input(0), 0);
        assert_eq!(Gate::wire_input(11), 11);
        assert_eq!(Gate::wire_output(0), 12);
        assert_eq!(Gate::wire_output(11), 23);
        assert_eq!(Gate::WIRE_SWAP, 24);
        assert_eq!(Gate::wire_delta(0), 25);
        assert_eq!(Gate::wire_delta(3), 28);
        assert_eq!(Gate::wire_concrete_out(0, 0), 29);
        assert_eq!(Gate::wire_bars_out(0, 0), 33);
        assert_eq!(Gate::wire_concrete_out(1, 0), 37);
        assert_eq!(Gate::wire_bars_out(1, 0), 49);
        assert_eq!(Gate::wire_concrete_out(2, 0), 53);
        assert_eq!(Gate::wire_bars_out(2, 0), 65);
        assert_eq!(Gate::wire_concrete_out(3, 0), 69);
        assert_eq!(Gate::wire_bars_out(3, 0), 81);
        assert_eq!(Gate::wire_concrete_out(4, 0), 85);
        assert_eq!(Gate::wire_bars_out(4, 0), 97);
        assert_eq!(Gate::wire_concrete_out(5, 0), 101);
        assert_eq!(Gate::wire_bars_out(5, 0), 113);
    }

    #[test]
    fn low_degree() {
        type F = GoldilocksField;
        let gate = MonolithGate::<F, 4>::new();
        test_low_degree(gate)
    }

    #[test]
    fn eval_fns() {
        const D: usize = 2;
        type C = MonolithGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = MonolithGate::<F, 2>::new();
        test_eval_fns::<F, C, _, D>(gate).unwrap();
    }
}
