#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::marker::PhantomData;
use core::ops::Range;

use crate::field::extension::algebra::ExtensionAlgebra;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::hash::poseidon::{Poseidon, SPONGE_WIDTH};
use crate::iop::ext_target::{ExtensionAlgebraTarget, ExtensionTarget};
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// Poseidon MDS Gate
#[derive(Debug, Default)]
pub struct PoseidonMdsGate<F: RichField + Extendable<D> + Poseidon, const D: usize>(PhantomData<F>);

impl<F: RichField + Extendable<D> + Poseidon, const D: usize> PoseidonMdsGate<F, D> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }

    pub(crate) const fn wires_input(i: usize) -> Range<usize> {
        assert!(i < SPONGE_WIDTH);
        i * D..(i + 1) * D
    }

    pub(crate) const fn wires_output(i: usize) -> Range<usize> {
        assert!(i < SPONGE_WIDTH);
        (SPONGE_WIDTH + i) * D..(SPONGE_WIDTH + i + 1) * D
    }

    // Following are methods analogous to ones in `Poseidon`, but for extension algebras.

    /// Same as `mds_row_shf` for an extension algebra of `F`.
    fn mds_row_shf_algebra(
        r: usize,
        v: &[ExtensionAlgebra<F::Extension, D>; SPONGE_WIDTH],
    ) -> ExtensionAlgebra<F::Extension, D> {
        debug_assert!(r < SPONGE_WIDTH);
        let mut res = ExtensionAlgebra::ZERO;

        for i in 0..SPONGE_WIDTH {
            let coeff = F::Extension::from_canonical_u64(<F as Poseidon>::MDS_MATRIX_CIRC[i]);
            res += v[(i + r) % SPONGE_WIDTH].scalar_mul(coeff);
        }
        {
            let coeff = F::Extension::from_canonical_u64(<F as Poseidon>::MDS_MATRIX_DIAG[r]);
            res += v[r].scalar_mul(coeff);
        }

        res
    }

    /// Same as `mds_row_shf_recursive` for an extension algebra of `F`.
    fn mds_row_shf_algebra_circuit(
        builder: &mut CircuitBuilder<F, D>,
        r: usize,
        v: &[ExtensionAlgebraTarget<D>; SPONGE_WIDTH],
    ) -> ExtensionAlgebraTarget<D> {
        debug_assert!(r < SPONGE_WIDTH);
        let mut res = builder.zero_ext_algebra();

        for i in 0..SPONGE_WIDTH {
            let coeff = builder.constant_extension(F::Extension::from_canonical_u64(
                <F as Poseidon>::MDS_MATRIX_CIRC[i],
            ));
            res = builder.scalar_mul_add_ext_algebra(coeff, v[(i + r) % SPONGE_WIDTH], res);
        }
        {
            let coeff = builder.constant_extension(F::Extension::from_canonical_u64(
                <F as Poseidon>::MDS_MATRIX_DIAG[r],
            ));
            res = builder.scalar_mul_add_ext_algebra(coeff, v[r], res);
        }

        res
    }

    /// Same as `mds_layer` for an extension algebra of `F`.
    fn mds_layer_algebra(
        state: &[ExtensionAlgebra<F::Extension, D>; SPONGE_WIDTH],
    ) -> [ExtensionAlgebra<F::Extension, D>; SPONGE_WIDTH] {
        let mut result = [ExtensionAlgebra::ZERO; SPONGE_WIDTH];

        for r in 0..SPONGE_WIDTH {
            result[r] = Self::mds_row_shf_algebra(r, state);
        }

        result
    }

    /// Same as `mds_layer_recursive` for an extension algebra of `F`.
    fn mds_layer_algebra_circuit(
        builder: &mut CircuitBuilder<F, D>,
        state: &[ExtensionAlgebraTarget<D>; SPONGE_WIDTH],
    ) -> [ExtensionAlgebraTarget<D>; SPONGE_WIDTH] {
        let mut result = [builder.zero_ext_algebra(); SPONGE_WIDTH];

        for r in 0..SPONGE_WIDTH {
            result[r] = Self::mds_row_shf_algebra_circuit(builder, r, state);
        }

        result
    }
}

impl<F: RichField + Extendable<D> + Poseidon, const D: usize> Gate<F, D> for PoseidonMdsGate<F, D> {
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
        Ok(PoseidonMdsGate::new())
    }

    fn export_circom_verification_code(&self) -> String {
        assert_eq!(D, 2);
        assert_eq!(SPONGE_WIDTH, 12);
        let template_str = format!(
            "template PoseidonMdsGate12() {{
  signal input constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input wires[NUM_OPENINGS_WIRES()][2];
  signal input public_input_hash[4];
  signal input constraints[NUM_GATE_CONSTRAINTS()][2];
  signal output out[NUM_GATE_CONSTRAINTS()][2];

  signal filter[2];
  $SET_FILTER;

  signal state[13][12][2][2];
  for (var r = 0; r < 12; r++) {{
    for (var i = 0; i < 12; i++) {{
      var j = i + r >= 12 ? i + r - 12 : i + r;
      if (i == 0) {{
        state[i][r][0] <== GlExtScalarMul()(wires[j * 2], MDS_MATRIX_CIRC(i));
        state[i][r][1] <== GlExtScalarMul()(wires[j * 2 + 1], MDS_MATRIX_CIRC(i));
      }} else {{
        state[i][r][0] <== GlExtAdd()(state[i - 1][r][0], GlExtScalarMul()(wires[j * 2], MDS_MATRIX_CIRC(i)));
        state[i][r][1] <== GlExtAdd()(state[i - 1][r][1], GlExtScalarMul()(wires[j * 2 + 1], MDS_MATRIX_CIRC(i)));
      }}
    }}
    state[12][r][0] <== GlExtAdd()(state[11][r][0], GlExtScalarMul()(wires[r * 2], MDS_MATRIX_DIAG(r)));
    state[12][r][1] <== GlExtAdd()(state[11][r][1], GlExtScalarMul()(wires[r * 2 + 1], MDS_MATRIX_DIAG(r)));
  }}

  for (var r = 0; r < 12; r ++) {{
    out[r * 2] <== ConstraintPush()(constraints[r * 2], filter, GlExtSub()(wires[(12 + r) * 2], state[12][r][0]));
    out[r * 2 + 1] <== ConstraintPush()(constraints[r * 2 + 1], filter, GlExtSub()(wires[(12 + r) * 2 + 1], state[12][r][1]));
  }}

  for (var i = 24; i < NUM_GATE_CONSTRAINTS(); i++) {{
    out[i] <== constraints[i];
  }}
}}"
        ).to_string();
        template_str
    }
    fn export_solidity_verification_code(&self) -> String {
        todo!()
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let inputs: [_; SPONGE_WIDTH] = (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let computed_outputs = Self::mds_layer_algebra(&inputs);

        (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_output(i)))
            .zip(computed_outputs)
            .flat_map(|(out, computed_out)| (out - computed_out).to_basefield_array())
            .collect()
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        let inputs: [_; SPONGE_WIDTH] = (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let computed_outputs = F::mds_layer_field(&inputs);

        yield_constr.many(
            (0..SPONGE_WIDTH)
                .map(|i| vars.get_local_ext(Self::wires_output(i)))
                .zip(computed_outputs)
                .flat_map(|(out, computed_out)| (out - computed_out).to_basefield_array()),
        )
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let inputs: [_; SPONGE_WIDTH] = (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let computed_outputs = Self::mds_layer_algebra_circuit(builder, &inputs);

        (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_output(i)))
            .zip(computed_outputs)
            .flat_map(|(out, computed_out)| {
                builder
                    .sub_ext_algebra(out, computed_out)
                    .to_ext_target_array()
            })
            .collect()
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        let gen = PoseidonMdsGenerator::<D> { row };
        vec![WitnessGeneratorRef::new(gen.adapter())]
    }

    fn num_wires(&self) -> usize {
        2 * D * SPONGE_WIDTH
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn num_constraints(&self) -> usize {
        SPONGE_WIDTH * D
    }
}

#[derive(Clone, Debug, Default)]
pub struct PoseidonMdsGenerator<const D: usize> {
    row: usize,
}

impl<F: RichField + Extendable<D> + Poseidon, const D: usize> SimpleGenerator<F, D>
    for PoseidonMdsGenerator<D>
{
    fn id(&self) -> String {
        "PoseidonMdsGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        (0..SPONGE_WIDTH)
            .flat_map(|i| {
                Target::wires_from_range(self.row, PoseidonMdsGate::<F, D>::wires_input(i))
            })
            .collect()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let get_local_get_target = |wire_range| ExtensionTarget::from_range(self.row, wire_range);
        let get_local_ext =
            |wire_range| witness.get_extension_target(get_local_get_target(wire_range));

        let inputs: [_; SPONGE_WIDTH] = (0..SPONGE_WIDTH)
            .map(|i| get_local_ext(PoseidonMdsGate::<F, D>::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let outputs = F::mds_layer_field(&inputs);

        for (i, &out) in outputs.iter().enumerate() {
            out_buffer.set_extension_target(
                get_local_get_target(PoseidonMdsGate::<F, D>::wires_output(i)),
                out,
            );
        }
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        Ok(Self { row })
    }
}

#[cfg(test)]
mod tests {
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::gates::poseidon_mds::PoseidonMdsGate;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = PoseidonMdsGate::<F, D>::new();
        test_low_degree(gate)
    }

    #[test]
    fn eval_fns() -> anyhow::Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = PoseidonMdsGate::<F, D>::new();
        test_eval_fns::<F, C, _, D>(gate)
    }
}
