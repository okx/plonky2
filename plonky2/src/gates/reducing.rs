#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::ops::Range;

use crate::field::extension::{Extendable, FieldExtension};
use crate::gates::gate::Gate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// Computes `sum alpha^i c_i` for a vector `c_i` of `num_coeffs` elements of the base field.
#[derive(Debug, Default, Clone)]
pub struct ReducingGate<const D: usize> {
    pub num_coeffs: usize,
}

impl<const D: usize> ReducingGate<D> {
    pub const fn new(num_coeffs: usize) -> Self {
        Self { num_coeffs }
    }

    pub fn max_coeffs_len(num_wires: usize, num_routed_wires: usize) -> usize {
        (num_routed_wires - 3 * D).min((num_wires - 2 * D) / (D + 1))
    }

    pub(crate) const fn wires_output() -> Range<usize> {
        0..D
    }
    pub(crate) const fn wires_alpha() -> Range<usize> {
        D..2 * D
    }
    pub(crate) const fn wires_old_acc() -> Range<usize> {
        2 * D..3 * D
    }
    const START_COEFFS: usize = 3 * D;
    pub(crate) const fn wires_coeffs(&self) -> Range<usize> {
        Self::START_COEFFS..Self::START_COEFFS + self.num_coeffs
    }
    const fn start_accs(&self) -> usize {
        Self::START_COEFFS + self.num_coeffs
    }
    const fn wires_accs(&self, i: usize) -> Range<usize> {
        if i == self.num_coeffs - 1 {
            // The last accumulator is the output.
            return Self::wires_output();
        }
        self.start_accs() + D * i..self.start_accs() + D * (i + 1)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for ReducingGate<D> {
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_coeffs)?;
        Ok(())
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized,
    {
        let num_coeffs = src.read_usize()?;
        Ok(Self::new(num_coeffs))
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let alpha = vars.get_local_ext_algebra(Self::wires_alpha());
        let old_acc = vars.get_local_ext_algebra(Self::wires_old_acc());
        let coeffs = self
            .wires_coeffs()
            .map(|i| vars.local_wires[i])
            .collect::<Vec<_>>();
        let accs = (0..self.num_coeffs)
            .map(|i| vars.get_local_ext_algebra(self.wires_accs(i)))
            .collect::<Vec<_>>();

        let mut constraints = Vec::with_capacity(<Self as Gate<F, D>>::num_constraints(self));
        let mut acc = old_acc;
        for i in 0..self.num_coeffs {
            constraints.push(acc * alpha + coeffs[i].into() - accs[i]);
            acc = accs[i];
        }

        constraints
            .into_iter()
            .flat_map(|alg| alg.to_basefield_array())
            .collect()
    }

    fn export_circom_verification_code(&self) -> String {
        let mut template_str = format!(
            "template Reducing$NUM_COEFFS() {{
  signal input constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input wires[NUM_OPENINGS_WIRES()][2];
  signal input public_input_hash[4];
  signal input constraints[NUM_GATE_CONSTRAINTS()][2];
  signal output out[NUM_GATE_CONSTRAINTS()][2];

  signal filter[2];
  $SET_FILTER;

  var acc_start = 2 * $D;
  signal m[$NUM_COEFFS][2][2];
  for (var i = 0; i < $NUM_COEFFS; i++) {{
    m[i] <== WiresAlgebraMul(acc_start, $D)(wires);
    out[i * $D] <== ConstraintPush()(constraints[i * $D], filter, GlExtAdd()(m[i][0], GlExtSub()(wires[3 * $D + i], wires[r_wires_accs_start(i, $NUM_COEFFS)])));
    for (var j = 1; j < $D; j++) {{
      out[i * $D + j] <== ConstraintPush()(constraints[i * $D + j], filter, GlExtSub()(m[i][j], wires[r_wires_accs_start(i, $NUM_COEFFS) + j]));
    }}
    acc_start = r_wires_accs_start(i, $NUM_COEFFS);
  }}

  for (var i = $NUM_COEFFS * $D; i < NUM_GATE_CONSTRAINTS(); i++) {{
    out[i] <== constraints[i];
  }}
}}
function r_wires_accs_start(i, num_coeffs) {{
  if (i == num_coeffs - 1) return 0;
  else return (3 + i) * $D + num_coeffs;
}}"
        ).to_string();

        template_str = template_str.replace("$NUM_COEFFS", &*self.num_coeffs.to_string());
        template_str = template_str.replace("$D", &*D.to_string());

        template_str
    }
    fn export_solidity_verification_code(&self) -> String {
        let mut template_str = format!(
            "library Reducing$NUM_COEFFSLib {{
    using GoldilocksFieldLib for uint64;
    using GoldilocksExtLib for uint64[2];

    function set_filter(GatesUtilsLib.EvaluationVars memory ev) internal pure {{
        $SET_FILTER;
    }}

    function wires_accs_start(uint32 i) internal pure returns(uint32) {{
        if (i == $NUM_COEFFS - 1) return 0;
        return (3 + i) * $D + $NUM_COEFFS;
    }}

    function eval(GatesUtilsLib.EvaluationVars memory ev, uint64[2][$NUM_GATE_CONSTRAINTS] memory constraints) internal pure {{
        uint32 acc_start = 2 * $D;
        for (uint32 i = 0; i < $NUM_COEFFS; i++) {{
            uint64[2][$D] memory m = GatesUtilsLib.wires_algebra_mul(ev.wires, acc_start, $D);
            GatesUtilsLib.push(constraints, ev.filter, i * $D, m[0].add(ev.wires[3 * $D + i].sub(ev.wires[wires_accs_start(i)])));
            for (uint32 j = 1; j < $D; j++) {{
                GatesUtilsLib.push(constraints, ev.filter, i * $D + j, m[j].sub(ev.wires[wires_accs_start(i) + j]));
            }}
            acc_start = wires_accs_start(i);
        }}
    }}
}}"
        )
            .to_string();

        template_str = template_str.replace("$NUM_COEFFS", &*self.num_coeffs.to_string());

        template_str
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        let alpha = vars.get_local_ext(Self::wires_alpha());
        let old_acc = vars.get_local_ext(Self::wires_old_acc());
        let coeffs = self
            .wires_coeffs()
            .map(|i| vars.local_wires[i])
            .collect::<Vec<_>>();
        let accs = (0..self.num_coeffs)
            .map(|i| vars.get_local_ext(self.wires_accs(i)))
            .collect::<Vec<_>>();

        let mut acc = old_acc;
        for i in 0..self.num_coeffs {
            yield_constr.many((acc * alpha + coeffs[i].into() - accs[i]).to_basefield_array());
            acc = accs[i];
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let alpha = vars.get_local_ext_algebra(Self::wires_alpha());
        let old_acc = vars.get_local_ext_algebra(Self::wires_old_acc());
        let coeffs = self
            .wires_coeffs()
            .map(|i| vars.local_wires[i])
            .collect::<Vec<_>>();
        let accs = (0..self.num_coeffs)
            .map(|i| vars.get_local_ext_algebra(self.wires_accs(i)))
            .collect::<Vec<_>>();

        let mut constraints = Vec::with_capacity(<Self as Gate<F, D>>::num_constraints(self));
        let mut acc = old_acc;
        for i in 0..self.num_coeffs {
            let coeff = builder.convert_to_ext_algebra(coeffs[i]);
            let mut tmp = builder.mul_add_ext_algebra(acc, alpha, coeff);
            tmp = builder.sub_ext_algebra(tmp, accs[i]);
            constraints.push(tmp);
            acc = accs[i];
        }

        constraints
            .into_iter()
            .flat_map(|alg| alg.to_ext_target_array())
            .collect()
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        vec![WitnessGeneratorRef::new(
            ReducingGenerator {
                row,
                gate: self.clone(),
            }
            .adapter(),
        )]
    }

    fn num_wires(&self) -> usize {
        2 * D + self.num_coeffs * (D + 1)
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        D * self.num_coeffs
    }
}

#[derive(Debug, Default, Clone)]
pub struct ReducingGenerator<const D: usize> {
    row: usize,
    gate: ReducingGate<D>,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for ReducingGenerator<D> {
    fn id(&self) -> String {
        "ReducingGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        ReducingGate::<D>::wires_alpha()
            .chain(ReducingGate::<D>::wires_old_acc())
            .chain(self.gate.wires_coeffs())
            .map(|i| Target::wire(self.row, i))
            .collect()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let extract_extension = |range: Range<usize>| -> F::Extension {
            let t = ExtensionTarget::from_range(self.row, range);
            witness.get_extension_target(t)
        };

        let alpha = extract_extension(ReducingGate::<D>::wires_alpha());
        let old_acc = extract_extension(ReducingGate::<D>::wires_old_acc());
        let coeffs = witness.get_targets(
            &self
                .gate
                .wires_coeffs()
                .map(|i| Target::wire(self.row, i))
                .collect::<Vec<_>>(),
        );
        let accs = (0..self.gate.num_coeffs)
            .map(|i| ExtensionTarget::from_range(self.row, self.gate.wires_accs(i)))
            .collect::<Vec<_>>();
        let output = ExtensionTarget::from_range(self.row, ReducingGate::<D>::wires_output());

        let mut acc = old_acc;
        for i in 0..self.gate.num_coeffs {
            let computed_acc = acc * alpha + coeffs[i].into();
            out_buffer.set_extension_target(accs[i], computed_acc);
            acc = computed_acc;
        }
        out_buffer.set_extension_target(output, acc);
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        <ReducingGate<D> as Gate<F, D>>::serialize(&self.gate, dst, _common_data)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let gate = <ReducingGate<D> as Gate<F, D>>::deserialize(src, _common_data)?;
        Ok(Self { row, gate })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::field::goldilocks_field::GoldilocksField;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::gates::reducing::ReducingGate;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        test_low_degree::<GoldilocksField, _, 4>(ReducingGate::new(22));
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        test_eval_fns::<F, C, _, D>(ReducingGate::new(22))
    }
}
