use core::ops::Range;

use crate::field::extension::Extendable;
use crate::gates::gate::Gate;
use crate::hash::hash_types::RichField;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::util::serialization::{Buffer, IoResult};
/// Trait for gates which interpolate a polynomial, whose points are a (base field) coset of the multiplicative subgroup
/// with the given size, and whose values are extension field elements, given by input wires.
/// Outputs the evaluation of the interpolant at a given (extension field) evaluation point.
pub(crate) trait InterpolationGate<F: RichField + Extendable<D>, const D: usize>:
    Gate<F, D> + Copy
{
    fn new(subgroup_bits: usize) -> Self;

    fn id(&self) -> String {
        // Custom implementation to not have the entire lookup table
        format!("InterpolationGate",)
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<()> {
        todo!()
    }

    fn deserialize(_src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        todo!()
    }

    fn num_points(&self) -> usize;

    /// Wire index of the coset shift.
    fn wire_shift(&self) -> usize {
        0
    }

    fn start_values(&self) -> usize {
        1
    }

    /// Wire indices of the `i`th interpolant value.
    fn wires_value(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.num_points());
        let start = self.start_values() + i * D;
        start..start + D
    }

    fn start_evaluation_point(&self) -> usize {
        self.start_values() + self.num_points() * D
    }

    /// Wire indices of the point to evaluate the interpolant at.
    fn wires_evaluation_point(&self) -> Range<usize> {
        let start = self.start_evaluation_point();
        start..start + D
    }

    fn start_evaluation_value(&self) -> usize {
        self.start_evaluation_point() + D
    }

    /// Wire indices of the interpolated value.
    fn wires_evaluation_value(&self) -> Range<usize> {
        let start = self.start_evaluation_value();
        start..start + D
    }

    fn start_coeffs(&self) -> usize {
        self.start_evaluation_value() + D
    }

    /// The number of routed wires required in the typical usage of this gate, where the points to
    /// interpolate, the evaluation point, and the corresponding value are all routed.
    fn num_routed_wires(&self) -> usize {
        self.start_coeffs()
    }

    /// Wire indices of the interpolant's `i`th coefficient.
    fn wires_coeff(&self, i: usize) -> Range<usize> {
        debug_assert!(i < self.num_points());
        let start = self.start_coeffs() + i * D;
        start..start + D
    }

    fn end_coeffs(&self) -> usize {
        self.start_coeffs() + D * self.num_points()
    }
}
