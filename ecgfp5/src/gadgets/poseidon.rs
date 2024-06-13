use crate::curve::scalar_field::Scalar;
use plonky2::{
    hash::poseidon::PoseidonHash,
    iop::target::Target,
    plonk::{circuit_builder::CircuitBuilder, config::Hasher},
};
use plonky2_ecdsa::gadgets::nonnative::NonNativeTarget;
use plonky2_field::{goldilocks_field::GoldilocksField, types::Field};

use super::base_field::{CircuitBuilderGFp5, QuinticExtensionTarget};

pub fn hash_to_scalar(domain: &[u8], msg: &[u8]) -> Scalar {
    let f_domain = u8_to_goldilocks(domain);
    let f_msg = u8_to_goldilocks(msg);
    let hashout = PoseidonHash::hash_no_pad(&[f_domain.as_slice(), f_msg.as_slice()].concat());
    Scalar::from_hashout(hashout)
}

pub fn hash_to_scalar_target(
    builder: &mut CircuitBuilder<GoldilocksField, 2>,
    domain: &[u8],
    msg: Vec<Target>,
) -> NonNativeTarget<Scalar> {
    let mut preimage = vec![];
    let f_domain: Vec<Target> =
        u8_to_goldilocks(domain).iter().map(|x| builder.constant(*x)).collect();
    preimage.extend(f_domain);
    preimage.extend(msg);
    let hashout = builder.hash_n_to_hash_no_pad::<PoseidonHash>(preimage);
    let mut limbs = [builder.zero(); 5];
    limbs[1..].copy_from_slice(&hashout.elements);
    let result = QuinticExtensionTarget::new(limbs);
    builder.encode_quintic_ext_as_scalar(result)
}

/// Convert [u8; 8] to one GoldilocksField
///
/// non-canoncial [u8; 8] will panic
pub fn u8_to_goldilocks(data: &[u8]) -> Vec<GoldilocksField> {
    const CHUNK_SIZE: usize = 8;
    data.chunks(CHUNK_SIZE)
        .map(|chunk| {
            let mut padded = [0u8; CHUNK_SIZE];
            let len = chunk.len().min(CHUNK_SIZE);
            padded[..len].copy_from_slice(&chunk[..len]);
            GoldilocksField::from_canonical_u64(u64::from_le_bytes(padded))
        })
        .collect::<Vec<GoldilocksField>>()
}

#[cfg(test)]
mod tests {
    use plonky2::{
        iop::witness::PartialWitness,
        plonk::{
            circuit_data::CircuitConfig,
            config::{GenericConfig, PoseidonGoldilocksConfig},
        },
    };
    use plonky2_ecdsa::gadgets::nonnative::PartialWitnessNonNative;
    use plonky2_field::types::Field64;

    use super::*;

    #[test]
    #[should_panic]
    fn test_u8_to_goldilocks_noncanonical() {
        let order_plus_one = (GoldilocksField::ORDER + 1).to_le_bytes();
        // noncanonical u64 will panic
        u8_to_goldilocks(&order_plus_one);
    }

    #[test]
    fn test_hash_to_scalar() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let domain = b"domain-plonky2-ecgfp5-poseidon";
        let msg = b"msg-to-hash-to-scalar";
        let scalar = hash_to_scalar(domain, msg);

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let msg_target: Vec<Target> =
            u8_to_goldilocks(msg).iter().map(|x| builder.constant(*x)).collect();
        let scalar_target = hash_to_scalar_target(&mut builder, domain, msg_target);

        let circuit = builder.build::<C>();
        let mut pw = PartialWitness::new();
        pw.set_nonnative_target(scalar_target, scalar);
        let proof = circuit.prove(pw).unwrap();
        circuit.verify(proof).unwrap();
    }
}
