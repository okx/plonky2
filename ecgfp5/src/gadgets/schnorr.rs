#![allow(dead_code)]

use crate::{
    curve::{curve::Point, scalar_field::Scalar},
    gadgets::curve::CircuitBuilderEcGFp5,
};
use plonky2::{
    field::types::Field,
    hash::{
        hashing::hash_n_to_m_no_pad,
        poseidon::{PoseidonHash, PoseidonPermutation},
    },
    iop::target::Target,
    plonk::{
        circuit_builder::CircuitBuilder,
        config::{GenericConfig, PoseidonGoldilocksConfig},
    },
};
use plonky2_ecdsa::gadgets::nonnative::CircuitBuilderNonNative;
use plonky2_field::{
    extension::quintic::QuinticExtension, goldilocks_field::GoldilocksField, types::Sample,
};
use rand::RngCore;

use super::base_field::{CircuitBuilderGFp5, QuinticExtensionTarget};

pub const D: usize = 2;
pub type C = PoseidonGoldilocksConfig;
pub type F = <C as GenericConfig<D>>::F;

#[derive(Clone, Debug)]
pub struct SchnorrSecretKey(pub(crate) Scalar);

#[derive(Clone, Debug)]
pub struct SchnorrPublicKey(pub(crate) Point);

#[derive(Clone, Debug)]
pub struct SchnorrSignature {
    pub(crate) s: Scalar,
    pub(crate) e: Scalar,
}

pub fn schnorr_keygen(rng: &mut dyn RngCore) -> (SchnorrPublicKey, SchnorrSecretKey) {
    let sk = Scalar::sample(rng);
    let pk = Point::GENERATOR * sk;
    (SchnorrPublicKey(pk), SchnorrSecretKey(sk))
}

pub fn schnorr_sign(
    message: &[GoldilocksField],
    sk: &SchnorrSecretKey,
    rng: &mut dyn RngCore,
) -> SchnorrSignature {
    // sample random k
    let k = Scalar::sample(rng);
    // compute r = k*G
    let r = k * Point::GENERATOR;
    // e = H(r || M)
    let mut preimage = r.encode().0.to_vec();
    preimage.extend(message.iter());
    let e_elems = hash(&preimage);
    let e = Scalar::from_gfp5(QuinticExtension(e_elems));
    // s = k - e*sk
    let s = k - e * sk.0;

    // signature = (s, e)
    SchnorrSignature { s, e }
}

pub fn schnorr_verify_rust(
    message: &[GoldilocksField],
    pk: &SchnorrPublicKey,
    sig: &SchnorrSignature,
) -> bool {
    let r = sig.s * Point::GENERATOR + sig.e * pk.0;
    let mut preimage = r.encode().0.to_vec();
    preimage.extend(message.iter());
    let e_elems = hash(&preimage);
    let e = Scalar::from_gfp5(QuinticExtension(e_elems));
    e == sig.e
}

pub fn schnorr_verify_circuit(
    builder: &mut CircuitBuilder<F, D>,
    message: &[GoldilocksField],
    pk: &SchnorrPublicKey,
    sig: &SchnorrSignature,
) {
    let message = builder.constants(message);

    let s = builder.constant_nonnative::<Scalar>(sig.s);
    let e = builder.constant_nonnative::<Scalar>(sig.e);
    let g = builder.curve_generator();
    let pk_target = builder.curve_constant(pk.0.to_weierstrass());

    // r_v = s*G + e*pk
    let r_v = builder.curve_muladd_2(g, pk_target, &s, &e);

    // e_v = H(r_v || M)
    let mut preimage = builder.curve_encode_to_quintic_ext(r_v).0.to_vec();
    preimage.extend(message);
    let e_v_ext = QuinticExtensionTarget(hash_target(builder, &preimage));
    let e_v = builder.encode_quintic_ext_as_scalar(e_v_ext);

    // check e_v == e
    builder.connect_nonnative(&e, &e_v);
}

/// we define a hash function whose digest is 5 GFp5 elems
///
/// note: this doesn't apply any padding, so this is vulnerable to length extension attacks
fn hash(message: &[F]) -> [F; 5] {
    let mut res = [F::ZERO; 5];
    let out = hash_n_to_m_no_pad::<F, PoseidonPermutation<F>>(message, 5);
    res.copy_from_slice(&out[..5]);

    res
}

fn hash_target(builder: &mut CircuitBuilder<F, { D }>, message: &[Target]) -> [Target; 5] {
    builder.hash_n_to_m_no_pad::<PoseidonHash>(message.to_vec(), 5).try_into().unwrap()
}

#[cfg(test)]
mod tests {
    use plonky2::{iop::witness::PartialWitness, plonk::circuit_data::CircuitConfig};
    use rand::thread_rng;

    use super::*;

    #[test]
    fn test_verify_rust() {
        let mut rng = thread_rng();
        let (pk, sk) = schnorr_keygen(&mut rng);
        let message = b"Hello, world!";
        let message_f = message.map(|b| F::from_canonical_u8(b));
        let sig = schnorr_sign(&message_f, &sk, &mut rng);
        assert!(schnorr_verify_rust(&message_f, &pk, &sig));
    }

    #[test]
    fn test_verify_circuit() {
        // keygen and sign
        let mut rng = thread_rng();
        let (pk, sk) = schnorr_keygen(&mut rng);
        let message = b"Hello, world!";
        let message_f = message.map(|b| F::from_canonical_u8(b));
        let sig = schnorr_sign(&message_f, &sk, &mut rng);

        // Verify in circuit
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        schnorr_verify_circuit(&mut builder, &message_f, &pk, &sig);
        // build circuit
        builder.print_gate_counts(0);
        let pw = PartialWitness::new();
        let circuit = builder.build::<C>();
        let proof = circuit.prove(pw).unwrap();
        circuit.verify(proof).expect("verifier failed");
    }
}
