use env_logger::{try_init_from_env, Env, DEFAULT_FILTER_ENV};
use log::Level;
use plonky2::{
    field::{secp256k1_scalar::Secp256K1Scalar, types::Sample},
    iop::witness::PartialWitness,
    plonk::{
        circuit_builder::CircuitBuilder,
        circuit_data::{CircuitConfig, CircuitData},
        config::{GenericConfig, PoseidonGoldilocksConfig},
        prover::prove,
    },
    util::timing::TimingTree,
};
use plonky2_ecdsa::{
    curve::{
        curve_types::{Curve, CurveScalar},
        ecdsa::{sign_message, ECDSAPublicKey, ECDSASecretKey, ECDSASignature},
        secp256k1::Secp256K1,
    },
    gadgets::{
        curve::CircuitBuilderCurve,
        ecdsa::{verify_message_circuit, ECDSAPublicKeyTarget, ECDSASignatureTarget},
        nonnative::CircuitBuilderNonNative,
    },
};
fn main() {
    let _ = try_init_from_env(Env::default().filter_or(DEFAULT_FILTER_ENV, "debug"));

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    type Curve = Secp256K1;

    let pw = PartialWitness::new();
    let config = CircuitConfig::standard_ecc_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let msg = Secp256K1Scalar::rand();
    let msg_target = builder.constant_nonnative(msg);

    let sk = ECDSASecretKey::<Curve>(Secp256K1Scalar::rand());
    let pk = ECDSAPublicKey((CurveScalar(sk.0) * Curve::GENERATOR_PROJECTIVE).to_affine());

    let pk_target = ECDSAPublicKeyTarget(builder.constant_affine_point(pk.0));

    let sig = sign_message(msg, sk);

    let ECDSASignature { r, s } = sig;
    let r_target = builder.constant_nonnative(r);
    let s_target = builder.constant_nonnative(s);
    let sig_target = ECDSASignatureTarget { r: r_target, s: s_target };

    verify_message_circuit(&mut builder, msg_target, sig_target, pk_target);

    builder.print_gate_counts(0);
    let circuit = builder.build::<C>();
    let CircuitData { prover_only, common, verifier_only: _ } = &circuit;

    let mut timing = TimingTree::new("prove", Level::Debug);
    let proof = prove(prover_only, common, pw, &mut timing).expect("prover failed");
    timing.print();

    circuit.verify(proof).expect("verifier failed");
}
