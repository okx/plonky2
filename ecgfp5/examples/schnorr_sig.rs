// toy example of a circuit that checks a schnorr signatuse

use env_logger::{try_init_from_env, Env, DEFAULT_FILTER_ENV};
use log::Level;
use plonky2::{
    iop::witness::PartialWitness,
    plonk::{
        circuit_builder::CircuitBuilder,
        circuit_data::{CircuitConfig, CircuitData},
        config::{GenericConfig, PoseidonGoldilocksConfig},
        prover::prove,
    },
    util::timing::TimingTree,
};

use plonky2_ecgfp5::gadgets::schnorr::{
    schnorr_keygen, schnorr_sign, schnorr_verify_circuit, schnorr_verify_rust,
};
use plonky2_field::types::Field;
use rand::thread_rng;

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

pub const SPONGE_RATE: usize = 8;

pub fn main() {
    init_logger();

    // Keygen
    let mut rng = thread_rng();
    let (pk, sk) = schnorr_keygen(&mut rng);
    // Sign
    let message = b"Hello, world!";
    let message_f = message.map(|b| F::from_canonical_u8(b));
    let sig = schnorr_sign(&message_f, &sk, &mut rng);
    // Verify in Rust
    assert!(schnorr_verify_rust(&message_f, &pk, &sig));

    // Verify in circuit
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    schnorr_verify_circuit(&mut builder, &message_f, &pk, &sig);
    // build circuit
    builder.print_gate_counts(0);
    let circuit = builder.build::<C>();
    let CircuitData { prover_only, common, verifier_only: _ } = &circuit;

    let pw = PartialWitness::new();
    let mut timing = TimingTree::new("prove", Level::Debug);
    let proof = prove(prover_only, common, pw, &mut timing).expect("prover failed");
    timing.print();

    circuit.verify(proof).expect("verifier failed");
}

fn init_logger() {
    let _ = try_init_from_env(Env::default().filter_or(DEFAULT_FILTER_ENV, "debug"));
}
