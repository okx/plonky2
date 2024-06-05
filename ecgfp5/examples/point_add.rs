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

use plonky2_ecgfp5::{
    curve::curve::Point,
    gadgets::curve::{CircuitBuilderEcGFp5, PartialWitnessCurve},
};
use plonky2_field::types::Sample;
use rand::thread_rng;

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

pub fn main() {
    init_logger();
    let mut rng = thread_rng();

    // curve point add
    println!("testing_curve_add...");

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let p = Point::sample(&mut rng);
    let p2 = Point::sample(&mut rng);
    let expected = p + p2;

    let p = builder.curve_constant(p.to_weierstrass());
    let p2 = builder.curve_constant(p2.to_weierstrass());
    let sum = builder.curve_add(p, p2);

    builder.print_gate_counts(0);
    let circuit = builder.build::<C>();

    let mut pw = PartialWitness::new();
    pw.set_curve_target(sum, expected.to_weierstrass());

    let CircuitData {
        prover_only,
        common,
        verifier_only: _,
    } = &circuit;

    let mut timing = TimingTree::new("prove", Level::Debug);
    let proof = prove(prover_only, common, pw, &mut timing).expect("prover failed");
    timing.print();

    circuit.verify(proof).expect("verifier failed");
}

fn init_logger() {
    let _ = try_init_from_env(Env::default().filter_or(DEFAULT_FILTER_ENV, "debug"));
}
