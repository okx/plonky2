// toy example of a circuit that checks a ecdsa signatuse

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
use plonky2_ecdsa::gadgets::nonnative::CircuitBuilderNonNative;
use plonky2_ecgfp5::{
    curve::{curve::Point, scalar_field::Scalar},
    gadgets::curve::CircuitBuilderEcGFp5,
};
use plonky2_field::types::{Field, Sample};
use rand::thread_rng;

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

pub const SPONGE_RATE: usize = 8;

// we define a hash function whose digest is 5 GFp5 elems
// note: this doesn't apply any padding, so this is vulnerable to length extension attacks
// fn sig_hash(message: &[F]) -> [F; 5] {
//     let mut res = [F::ZERO; 5];
//     let out = hash_n_to_m_no_pad::<F, PoseidonPermutation>(message, 5);
//     res.copy_from_slice(&out[..5]);

//     res
// }

pub fn main() {
    init_logger();
    let mut rng = thread_rng();

    /* Generate Keypair */
    let sk = Scalar::sample(&mut rng);
    let pk = Point::GENERATOR * sk;

    /* Signing */
    // 1. message digest & encoding
    // let message_bytes = b"I'm going to be the king of pirates!";
    // let message_elems = message_bytes.map(|b| F::from_canonical_u8(b));
    // let e = sig_hash(&message_elems);
    // 2. z = Ln lestmost bits of e = scalar of e
    // let z = Scalar::from_gfp5(QuinticExtension(e));

    // 1-2. Sample random z
    let z = Scalar::sample(&mut rng);
    // 3-5
    let (k, r) = {
        // 3. sample random k
        let mut k = Scalar::sample(&mut rng);
        // 4. compute (x1, y1) = k*G
        let mut rr = Point::GENERATOR * k;
        // If x1 is zero, we need to sample a new k.
        while rr.x.is_zero() {
            k = Scalar::sample(&mut rng);
            rr = Point::GENERATOR * k;
        }
        // 5. r = x1 mod n
        let r = Scalar::from_gfp5(rr.x);
        (k, r)
    };
    // 6. s = k^{-1} * (z + r*sk) mod n
    let s = k.inverse() * (z + r * sk);

    /* Verify in Rust */
    {
        let u1 = z * s.inverse();
        let u2 = r * s.inverse();
        let point = u1 * Point::GENERATOR + u2 * pk;
        assert!(point.equals(k * Point::GENERATOR));
        assert_eq!(point.is_neutral(), false);
    }

    /* Verify in circuit */
    let config = CircuitConfig::wide_ecc_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    // Verify(msg, (r,s), pk)
    // Sig = (r, s)
    let r_target = builder.constant_nonnative::<Scalar>(r);
    let s_target = builder.constant_nonnative::<Scalar>(s);
    let pk_target = builder.curve_constant(pk.to_weierstrass());

    let g = builder.curve_generator();
    // 1. Check pk is a valid curve point
    {

        // let a = builder.constant_nonnative(WeierstrassPoint::A);
        // let b = builder.constant_nonnative(WeierstrassPoint::B);
        // let px = &pk_target.0.0[0];
        // let py = &pk_target.0.0[1];
        // let y_squared = builder.mul_nonnative(py, py);
        // let x_squared = builder.mul_nonnative(px, px);
        // let x_cuded = builder.mul_nonnative(&x_squared, px);
        // let a_x = builder.mul_nonnative(&a, px);
        // let a_x_plus_b = builder.add_nonnative(&a_x, &b);
        // let rhs = builder.add_nonnative(&x_cuded, &a_x_plus_b);
        // builder.connect_nonnative(&y_squared, &rhs);
    }

    // 2. Check r and s are in [1, n-1]

    // 3. e = HASH(msg)
    // 4. z = Ln leastmost bits of e
    // We get z from witness directly
    let z_target = builder.constant_nonnative::<Scalar>(z);

    // 5. u1 = z * s^{-1} mod n, u2 = r * s^{-1} mod n
    let s_inv = builder.inv_nonnative(&s_target);
    let u1 = builder.mul_nonnative(&z_target, &s_inv);
    let u2 = builder.mul_nonnative(&r_target, &s_inv);

    // 6. point = u1*G + u2*pk
    let point = builder.curve_muladd_2(g, pk_target, &u1, &u2);

    // 7. Check point = r*G and point != O (identity element)
    let r_g = builder.curve_scalar_mul(g, &r_target);
    builder.curve_eq(point, r_g);
    builder.curve_assert_not_zero(point);

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
