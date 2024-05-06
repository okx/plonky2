use serde::Serialize;

use super::{monolith::{Monolith, MonolithHash, LOOKUP_BITS, N_ROUNDS, SPONGE_WIDTH}, poseidon::PoseidonHash};
use crate::{field::goldilocks_field::GoldilocksField, plonk::config::GenericConfig};
use crate::field::extension::quadratic::QuadraticExtension;

pub const MONOLITH_ROUND_CONSTANTS: [[u64; SPONGE_WIDTH]; N_ROUNDS + 1] = match LOOKUP_BITS {
    8 => [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [
            13596126580325903823,
            5676126986831820406,
            11349149288412960427,
            3368797843020733411,
            16240671731749717664,
            9273190757374900239,
            14446552112110239438,
            4033077683985131644,
            4291229347329361293,
            13231607645683636062,
            1383651072186713277,
            8898815177417587567,
        ],
        [
            2383619671172821638,
            6065528368924797662,
            16737578966352303081,
            2661700069680749654,
            7414030722730336790,
            18124970299993404776,
            9169923000283400738,
            15832813151034110977,
            16245117847613094506,
            11056181639108379773,
            10546400734398052938,
            8443860941261719174,
        ],
        [
            15799082741422909885,
            13421235861052008152,
            15448208253823605561,
            2540286744040770964,
            2895626806801935918,
            8644593510196221619,
            17722491003064835823,
            5166255496419771636,
            1015740739405252346,
            4400043467547597488,
            5176473243271652644,
            4517904634837939508,
        ],
        [
            18341030605319882173,
            13366339881666916534,
            6291492342503367536,
            10004214885638819819,
            4748655089269860551,
            1520762444865670308,
            8393589389936386108,
            11025183333304586284,
            5993305003203422738,
            458912836931247573,
            5947003897778655410,
            17184667486285295106,
        ],
        [
            15710528677110011358,
            8929476121507374707,
            2351989866172789037,
            11264145846854799752,
            14924075362538455764,
            10107004551857451916,
            18325221206052792232,
            16751515052585522105,
            15305034267720085905,
            15639149412312342017,
            14624541102106656564,
            3542311898554959098,
        ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    16 => [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [
            18336847912085310782,
            16981085523750439062,
            13429031554613510028,
            14626146163475314696,
            17132599202993726423,
            8006190003318006507,
            11343032213505247196,
            14124666955091711556,
            8430380888588022602,
            8028059853581205264,
            10576927460643802925,
            264807431271531499,
        ],
        [
            4974395136075591328,
            12767804748363387455,
            4282984340606842818,
            9962032970357721094,
            13290063373589851073,
            682582873026109162,
            1443405731716023143,
            1102365195228642031,
            2045097484032658744,
            4705239685543555952,
            7749631247106030298,
            14498144818552307386,
        ],
        [
            2422278540391021322,
            16279967701033470233,
            11928233299971145130,
            289434792182172450,
            9247027096240775287,
            13564504933984041357,
            13716745789926357653,
            17062841883145120930,
            4787227470665224131,
            3941766098336857538,
            10415914353862079098,
            2031314485617648836,
        ],
        [
            15757165366981665927,
            5316332562976837179,
            6408794885240907199,
            15433272772010162147,
            16177208255639089922,
            6438767259788073242,
            1850299052911296965,
            12036975040590254229,
            14345891531575426146,
            7475247528756702227,
            3952963486672887438,
            15765121003485081487,
        ],
        [
            8288959343482523513,
            6774706297840606862,
            15381728973932837801,
            15052040954696745676,
            9925792545634777672,
            9264032288608603069,
            11473431200717914600,
            2655107155645324988,
            8397223040566002342,
            9234186621285090301,
            1463633689352888362,
            18441834386923465669,
        ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    _ => panic!("Unsupported lookup size"),
};

pub const MONOLITH_MAT_12: [[u64; SPONGE_WIDTH]; SPONGE_WIDTH] = [
    [7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8],
    [8, 7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21],
    [21, 8, 7, 23, 8, 26, 13, 10, 9, 7, 6, 22],
    [22, 21, 8, 7, 23, 8, 26, 13, 10, 9, 7, 6],
    [6, 22, 21, 8, 7, 23, 8, 26, 13, 10, 9, 7],
    [7, 6, 22, 21, 8, 7, 23, 8, 26, 13, 10, 9],
    [9, 7, 6, 22, 21, 8, 7, 23, 8, 26, 13, 10],
    [10, 9, 7, 6, 22, 21, 8, 7, 23, 8, 26, 13],
    [13, 10, 9, 7, 6, 22, 21, 8, 7, 23, 8, 26],
    [26, 13, 10, 9, 7, 6, 22, 21, 8, 7, 23, 8],
    [8, 26, 13, 10, 9, 7, 6, 22, 21, 8, 7, 23],
    [23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8, 7],
];

impl Monolith for GoldilocksField {
    #[cfg(feature = "default-sponge-params")]
    fn concrete_u128(state_u128: &mut [u128; SPONGE_WIDTH], round_constants: &[u64; SPONGE_WIDTH]) {
        mds_multiply_u128(state_u128, round_constants)
    }

    const ROUND_CONSTANTS: [[u64; SPONGE_WIDTH]; N_ROUNDS + 1] = MONOLITH_ROUND_CONSTANTS;

    const MAT_12: [[u64; SPONGE_WIDTH]; SPONGE_WIDTH] = MONOLITH_MAT_12;
}

/// Configuration using Monolith over the Goldilocks field.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize)]
pub struct MonolithGoldilocksConfig;
impl GenericConfig<2> for MonolithGoldilocksConfig {
    type F = GoldilocksField;
    type FE = QuadraticExtension<Self::F>;
    type Hasher = MonolithHash;
    type InnerHasher = PoseidonHash;
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use rstest::rstest;
    use serial_test::serial;

    use crate::gates::gadget::tests::{prove_circuit_with_hash, recursive_proof, test_monolith_hash_circuit};
    use crate::gates::monolith::generate_config_for_monolith_gate;
    use crate::hash::hash_types::RichField;
    use crate::hash::monolith::*;
    use crate::field::extension::Extendable;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::hash::monolith_goldilocks::MonolithGoldilocksConfig;
    use crate::hash::poseidon::PoseidonHash;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{AlgebraicHasher, GenericConfig, Hasher, PoseidonGoldilocksConfig};

    use self::test::check_test_vectors;

    #[test]
    fn test_vectors() {
        // Test inputs are:
        // 1. 0..WIDTH-1

        #[rustfmt::skip]
            let test_vectors12: Vec<([u64; 12], [u64; 12])> = match LOOKUP_BITS {
            8 => vec![
                ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ],
                 [5867581605548782913, 588867029099903233, 6043817495575026667, 805786589926590032, 9919982299747097782, 6718641691835914685, 7951881005429661950, 15453177927755089358, 974633365445157727, 9654662171963364206, 6281307445101925412, 13745376999934453119]),
            ],
            16 => vec![
                ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ],
                 [15270549627416999494, 2608801733076195295, 2511564300649802419, 14351608014180687564, 4101801939676807387, 234091379199311770, 3560400203616478913, 17913168886441793528, 7247432905090441163, 667535998170608897, 5848119428178849609, 7505720212650520546]),
            ],
            _ => panic!("unsupported lookup size"),
        };

        check_test_vectors::<GoldilocksField>(test_vectors12);
    }

    // helper struct employed to bind a Hasher implementation `H` with the circuit configuration to
    // be employed to build the circuit when such Hasher `H` is employed in the circuit
    struct HasherConfig<
        const D: usize,
        F: RichField + Monolith + Extendable<D>,
        H: Hasher<F> + AlgebraicHasher<F>,
    > {
        field: PhantomData<F>,
        hasher: PhantomData<H>,
        circuit_config: CircuitConfig,
    }

    #[rstest]
    #[serial]
    fn test_circuit_with_hash_functions<
        F: RichField + Monolith + Extendable<D>,
        C: GenericConfig<D, F = F>,
        H: Hasher<F> + AlgebraicHasher<F>,
        const D: usize,
    >(
        #[values(PoseidonGoldilocksConfig, MonolithGoldilocksConfig)] _c: C,
        #[values(HasherConfig::<2, GoldilocksField, PoseidonHash> {
        field: PhantomData::default(),
        hasher: PhantomData::default(),
        circuit_config: CircuitConfig::standard_recursion_config(),
        }, HasherConfig::<2, GoldilocksField , MonolithHash> {
        field: PhantomData::default(),
        hasher: PhantomData::default(),
        circuit_config: generate_config_for_monolith_gate::<GoldilocksField,2>(),
        })]
        config: HasherConfig<D, F, H>,
    ) {
        let _ = env_logger::builder().is_test(true).try_init();

        let (cd, proof) =
            prove_circuit_with_hash::<F, C, D, H>(config.circuit_config, 4096, true).unwrap();

        cd.verify(proof).unwrap()
    }
    // helper struct employed to bind a GenericConfig `C` with the circuit configuration
    // to be employed to build the circuit when such `C` is employed in the circuit
    struct HashConfig<const D: usize, C: GenericConfig<D>> {
        gen_config: PhantomData<C>,
        circuit_config: CircuitConfig,
    }

    #[rstest]
    #[serial]
    fn test_recursive_circuit_with_hash_functions<
        F: RichField + Monolith + Extendable<D>,
        C: GenericConfig<D, F = F>,
        InnerC: GenericConfig<D, F = F>,
        const D: usize,
    >(
        #[values(PoseidonGoldilocksConfig, MonolithGoldilocksConfig)] _c: C,
        #[values(HashConfig::<2, PoseidonGoldilocksConfig> {
        gen_config: PhantomData::default(),
        circuit_config: CircuitConfig::standard_recursion_config(),
        }, HashConfig::<2, MonolithGoldilocksConfig> {
        gen_config: PhantomData::default(),
        circuit_config: generate_config_for_monolith_gate::<GoldilocksField,2>(),
        })]
        inner_conf: HashConfig<D, InnerC>,
    ) where
        C::Hasher: AlgebraicHasher<F>,
        InnerC::Hasher: AlgebraicHasher<F>,
    {
        let _ = env_logger::builder().is_test(true).try_init();

        let (cd, proof) = prove_circuit_with_hash::<F, InnerC, D, PoseidonHash>(
            CircuitConfig::standard_recursion_config(),
            2048,
            false,
        )
        .unwrap();

        println!("base proof generated");

        println!("base circuit size: {}", cd.common.degree_bits());

        let (rec_cd, rec_proof) =
            recursive_proof::<F, C, InnerC, D>(proof, &cd, &inner_conf.circuit_config).unwrap();

        println!(
            "recursive proof generated, recursion circuit size: {}",
            rec_cd.common.degree_bits()
        );

        rec_cd.verify(rec_proof).unwrap();
    }

    #[test]
    fn test_monolith_hash() {
        const D: usize = 2;
        type C = MonolithGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let config = generate_config_for_monolith_gate::<F, D>();
        let _ = env_logger::builder().is_test(true).try_init();
        test_monolith_hash_circuit::<F, C, D>(config)
    }

}