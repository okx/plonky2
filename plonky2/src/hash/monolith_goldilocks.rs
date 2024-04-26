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

mod monolith_mds_12 {
    use crate::hash::monolith::split;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::{Field, PrimeField64};

    /// This module contains helper functions as well as constants used to perform a 12x12 vector-matrix
    /// multiplication. The special form of our MDS matrix i.e. being circulant, allows us to reduce
    /// the vector-matrix multiplication to a Hadamard product of two vectors in "frequency domain".
    /// This follows from the simple fact that every circulant matrix has the columns of the discrete
    /// Fourier transform matrix as orthogonal eigenvectors.
    /// The implementation also avoids the use of 3-point FFTs, and 3-point iFFTs, and substitutes that
    /// with explicit expressions. It also avoids, due to the form of our matrix in the frequency domain,
    /// divisions by 2 and repeated modular reductions. This is because of our explicit choice of
    /// an MDS matrix that has small powers of 2 entries in frequency domain.
    /// The following implementation has benefited greatly from the discussions and insights of
    /// Hamish Ivey-Law and Jacqueline Nabaglo of Polygon Zero.
    /// The circulant matrix is identified by its first row: [7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8].

    // MDS matrix in frequency domain.
    // More precisely, this is the output of the three 4-point (real) FFTs of the first column of
    // the MDS matrix i.e. just before the multiplication with the appropriate twiddle factors
    // and application of the final four 3-point FFT in order to get the full 12-point FFT.
    // The entries have been scaled appropriately in order to avoid divisions by 2 in iFFT2 and iFFT4.
    // The code to generate the matrix in frequency domain is based on an adaptation of a code, to generate
    // MDS matrices efficiently in original domain, that was developed by the Polygon Zero team.
    const MDS_FREQ_BLOCK_ONE: [i64; 3] = [16, 8, 16];
    const MDS_FREQ_BLOCK_TWO: [(i64, i64); 3] = [(-1, 2), (-1, 1), (4, 8)];
    const MDS_FREQ_BLOCK_THREE: [i64; 3] = [-8, 1, 1];

    pub(crate) fn mds_multiply_u128(state: &mut [u128; 12], round_constants: &[u64; 12]) {
        // Using the linearity of the operations we can split the state into a low||high decomposition
        // and operate on each with no overflow and then combine/reduce the result to a field element.
        let mut state_l = [0u64; 12];
        let mut state_h = [0u64; 12];

        for r in 0..12 {
            let s = state[r];
            state_h[r] = (s >> 32) as u64;
            state_l[r] = (s as u32) as u64;
        }

        let state_h = mds_multiply_freq(state_h);
        let state_l = mds_multiply_freq(state_l);

        for r in 0..12 {
            // Both have less than 40 bits
            state[r] = state_l[r] as u128 + ((state_h[r] as u128) << 32);
            state[r] += round_constants[r] as u128;
            state[r] = GoldilocksField::from_noncanonical_u96(split(state[r])).0 as u128;
        }
    }

    // We use split 3 x 4 FFT transform in order to transform our vectors into the frequency domain.
    #[inline(always)]
    pub(crate) fn mds_multiply_freq(state: [u64; 12]) -> [u64; 12] {
        let [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11] = state;

        let (u0, u1, u2) = fft4_real([s0, s3, s6, s9]);
        let (u4, u5, u6) = fft4_real([s1, s4, s7, s10]);
        let (u8, u9, u10) = fft4_real([s2, s5, s8, s11]);

        // This where the multiplication in frequency domain is done. More precisely, and with
        // the appropriate permuations in between, the sequence of
        // 3-point FFTs --> multiplication by twiddle factors --> Hadamard multiplication -->
        // 3 point iFFTs --> multiplication by (inverse) twiddle factors
        // is "squashed" into one step composed of the functions "block1", "block2" and "block3".
        // The expressions in the aforementioned functions are the result of explicit computations
        // combined with the Karatsuba trick for the multiplication of Complex numbers.

        let [v0, v4, v8] = block1([u0, u4, u8], MDS_FREQ_BLOCK_ONE);
        let [v1, v5, v9] = block2([u1, u5, u9], MDS_FREQ_BLOCK_TWO);
        let [v2, v6, v10] = block3([u2, u6, u10], MDS_FREQ_BLOCK_THREE);
        // The 4th block is not computed as it is similar to the 2nd one, up to complex conjugation,
        // and is, due to the use of the real FFT and iFFT, redundant.

        let [s0, s3, s6, s9] = ifft4_real_unreduced((v0, v1, v2));
        let [s1, s4, s7, s10] = ifft4_real_unreduced((v4, v5, v6));
        let [s2, s5, s8, s11] = ifft4_real_unreduced((v8, v9, v10));

        [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11]
    }

    #[inline(always)]
    fn block1(x: [i64; 3], y: [i64; 3]) -> [i64; 3] {
        let [x0, x1, x2] = x;
        let [y0, y1, y2] = y;
        let z0 = x0 * y0 + x1 * y2 + x2 * y1;
        let z1 = x0 * y1 + x1 * y0 + x2 * y2;
        let z2 = x0 * y2 + x1 * y1 + x2 * y0;

        [z0, z1, z2]
    }

    #[inline(always)]
    fn block2(x: [(i64, i64); 3], y: [(i64, i64); 3]) -> [(i64, i64); 3] {
        let [(x0r, x0i), (x1r, x1i), (x2r, x2i)] = x;
        let [(y0r, y0i), (y1r, y1i), (y2r, y2i)] = y;
        let x0s = x0r + x0i;
        let x1s = x1r + x1i;
        let x2s = x2r + x2i;
        let y0s = y0r + y0i;
        let y1s = y1r + y1i;
        let y2s = y2r + y2i;

        // Compute x0​y0 ​− ix1​y2​ − ix2​y1​ using Karatsuba for complex numbers multiplication
        let m0 = (x0r * y0r, x0i * y0i);
        let m1 = (x1r * y2r, x1i * y2i);
        let m2 = (x2r * y1r, x2i * y1i);
        let z0r = (m0.0 - m0.1) + (x1s * y2s - m1.0 - m1.1) + (x2s * y1s - m2.0 - m2.1);
        let z0i = (x0s * y0s - m0.0 - m0.1) + (-m1.0 + m1.1) + (-m2.0 + m2.1);
        let z0 = (z0r, z0i);

        // Compute x0​y1​ + x1​y0​ − ix2​y2 using Karatsuba for complex numbers multiplication
        let m0 = (x0r * y1r, x0i * y1i);
        let m1 = (x1r * y0r, x1i * y0i);
        let m2 = (x2r * y2r, x2i * y2i);
        let z1r = (m0.0 - m0.1) + (m1.0 - m1.1) + (x2s * y2s - m2.0 - m2.1);
        let z1i = (x0s * y1s - m0.0 - m0.1) + (x1s * y0s - m1.0 - m1.1) + (-m2.0 + m2.1);
        let z1 = (z1r, z1i);

        // Compute x0​y2​ + x1​y1 ​+ x2​y0​ using Karatsuba for complex numbers multiplication
        let m0 = (x0r * y2r, x0i * y2i);
        let m1 = (x1r * y1r, x1i * y1i);
        let m2 = (x2r * y0r, x2i * y0i);
        let z2r = (m0.0 - m0.1) + (m1.0 - m1.1) + (m2.0 - m2.1);
        let z2i = (x0s * y2s - m0.0 - m0.1) + (x1s * y1s - m1.0 - m1.1) + (x2s * y0s - m2.0 - m2.1);
        let z2 = (z2r, z2i);

        [z0, z1, z2]
    }

    #[inline(always)]
    fn block3(x: [i64; 3], y: [i64; 3]) -> [i64; 3] {
        let [x0, x1, x2] = x;
        let [y0, y1, y2] = y;
        let z0 = x0 * y0 - x1 * y2 - x2 * y1;
        let z1 = x0 * y1 + x1 * y0 - x2 * y2;
        let z2 = x0 * y2 + x1 * y1 + x2 * y0;

        [z0, z1, z2]
    }

    /// Real 2-FFT over u64 integers.
    #[inline(always)]
    fn fft2_real(x: [u64; 2]) -> [i64; 2] {
        [(x[0] as i64 + x[1] as i64), (x[0] as i64 - x[1] as i64)]
    }

    /// Real 2-iFFT over u64 integers.
    /// Division by two to complete the inverse FFT is expected to be performed ***outside*** of this function.
    #[inline(always)]
    fn ifft2_real_unreduced(y: [i64; 2]) -> [u64; 2] {
        [(y[0] + y[1]) as u64, (y[0] - y[1]) as u64]
    }

    /// Real 4-FFT over u64 integers.
    #[inline(always)]
    fn fft4_real(x: [u64; 4]) -> (i64, (i64, i64), i64) {
        let [z0, z2] = fft2_real([x[0], x[2]]);
        let [z1, z3] = fft2_real([x[1], x[3]]);
        let y0 = z0 + z1;
        let y1 = (z2, -z3);
        let y2 = z0 - z1;
        (y0, y1, y2)
    }

    /// Real 4-iFFT over u64 integers.
    /// Division by four to complete the inverse FFT is expected to be performed ***outside*** of this function.
    #[inline(always)]
    fn ifft4_real_unreduced(y: (i64, (i64, i64), i64)) -> [u64; 4] {
        let z0 = y.0 + y.2;
        let z1 = y.0 - y.2;
        let z2 = y.1 .0;
        let z3 = -y.1 .1;

        let [x0, x2] = ifft2_real_unreduced([z0, z2]);
        let [x1, x3] = ifft2_real_unreduced([z1, z3]);

        [x0, x1, x2, x3]
    }
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

    /*
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
    */
}