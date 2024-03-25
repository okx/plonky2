//! Implementation of the Poseidon hash function, as described in
//! <https://eprint.iacr.org/2019/458.pdf>

use alloc::vec;
use core::iter::repeat;
use std::fmt::Debug;

use plonky2_field::goldilocks_field::GoldilocksField;

#[cfg(target_feature = "avx2")]
use super::arch::x86_64::poseidon2_goldilocks_avx2::{
    add_rc_avx, internal_layer_avx, matmul_internal_avx, permute_mut_avx
};
#[cfg(target_feature = "avx2")]
use super::arch::x86_64::goldilocks_avx2::sbox_avx;
use super::hash_types::{HashOutTarget, NUM_HASH_OUT_ELTS};
use super::poseidon::PoseidonHash;
use crate::field::extension::Extendable;
use crate::gates::poseidon::PoseidonGate;
use crate::hash::hash_types::{HashOut, RichField};
use crate::hash::hashing::PlonkyPermutation;
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::{AlgebraicHasher, Hasher, HasherType};

pub const SPONGE_RATE: usize = 8;
pub const SPONGE_CAPACITY: usize = 4;
pub const SPONGE_WIDTH: usize = SPONGE_RATE + SPONGE_CAPACITY;

pub const MATRIX_DIAG_12_GOLDILOCKS: [u64; 12] = [
    0xc3b6c08e23ba9300,
    0xd84b5de94a324fb6,
    0x0d0c371c5b35b84f,
    0x7964f570e7188037,
    0x5daf18bbd996604b,
    0x6743bc47b9595257,
    0x5528b9362c59bb70,
    0xac45e25b7127b68b,
    0xa2077d7dfbb606b5,
    0xf3faac6faee378ae,
    0x0c6388b51545e883,
    0xd27dbb6944917b60,
];

pub const RC12: [[u64; 12]; 30] = [
    [
        1431286215153372998,
        3509349009260703107,
        2289575380984896342,
        10625215922958251110,
        17137022507167291684,
        17143426961497010024,
        9589775313463224365,
        7736066733515538648,
        2217569167061322248,
        10394930802584583083,
        4612393375016695705,
        5332470884919453534,
    ],
    [
        8724526834049581439,
        17673787971454860688,
        2519987773101056005,
        7999687124137420323,
        18312454652563306701,
        15136091233824155669,
        1257110570403430003,
        5665449074466664773,
        16178737609685266571,
        52855143527893348,
        8084454992943870230,
        2597062441266647183,
    ],
    [
        3342624911463171251,
        6781356195391537436,
        4697929572322733707,
        4179687232228901671,
        17841073646522133059,
        18340176721233187897,
        13152929999122219197,
        6306257051437840427,
        4974451914008050921,
        11258703678970285201,
        581736081259960204,
        18323286026903235604,
    ],
    [
        10250026231324330997,
        13321947507807660157,
        13020725208899496943,
        11416990495425192684,
        7221795794796219413,
        2607917872900632985,
        2591896057192169329,
        10485489452304998145,
        9480186048908910015,
        2645141845409940474,
        16242299839765162610,
        12203738590896308135,
    ],
    [5395176197344543510, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [17941136338888340715, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [7559392505546762987, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [549633128904721280, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [15658455328409267684, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [10078371877170729592, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2349868247408080783, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [13105911261634181239, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [12868653202234053626, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [9471330315555975806, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4580289636625406680, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [13222733136951421572, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4555032575628627551, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [7619130111929922899, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4547848507246491777, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5662043532568004632, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [15723873049665279492, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [13585630674756818185, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [6990417929677264473, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [6373257983538884779, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1005856792729125863, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [17850970025369572891, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        14306783492963476045,
        12653264875831356889,
        10887434669785806501,
        7221072982690633460,
        9953585853856674407,
        13497620366078753434,
        18140292631504202243,
        17311934738088402529,
        6686302214424395771,
        11193071888943695519,
        10233795775801758543,
        3362219552562939863,
    ],
    [
        8595401306696186761,
        7753411262943026561,
        12415218859476220947,
        12517451587026875834,
        3257008032900598499,
        2187469039578904770,
        657675168296710415,
        8659969869470208989,
        12526098871288378639,
        12525853395769009329,
        15388161689979551704,
        7880966905416338909,
    ],
    [
        2911694411222711481,
        6420652251792580406,
        323544930728360053,
        11718666476052241225,
        2449132068789045592,
        17993014181992530560,
        15161788952257357966,
        3788504801066818367,
        1282111773460545571,
        8849495164481705550,
        8380852402060721190,
        2161980224591127360,
    ],
    [
        2440151485689245146,
        17521895002090134367,
        13821005335130766955,
        17513705631114265826,
        17068447856797239529,
        17964439003977043993,
        5685000919538239429,
        11615940660682589106,
        2522854885180605258,
        12584118968072796115,
        17841258728624635591,
        10821564568873127316,
    ],
];

// shifted RC12
pub const RC12S: [[u64; 12]; 30] = [
[ 10654658252008148806, 12732721046115478915, 11512947417839672150, 1401843886103475302, 7913650470312515876, 7920054924642234216, 366403276608448557, 16959438770370314456, 11440941203916098056, 1171558765729807275, 13835765411871471513, 14555842921774229342,  ],
[ 17947898870904357247, 8450415934600084880, 11743359809955831813, 17223059160992196131, 9089082615708530893, 5912719196969379861, 10480482607258205811, 14888821111321440581, 6955365572830490763, 9276227180382669156, 17307827029798646038, 11820434478121422991,  ],
[ 12565996948317947059, 16004728232246313244, 13921301609177509515, 13403059269083677479, 8617701609667357251, 9116804684378412089, 3929557962267443389, 15529629088292616235, 14197823950862826729, 2035331642115509393, 9805108118114736012, 9099913990048459796,  ],
[ 1026654194469555189, 4098575470952884349, 3797353172044721135, 2193618458570416876, 16445167831650995221, 11831289909755408793, 11815268094046945137, 1262117415450222337, 256814012054134207, 11868513882264716282, 7018927802910386802, 2980366554041532327,  ],
[ 14618548234199319318, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 8717764302033564907, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 16782764542401538795, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 9773005165759497088, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 6435083291554491876, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 854999840315953784, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 11573240284262856591, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 3882539224779405431, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 3645281165379277818, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 247958278701199998, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 13803661673480182488, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 3999361100096645764, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 13778404612483403359, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 16842502148784698707, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 13771220544101267585, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 14885415569422780440, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 6500501012810503684, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 4362258637902042377, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 16213789966532040281, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 15596630020393660587, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 10229228829583901671, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 8627597988514797083, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  ],
[ 5083411456108700237, 3429892838976581081, 1664062632931030693, 16444445019545409268, 730213817001898599, 4274248329223977626, 8916920594649426435, 8088562701233626721, 15909674251279171579, 1969699852088919711, 1010423738946982735, 12585591589417715671,  ],
[ 17818773343550962569, 16976783299797802369, 3191846822621445139, 3294079550172100026, 12480380069755374307, 11410841076433680578, 9881047205151486223, 17883341906324984797, 3302726834433602831, 3302481358914233521, 6164789653124775896, 17104338942271114717,  ],
[ 12135066448077487289, 15644024288647356214, 9546916967583135861, 2495294439197465417, 11672504105643821400, 8769642145137754752, 5938416915402582158, 13011876837921594175, 10505483810315321379, 18072867201336481358, 17604224438915496998, 11385352261445903168,  ],
[ 11663523522544020954, 8298522965235358559, 4597633298275991147, 8290333594259490018, 7845075819942463721, 8741066967122268185, 14908372956393015237, 2392568623827813298, 11746226922035381066, 3360746931218020307, 8617886691769859783, 1598192532018351508,  ]
];

extern crate alloc;

// The t x t matrix M_E := circ(2M_4, M_4, ..., M_4), where M_4 is the 4 x 4 matrix
// [ 5 7 1 3 ]
// [ 4 6 1 1 ]
// [ 1 3 5 7 ]
// [ 1 1 4 6 ].
// The permutation calculation is based on Appendix B from the Poseidon2 paper.
#[derive(Copy, Clone, Default)]
pub struct Poseidon2MEMatrix;

// Multiply a 4-element vector x by M_4, in place.
// This uses the formula from the start of Appendix B, with multiplications unrolled into additions.
pub fn apply_m_4<F>(x: &mut [F])
where
    F: RichField,
{
    let t0 = x[0].clone() + x[1].clone();
    let t1 = x[2].clone() + x[3].clone();
    let t2 = x[1].clone() + x[1].clone() + t1.clone();
    let t3 = x[3].clone() + x[3].clone() + t0.clone();
    let t4 = t1.clone() + t1.clone() + t1.clone() + t1 + t3.clone();
    let t5 = t0.clone() + t0.clone() + t0.clone() + t0 + t2.clone();
    let t6 = t3 + t5.clone();
    let t7 = t2 + t4.clone();
    x[0] = t6;
    x[1] = t5;
    x[2] = t7;
    x[3] = t4;
}

trait P2Permutation<T: Clone>: Clone + Sync {
    fn permute(&self, mut input: T) -> T {
        self.permute_mut(&mut input);
        input
    }

    fn permute_mut(&self, input: &mut T);
}

impl<F> P2Permutation<[F; SPONGE_WIDTH]> for Poseidon2MEMatrix
where
    F: RichField,
{
    #[cfg(not(target_feature = "avx2"))]
    fn permute_mut(&self, state: &mut [F; SPONGE_WIDTH]) {
        // First, we apply M_4 to each consecutive four elements of the state.
        // In Appendix B's terminology, this replaces each x_i with x_i'.
        for i in (0..SPONGE_WIDTH).step_by(4) {
            apply_m_4(&mut state[i..i + 4]);
        }

        // Now, we apply the outer circulant matrix (to compute the y_i values).

        // We first precompute the four sums of every four elements.
        let sums: [F; 4] = core::array::from_fn(|k| {
            (0..SPONGE_WIDTH)
                .step_by(4)
                .map(|j| state[j + k].clone())
                .sum::<F>()
        });

        // The formula for each y_i involves 2x_i' term and x_j' terms for each j that equals i mod 4.
        // In other words, we can add a single copy of x_i' to the appropriate one of our precomputed sums
        for i in 0..SPONGE_WIDTH {
            state[i] += sums[i % 4].clone();
        }
    }

    #[cfg(target_feature = "avx2")]
    fn permute_mut(&self, state: &mut [F; SPONGE_WIDTH]) {
        permute_mut_avx(state);
    }
}

#[derive(Debug, Clone, Default)]
struct DiffusionMatrixGoldilocks;

pub fn matmul_internal<F: RichField>(
    state: &mut [F; SPONGE_WIDTH],
    mat_internal_diag_m_1: [u64; SPONGE_WIDTH],
) {
    let sum: F = state.iter().cloned().sum();
    for i in 0..SPONGE_WIDTH {
        state[i] *= F::from_canonical_u64(mat_internal_diag_m_1[i]);
        state[i] += sum.clone();
    }
}

impl<F: RichField> P2Permutation<[F; 12]> for DiffusionMatrixGoldilocks {
    #[cfg(not(target_feature = "avx2"))]
    fn permute_mut(&self, state: &mut [F; 12]) {
        matmul_internal::<F>(state, MATRIX_DIAG_12_GOLDILOCKS);
    }

    #[cfg(target_feature = "avx2")]
    fn permute_mut(&self, state: &mut [F; 12]) {
        matmul_internal_avx::<F>(state, MATRIX_DIAG_12_GOLDILOCKS);
    }
}

pub trait Poseidon2: RichField {
    // const WIDTH: usize = 12;
    // const D: u64 = 7;
    const ROUNDS_F: usize = 8;
    const ROUNDS_P: usize = 22;

    #[inline]
    fn add_rc<F>(state: &mut [F; SPONGE_WIDTH], rc: &[u64; SPONGE_WIDTH])
    where
        F: RichField,
    {
        for i in 0..SPONGE_WIDTH {
            state[i] = state[i] + F::from_canonical_u64(rc[i]);
        }
    }

    #[inline]
    fn sbox_p<F>(input: &F) -> F
    where
        F: RichField,
    {
        // this is inefficient, so we change to the one below
        // input.exp_u64(7)
        let x2 = (*input) * (*input);
        let x4 = x2 * x2;
        let x3 = x2 * (*input);
        x3 * x4
    }

    #[inline]
    fn sbox<F>(state: &mut [F; SPONGE_WIDTH])
    where
        F: RichField,
    {
        for i in 0..SPONGE_WIDTH {
            state[i] = Self::sbox_p(&state[i]);
        }
    }

    #[inline]
    #[cfg(not(target_feature = "avx2"))]
    fn poseidon2(state: &mut [Self; SPONGE_WIDTH]) {
        let external_linear_layer = Poseidon2MEMatrix;

        // The initial linear layer.
        external_linear_layer.permute_mut(state);

        // The first half of the external rounds.
        let rounds = Self::ROUNDS_F + Self::ROUNDS_P;
        let rounds_f_beginning = Self::ROUNDS_F / 2;
        for r in 0..rounds_f_beginning {
            Self::add_rc(state, &RC12[r]);
            Self::sbox(state);
            external_linear_layer.permute_mut(state);
        }

        // The internal rounds.
        let p_end = rounds_f_beginning + Self::ROUNDS_P;
        for r in rounds_f_beginning..p_end {
            state[0] += Self::from_canonical_u64(RC12[r][0]);
            state[0] = Self::sbox_p(&state[0]);
            matmul_internal(state, MATRIX_DIAG_12_GOLDILOCKS);
        }

        // The second half of the external rounds.
        for r in p_end..rounds {
            Self::add_rc(state, &RC12[r]);
            Self::sbox(state);
            external_linear_layer.permute_mut(state);
        }
    }

    #[inline]
    #[cfg(target_feature = "avx2")]
    fn poseidon2(state: &mut [Self; SPONGE_WIDTH]) {
        permute_mut_avx(state);

        // The first half of the external rounds.
        let rounds = Self::ROUNDS_F + Self::ROUNDS_P;
        let rounds_f_beginning = Self::ROUNDS_F / 2;
        for r in 0..rounds_f_beginning {
            add_rc_avx(state, &RC12S[r]);
            sbox_avx(state);
            permute_mut_avx(state);
        }

        // The internal rounds.
        let p_end = rounds_f_beginning + Self::ROUNDS_P;
        internal_layer_avx(state, MATRIX_DIAG_12_GOLDILOCKS, rounds_f_beginning, p_end);
        /*
        for r in rounds_f_beginning..p_end {
            state[0] += Self::from_canonical_u64(RC12[r][0]);
            state[0] = Self::sbox_p(&state[0]);
            matmul_internal_avx(state, MATRIX_DIAG_12_GOLDILOCKS);
        }
        */

        // The second half of the external rounds.
        for r in p_end..rounds {
            add_rc_avx(state, &RC12S[r]);
            sbox_avx(state);
            permute_mut_avx(state);
        }
    }
}

#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Poseidon2Permutation<T> {
    state: [T; SPONGE_WIDTH],
}

impl<T: Eq> Eq for Poseidon2Permutation<T> {}

impl<T> AsRef<[T]> for Poseidon2Permutation<T> {
    fn as_ref(&self) -> &[T] {
        &self.state
    }
}

trait Permuter2: Sized {
    fn permute(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH];
}

impl<F: Poseidon2> Permuter2 for F {
    fn permute(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        let mut inout = input.clone();
        <F as Poseidon2>::poseidon2(&mut inout);
        inout
    }
}

impl Permuter2 for Target {
    fn permute(_input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        panic!("Call `permute_swapped()` instead of `permute()`");
    }
}

impl<T: Copy + Debug + Default + Eq + Permuter2 + Send + Sync> PlonkyPermutation<T>
    for Poseidon2Permutation<T>
{
    const RATE: usize = SPONGE_RATE;
    const WIDTH: usize = SPONGE_WIDTH;

    fn new<I: IntoIterator<Item = T>>(elts: I) -> Self {
        let mut perm = Self {
            state: [T::default(); SPONGE_WIDTH],
        };
        perm.set_from_iter(elts, 0);
        perm
    }

    fn set_elt(&mut self, elt: T, idx: usize) {
        self.state[idx] = elt;
    }

    fn set_from_slice(&mut self, elts: &[T], start_idx: usize) {
        let begin = start_idx;
        let end = start_idx + elts.len();
        self.state[begin..end].copy_from_slice(elts);
    }

    fn set_from_iter<I: IntoIterator<Item = T>>(&mut self, elts: I, start_idx: usize) {
        for (s, e) in self.state[start_idx..].iter_mut().zip(elts) {
            *s = e;
        }
    }

    fn permute(&mut self) {
        self.state = T::permute(self.state);
    }

    fn squeeze(&self) -> &[T] {
        &self.state[..Self::RATE]
    }
}

impl Poseidon2 for GoldilocksField {}

/// Poseidon hash function.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Poseidon2Hash;
impl<F: RichField + Poseidon2> Hasher<F> for Poseidon2Hash {
    const HASHER_TYPE: HasherType = HasherType::Poseidon2;
    const HASH_SIZE: usize = 4 * 8;
    type Hash = HashOut<F>;
    type Permutation = Poseidon2Permutation<F>;

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        let mut perm = Self::Permutation::new(repeat(F::ZERO));

        // Absorb all input chunks.
        for input_chunk in input.chunks(Self::Permutation::RATE) {
            perm.set_from_slice(input_chunk, 0);
            perm.permute();
        }

        HashOut {
            elements: perm.squeeze()[..NUM_HASH_OUT_ELTS].try_into().unwrap(),
        }
    }

    fn two_to_one(x: Self::Hash, y: Self::Hash) -> Self::Hash {
        debug_assert_eq!(x.elements.len(), NUM_HASH_OUT_ELTS);
        debug_assert_eq!(y.elements.len(), NUM_HASH_OUT_ELTS);
        debug_assert!(Self::Permutation::RATE >= NUM_HASH_OUT_ELTS);

        let mut perm = Self::Permutation::new(repeat(F::ZERO));
        perm.set_from_slice(&x.elements, 0);
        perm.set_from_slice(&y.elements, NUM_HASH_OUT_ELTS);

        perm.permute();

        HashOut {
            elements: perm.squeeze()[..NUM_HASH_OUT_ELTS].try_into().unwrap(),
        }
    }

    fn hash_public_inputs(input: &[F]) -> Self::Hash {
        Poseidon2Hash::hash_no_pad(input)
    }
}

impl<F: RichField + Poseidon2> AlgebraicHasher<F> for Poseidon2Hash {
    type AlgebraicPermutation = Poseidon2Permutation<Target>;

    fn permute_swapped<const D: usize>(
        inputs: Self::AlgebraicPermutation,
        swap: BoolTarget,
        builder: &mut CircuitBuilder<F, D>,
    ) -> Self::AlgebraicPermutation
    where
        F: RichField + Extendable<D>,
    {
        todo!()
    }

    fn public_inputs_hash<const D: usize>(
        inputs: Vec<Target>,
        builder: &mut CircuitBuilder<F, D>,
    ) -> HashOutTarget
    where
        F: RichField + Extendable<D> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use plonky2_field::goldilocks_field::GoldilocksField;
    use plonky2_field::types::Field;
    use rand::Rng;
    use zkhash::ark_ff::{BigInteger, PrimeField};
    use zkhash::fields::goldilocks::FpGoldiLocks;
    use zkhash::poseidon2::poseidon2::Poseidon2 as Poseidon2Ref;
    use zkhash::poseidon2::poseidon2_instance_goldilocks::POSEIDON2_GOLDILOCKS_12_PARAMS;

    use crate::hash::poseidon2::Poseidon2;

    fn goldilocks_from_ark_ff(input: FpGoldiLocks) -> GoldilocksField {
        let as_bigint = input.into_bigint();
        let mut as_bytes = as_bigint.to_bytes_le();
        as_bytes.resize(8, 0);
        let as_u64 = u64::from_le_bytes(as_bytes[0..8].try_into().unwrap());
        GoldilocksField::from_canonical_u64(as_u64)
    }

    #[test]
    fn test_poseidon2_goldilocks_width_12() {
        const WIDTH: usize = 12;

        let mut rng = rand::thread_rng();

        // Poiseidon2 reference implementation from zkhash repo.
        let poseidon2_ref = Poseidon2Ref::new(&POSEIDON2_GOLDILOCKS_12_PARAMS);

        // Generate random input and convert to both Goldilocks field formats.
        let input_u64 = rng.gen::<[u64; WIDTH]>();
        let input_ref = input_u64
            .iter()
            .cloned()
            .map(FpGoldiLocks::from)
            .collect::<Vec<_>>();
        let input = input_u64.map(GoldilocksField::from_canonical_u64);

        // Check that the conversion is correct.
        assert!(input_ref
            .iter()
            .zip(input.iter())
            .all(|(a, b)| goldilocks_from_ark_ff(*a) == *b));

        // Run reference implementation.
        let output_ref = poseidon2_ref.permutation(&input_ref);
        let expected: [GoldilocksField; WIDTH] = output_ref
            .iter()
            .cloned()
            .map(goldilocks_from_ark_ff)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Run our implementation.
        let mut output: [GoldilocksField; 12] = input;
        Poseidon2::poseidon2(&mut output);

        assert_eq!(output, expected);
    }
}
