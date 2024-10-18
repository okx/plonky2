use core::arch::x86_64::*;

use unroll::unroll_for_loops;

use super::poseidon_goldilocks_avx2::{
    MDS_FREQ_BLOCK_ONE, MDS_FREQ_BLOCK_THREE, MDS_FREQ_BLOCK_TWO,
};
use crate::field::types::PrimeField64;
use crate::hash::arch::x86_64::goldilocks_avx512::*;
use crate::hash::arch::x86_64::poseidon_goldilocks_avx2::FAST_PARTIAL_ROUND_W_HATS;
use crate::hash::hash_types::RichField;
use crate::hash::poseidon::{
    add_u160_u128, reduce_u160, Poseidon, ALL_ROUND_CONSTANTS, HALF_N_FULL_ROUNDS,
    N_PARTIAL_ROUNDS, N_ROUNDS, SPONGE_RATE, SPONGE_WIDTH,
};
use crate::hash::poseidon_goldilocks::poseidon12_mds::block2;

#[allow(dead_code)]
const MDS_MATRIX_CIRC: [u64; 12] = [17, 15, 41, 16, 2, 28, 13, 13, 39, 18, 34, 20];

#[allow(dead_code)]
const MDS_MATRIX_DIAG: [u64; 12] = [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

const FAST_PARTIAL_FIRST_ROUND_CONSTANT: [u64; 12] = [
    0x3cc3f892184df408,
    0xe993fd841e7e97f1,
    0xf2831d3575f0f3af,
    0xd2500e0a350994ca,
    0xc5571f35d7288633,
    0x91d89c5184109a02,
    0xf37f925d04e5667b,
    0x2d6e448371955a69,
    0x740ef19ce01398a1,
    0x694d24c0752fdf45,
    0x60936af96ee2f148,
    0xc33448feadc78f0c,
];

const FAST_PARTIAL_FIRST_ROUND_CONSTANT_AVX512: [u64; 24] = [
    0x3cc3f892184df408,
    0xe993fd841e7e97f1,
    0xf2831d3575f0f3af,
    0xd2500e0a350994ca,
    0x3cc3f892184df408,
    0xe993fd841e7e97f1,
    0xf2831d3575f0f3af,
    0xd2500e0a350994ca,
    0xc5571f35d7288633,
    0x91d89c5184109a02,
    0xf37f925d04e5667b,
    0x2d6e448371955a69,
    0xc5571f35d7288633,
    0x91d89c5184109a02,
    0xf37f925d04e5667b,
    0x2d6e448371955a69,
    0x740ef19ce01398a1,
    0x694d24c0752fdf45,
    0x60936af96ee2f148,
    0xc33448feadc78f0c,
    0x740ef19ce01398a1,
    0x694d24c0752fdf45,
    0x60936af96ee2f148,
    0xc33448feadc78f0c,
];

const FAST_PARTIAL_ROUND_CONSTANTS: [u64; N_PARTIAL_ROUNDS] = [
    0x74cb2e819ae421ab,
    0xd2559d2370e7f663,
    0x62bf78acf843d17c,
    0xd5ab7b67e14d1fb4,
    0xb9fe2ae6e0969bdc,
    0xe33fdf79f92a10e8,
    0x0ea2bb4c2b25989b,
    0xca9121fbf9d38f06,
    0xbdd9b0aa81f58fa4,
    0x83079fa4ecf20d7e,
    0x650b838edfcc4ad3,
    0x77180c88583c76ac,
    0xaf8c20753143a180,
    0xb8ccfe9989a39175,
    0x954a1729f60cc9c5,
    0xdeb5b550c4dca53b,
    0xf01bb0b00f77011e,
    0xa1ebb404b676afd9,
    0x860b6e1597a0173e,
    0x308bb65a036acbce,
    0x1aca78f31c97c876,
    0x0,
];

#[allow(unused)]
const FAST_PARTIAL_ROUND_INITIAL_MATRIX: [[u64; 12]; 12] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [
        0,
        0x80772dc2645b280b,
        0xdc927721da922cf8,
        0xc1978156516879ad,
        0x90e80c591f48b603,
        0x3a2432625475e3ae,
        0x00a2d4321cca94fe,
        0x77736f524010c932,
        0x904d3f2804a36c54,
        0xbf9b39e28a16f354,
        0x3a1ded54a6cd058b,
        0x42392870da5737cf,
    ],
    [
        0,
        0xe796d293a47a64cb,
        0xb124c33152a2421a,
        0x0ee5dc0ce131268a,
        0xa9032a52f930fae6,
        0x7e33ca8c814280de,
        0xad11180f69a8c29e,
        0xc75ac6d5b5a10ff3,
        0xf0674a8dc5a387ec,
        0xb36d43120eaa5e2b,
        0x6f232aab4b533a25,
        0x3a1ded54a6cd058b,
    ],
    [
        0,
        0xdcedab70f40718ba,
        0x14a4a64da0b2668f,
        0x4715b8e5ab34653b,
        0x1e8916a99c93a88e,
        0xbba4b5d86b9a3b2c,
        0xe76649f9bd5d5c2e,
        0xaf8e2518a1ece54d,
        0xdcda1344cdca873f,
        0xcd080204256088e5,
        0xb36d43120eaa5e2b,
        0xbf9b39e28a16f354,
    ],
    [
        0,
        0xf4a437f2888ae909,
        0xc537d44dc2875403,
        0x7f68007619fd8ba9,
        0xa4911db6a32612da,
        0x2f7e9aade3fdaec1,
        0xe7ffd578da4ea43d,
        0x43a608e7afa6b5c2,
        0xca46546aa99e1575,
        0xdcda1344cdca873f,
        0xf0674a8dc5a387ec,
        0x904d3f2804a36c54,
    ],
    [
        0,
        0xf97abba0dffb6c50,
        0x5e40f0c9bb82aab5,
        0x5996a80497e24a6b,
        0x07084430a7307c9a,
        0xad2f570a5b8545aa,
        0xab7f81fef4274770,
        0xcb81f535cf98c9e9,
        0x43a608e7afa6b5c2,
        0xaf8e2518a1ece54d,
        0xc75ac6d5b5a10ff3,
        0x77736f524010c932,
    ],
    [
        0,
        0x7f8e41e0b0a6cdff,
        0x4b1ba8d40afca97d,
        0x623708f28fca70e8,
        0xbf150dc4914d380f,
        0xc26a083554767106,
        0x753b8b1126665c22,
        0xab7f81fef4274770,
        0xe7ffd578da4ea43d,
        0xe76649f9bd5d5c2e,
        0xad11180f69a8c29e,
        0x00a2d4321cca94fe,
    ],
    [
        0,
        0x726af914971c1374,
        0x1d7f8a2cce1a9d00,
        0x18737784700c75cd,
        0x7fb45d605dd82838,
        0x862361aeab0f9b6e,
        0xc26a083554767106,
        0xad2f570a5b8545aa,
        0x2f7e9aade3fdaec1,
        0xbba4b5d86b9a3b2c,
        0x7e33ca8c814280de,
        0x3a2432625475e3ae,
    ],
    [
        0,
        0x64dd936da878404d,
        0x4db9a2ead2bd7262,
        0xbe2e19f6d07f1a83,
        0x02290fe23c20351a,
        0x7fb45d605dd82838,
        0xbf150dc4914d380f,
        0x07084430a7307c9a,
        0xa4911db6a32612da,
        0x1e8916a99c93a88e,
        0xa9032a52f930fae6,
        0x90e80c591f48b603,
    ],
    [
        0,
        0x85418a9fef8a9890,
        0xd8a2eb7ef5e707ad,
        0xbfe85ababed2d882,
        0xbe2e19f6d07f1a83,
        0x18737784700c75cd,
        0x623708f28fca70e8,
        0x5996a80497e24a6b,
        0x7f68007619fd8ba9,
        0x4715b8e5ab34653b,
        0x0ee5dc0ce131268a,
        0xc1978156516879ad,
    ],
    [
        0,
        0x156048ee7a738154,
        0x91f7562377e81df5,
        0xd8a2eb7ef5e707ad,
        0x4db9a2ead2bd7262,
        0x1d7f8a2cce1a9d00,
        0x4b1ba8d40afca97d,
        0x5e40f0c9bb82aab5,
        0xc537d44dc2875403,
        0x14a4a64da0b2668f,
        0xb124c33152a2421a,
        0xdc927721da922cf8,
    ],
    [
        0,
        0xd841e8ef9dde8ba0,
        0x156048ee7a738154,
        0x85418a9fef8a9890,
        0x64dd936da878404d,
        0x726af914971c1374,
        0x7f8e41e0b0a6cdff,
        0xf97abba0dffb6c50,
        0xf4a437f2888ae909,
        0xdcedab70f40718ba,
        0xe796d293a47a64cb,
        0x80772dc2645b280b,
    ],
];

const FAST_PARTIAL_ROUND_INITIAL_MATRIX_AVX512: [[u64; 24]; 12] = [
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        0,
        0x80772dc2645b280b,
        0xdc927721da922cf8,
        0xc1978156516879ad,
        0,
        0x80772dc2645b280b,
        0xdc927721da922cf8,
        0xc1978156516879ad,
        0x90e80c591f48b603,
        0x3a2432625475e3ae,
        0x00a2d4321cca94fe,
        0x77736f524010c932,
        0x90e80c591f48b603,
        0x3a2432625475e3ae,
        0x00a2d4321cca94fe,
        0x77736f524010c932,
        0x904d3f2804a36c54,
        0xbf9b39e28a16f354,
        0x3a1ded54a6cd058b,
        0x42392870da5737cf,
        0x904d3f2804a36c54,
        0xbf9b39e28a16f354,
        0x3a1ded54a6cd058b,
        0x42392870da5737cf,
    ],
    [
        0,
        0xe796d293a47a64cb,
        0xb124c33152a2421a,
        0x0ee5dc0ce131268a,
        0,
        0xe796d293a47a64cb,
        0xb124c33152a2421a,
        0x0ee5dc0ce131268a,
        0xa9032a52f930fae6,
        0x7e33ca8c814280de,
        0xad11180f69a8c29e,
        0xc75ac6d5b5a10ff3,
        0xa9032a52f930fae6,
        0x7e33ca8c814280de,
        0xad11180f69a8c29e,
        0xc75ac6d5b5a10ff3,
        0xf0674a8dc5a387ec,
        0xb36d43120eaa5e2b,
        0x6f232aab4b533a25,
        0x3a1ded54a6cd058b,
        0xf0674a8dc5a387ec,
        0xb36d43120eaa5e2b,
        0x6f232aab4b533a25,
        0x3a1ded54a6cd058b,
    ],
    [
        0,
        0xdcedab70f40718ba,
        0x14a4a64da0b2668f,
        0x4715b8e5ab34653b,
        0,
        0xdcedab70f40718ba,
        0x14a4a64da0b2668f,
        0x4715b8e5ab34653b,
        0x1e8916a99c93a88e,
        0xbba4b5d86b9a3b2c,
        0xe76649f9bd5d5c2e,
        0xaf8e2518a1ece54d,
        0x1e8916a99c93a88e,
        0xbba4b5d86b9a3b2c,
        0xe76649f9bd5d5c2e,
        0xaf8e2518a1ece54d,
        0xdcda1344cdca873f,
        0xcd080204256088e5,
        0xb36d43120eaa5e2b,
        0xbf9b39e28a16f354,
        0xdcda1344cdca873f,
        0xcd080204256088e5,
        0xb36d43120eaa5e2b,
        0xbf9b39e28a16f354,
    ],
    [
        0,
        0xf4a437f2888ae909,
        0xc537d44dc2875403,
        0x7f68007619fd8ba9,
        0,
        0xf4a437f2888ae909,
        0xc537d44dc2875403,
        0x7f68007619fd8ba9,
        0xa4911db6a32612da,
        0x2f7e9aade3fdaec1,
        0xe7ffd578da4ea43d,
        0x43a608e7afa6b5c2,
        0xa4911db6a32612da,
        0x2f7e9aade3fdaec1,
        0xe7ffd578da4ea43d,
        0x43a608e7afa6b5c2,
        0xca46546aa99e1575,
        0xdcda1344cdca873f,
        0xf0674a8dc5a387ec,
        0x904d3f2804a36c54,
        0xca46546aa99e1575,
        0xdcda1344cdca873f,
        0xf0674a8dc5a387ec,
        0x904d3f2804a36c54,
    ],
    [
        0,
        0xf97abba0dffb6c50,
        0x5e40f0c9bb82aab5,
        0x5996a80497e24a6b,
        0,
        0xf97abba0dffb6c50,
        0x5e40f0c9bb82aab5,
        0x5996a80497e24a6b,
        0x07084430a7307c9a,
        0xad2f570a5b8545aa,
        0xab7f81fef4274770,
        0xcb81f535cf98c9e9,
        0x07084430a7307c9a,
        0xad2f570a5b8545aa,
        0xab7f81fef4274770,
        0xcb81f535cf98c9e9,
        0x43a608e7afa6b5c2,
        0xaf8e2518a1ece54d,
        0xc75ac6d5b5a10ff3,
        0x77736f524010c932,
        0x43a608e7afa6b5c2,
        0xaf8e2518a1ece54d,
        0xc75ac6d5b5a10ff3,
        0x77736f524010c932,
    ],
    [
        0,
        0x7f8e41e0b0a6cdff,
        0x4b1ba8d40afca97d,
        0x623708f28fca70e8,
        0,
        0x7f8e41e0b0a6cdff,
        0x4b1ba8d40afca97d,
        0x623708f28fca70e8,
        0xbf150dc4914d380f,
        0xc26a083554767106,
        0x753b8b1126665c22,
        0xab7f81fef4274770,
        0xbf150dc4914d380f,
        0xc26a083554767106,
        0x753b8b1126665c22,
        0xab7f81fef4274770,
        0xe7ffd578da4ea43d,
        0xe76649f9bd5d5c2e,
        0xad11180f69a8c29e,
        0x00a2d4321cca94fe,
        0xe7ffd578da4ea43d,
        0xe76649f9bd5d5c2e,
        0xad11180f69a8c29e,
        0x00a2d4321cca94fe,
    ],
    [
        0,
        0x726af914971c1374,
        0x1d7f8a2cce1a9d00,
        0x18737784700c75cd,
        0,
        0x726af914971c1374,
        0x1d7f8a2cce1a9d00,
        0x18737784700c75cd,
        0x7fb45d605dd82838,
        0x862361aeab0f9b6e,
        0xc26a083554767106,
        0xad2f570a5b8545aa,
        0x7fb45d605dd82838,
        0x862361aeab0f9b6e,
        0xc26a083554767106,
        0xad2f570a5b8545aa,
        0x2f7e9aade3fdaec1,
        0xbba4b5d86b9a3b2c,
        0x7e33ca8c814280de,
        0x3a2432625475e3ae,
        0x2f7e9aade3fdaec1,
        0xbba4b5d86b9a3b2c,
        0x7e33ca8c814280de,
        0x3a2432625475e3ae,
    ],
    [
        0,
        0x64dd936da878404d,
        0x4db9a2ead2bd7262,
        0xbe2e19f6d07f1a83,
        0,
        0x64dd936da878404d,
        0x4db9a2ead2bd7262,
        0xbe2e19f6d07f1a83,
        0x02290fe23c20351a,
        0x7fb45d605dd82838,
        0xbf150dc4914d380f,
        0x07084430a7307c9a,
        0x02290fe23c20351a,
        0x7fb45d605dd82838,
        0xbf150dc4914d380f,
        0x07084430a7307c9a,
        0xa4911db6a32612da,
        0x1e8916a99c93a88e,
        0xa9032a52f930fae6,
        0x90e80c591f48b603,
        0xa4911db6a32612da,
        0x1e8916a99c93a88e,
        0xa9032a52f930fae6,
        0x90e80c591f48b603,
    ],
    [
        0,
        0x85418a9fef8a9890,
        0xd8a2eb7ef5e707ad,
        0xbfe85ababed2d882,
        0,
        0x85418a9fef8a9890,
        0xd8a2eb7ef5e707ad,
        0xbfe85ababed2d882,
        0xbe2e19f6d07f1a83,
        0x18737784700c75cd,
        0x623708f28fca70e8,
        0x5996a80497e24a6b,
        0xbe2e19f6d07f1a83,
        0x18737784700c75cd,
        0x623708f28fca70e8,
        0x5996a80497e24a6b,
        0x7f68007619fd8ba9,
        0x4715b8e5ab34653b,
        0x0ee5dc0ce131268a,
        0xc1978156516879ad,
        0x7f68007619fd8ba9,
        0x4715b8e5ab34653b,
        0x0ee5dc0ce131268a,
        0xc1978156516879ad,
    ],
    [
        0,
        0x156048ee7a738154,
        0x91f7562377e81df5,
        0xd8a2eb7ef5e707ad,
        0,
        0x156048ee7a738154,
        0x91f7562377e81df5,
        0xd8a2eb7ef5e707ad,
        0x4db9a2ead2bd7262,
        0x1d7f8a2cce1a9d00,
        0x4b1ba8d40afca97d,
        0x5e40f0c9bb82aab5,
        0x4db9a2ead2bd7262,
        0x1d7f8a2cce1a9d00,
        0x4b1ba8d40afca97d,
        0x5e40f0c9bb82aab5,
        0xc537d44dc2875403,
        0x14a4a64da0b2668f,
        0xb124c33152a2421a,
        0xdc927721da922cf8,
        0xc537d44dc2875403,
        0x14a4a64da0b2668f,
        0xb124c33152a2421a,
        0xdc927721da922cf8,
    ],
    [
        0,
        0xd841e8ef9dde8ba0,
        0x156048ee7a738154,
        0x85418a9fef8a9890,
        0,
        0xd841e8ef9dde8ba0,
        0x156048ee7a738154,
        0x85418a9fef8a9890,
        0x64dd936da878404d,
        0x726af914971c1374,
        0x7f8e41e0b0a6cdff,
        0xf97abba0dffb6c50,
        0x64dd936da878404d,
        0x726af914971c1374,
        0x7f8e41e0b0a6cdff,
        0xf97abba0dffb6c50,
        0xf4a437f2888ae909,
        0xdcedab70f40718ba,
        0xe796d293a47a64cb,
        0x80772dc2645b280b,
        0xf4a437f2888ae909,
        0xdcedab70f40718ba,
        0xe796d293a47a64cb,
        0x80772dc2645b280b,
    ],
];

#[rustfmt::skip]
pub const ALL_ROUND_CONSTANTS_AVX512: [u64; 2 * SPONGE_WIDTH * N_ROUNDS]  = [
    0xb585f766f2144405, 0x7746a55f43921ad7, 0xb2fb0d31cee799b4, 0xf6760a4803427d7, 0xb585f766f2144405, 0x7746a55f43921ad7, 0xb2fb0d31cee799b4, 0xf6760a4803427d7,
    0xe10d666650f4e012, 0x8cae14cb07d09bf1, 0xd438539c95f63e9f, 0xef781c7ce35b4c3d, 0xe10d666650f4e012, 0x8cae14cb07d09bf1, 0xd438539c95f63e9f, 0xef781c7ce35b4c3d,
    0xcdc4a239b0c44426, 0x277fa208bf337bff, 0xe17653a29da578a1, 0xc54302f225db2c76, 0xcdc4a239b0c44426, 0x277fa208bf337bff, 0xe17653a29da578a1, 0xc54302f225db2c76,
    0x86287821f722c881, 0x59cd1a8a41c18e55, 0xc3b919ad495dc574, 0xa484c4c5ef6a0781, 0x86287821f722c881, 0x59cd1a8a41c18e55, 0xc3b919ad495dc574, 0xa484c4c5ef6a0781,
    0x308bbd23dc5416cc, 0x6e4a40c18f30c09c, 0x9a2eedb70d8f8cfa, 0xe360c6e0ae486f38, 0x308bbd23dc5416cc, 0x6e4a40c18f30c09c, 0x9a2eedb70d8f8cfa, 0xe360c6e0ae486f38,
    0xd5c7718fbfc647fb, 0xc35eae071903ff0b, 0x849c2656969c4be7, 0xc0572c8c08cbbbad, 0xd5c7718fbfc647fb, 0xc35eae071903ff0b, 0x849c2656969c4be7, 0xc0572c8c08cbbbad,
    0xe9fa634a21de0082, 0xf56f6d48959a600d, 0xf7d713e806391165, 0x8297132b32825daf, 0xe9fa634a21de0082, 0xf56f6d48959a600d, 0xf7d713e806391165, 0x8297132b32825daf,
    0xad6805e0e30b2c8a, 0xac51d9f5fcf8535e, 0x502ad7dc18c2ad87, 0x57a1550c110b3041, 0xad6805e0e30b2c8a, 0xac51d9f5fcf8535e, 0x502ad7dc18c2ad87, 0x57a1550c110b3041,
    0x66bbd30e6ce0e583, 0xda2abef589d644e, 0xf061274fdb150d61, 0x28b8ec3ae9c29633, 0x66bbd30e6ce0e583, 0xda2abef589d644e, 0xf061274fdb150d61, 0x28b8ec3ae9c29633,
    0x92a756e67e2b9413, 0x70e741ebfee96586, 0x19d5ee2af82ec1c, 0x6f6f2ed772466352, 0x92a756e67e2b9413, 0x70e741ebfee96586, 0x19d5ee2af82ec1c, 0x6f6f2ed772466352,
    0x7cf416cfe7e14ca1, 0x61df517b86a46439, 0x85dc499b11d77b75, 0x4b959b48b9c10733, 0x7cf416cfe7e14ca1, 0x61df517b86a46439, 0x85dc499b11d77b75, 0x4b959b48b9c10733,
    0xe8be3e5da8043e57, 0xf5c0bc1de6da8699, 0x40b12cbf09ef74bf, 0xa637093ecb2ad631, 0xe8be3e5da8043e57, 0xf5c0bc1de6da8699, 0x40b12cbf09ef74bf, 0xa637093ecb2ad631,
    0x3cc3f892184df408, 0x2e479dc157bf31bb, 0x6f49de07a6234346, 0x213ce7bede378d7b, 0x3cc3f892184df408, 0x2e479dc157bf31bb, 0x6f49de07a6234346, 0x213ce7bede378d7b,
    0x5b0431345d4dea83, 0xa2de45780344d6a1, 0x7103aaf94a7bf308, 0x5326fc0d97279301, 0x5b0431345d4dea83, 0xa2de45780344d6a1, 0x7103aaf94a7bf308, 0x5326fc0d97279301,
    0xa9ceb74fec024747, 0x27f8ec88bb21b1a3, 0xfceb4fda1ded0893, 0xfac6ff1346a41675, 0xa9ceb74fec024747, 0x27f8ec88bb21b1a3, 0xfceb4fda1ded0893, 0xfac6ff1346a41675,
    0x7131aa45268d7d8c, 0x9351036095630f9f, 0xad535b24afc26bfb, 0x4627f5c6993e44be, 0x7131aa45268d7d8c, 0x9351036095630f9f, 0xad535b24afc26bfb, 0x4627f5c6993e44be,
    0x645cf794b8f1cc58, 0x241c70ed0af61617, 0xacb8e076647905f1, 0x3737e9db4c4f474d, 0x645cf794b8f1cc58, 0x241c70ed0af61617, 0xacb8e076647905f1, 0x3737e9db4c4f474d,
    0xe7ea5e33e75fffb6, 0x90dee49fc9bfc23a, 0xd1b1edf76bc09c92, 0xb65481ba645c602, 0xe7ea5e33e75fffb6, 0x90dee49fc9bfc23a, 0xd1b1edf76bc09c92, 0xb65481ba645c602,
    0x99ad1aab0814283b, 0x438a7c91d416ca4d, 0xb60de3bcc5ea751c, 0xc99cab6aef6f58bc, 0x99ad1aab0814283b, 0x438a7c91d416ca4d, 0xb60de3bcc5ea751c, 0xc99cab6aef6f58bc,
    0x69a5ed92a72ee4ff, 0x5e7b329c1ed4ad71, 0x5fc0ac0800144885, 0x32db829239774eca, 0x69a5ed92a72ee4ff, 0x5e7b329c1ed4ad71, 0x5fc0ac0800144885, 0x32db829239774eca,
    0xade699c5830f310, 0x7cc5583b10415f21, 0x85df9ed2e166d64f, 0x6604df4fee32bcb1, 0xade699c5830f310, 0x7cc5583b10415f21, 0x85df9ed2e166d64f, 0x6604df4fee32bcb1,
    0xeb84f608da56ef48, 0xda608834c40e603d, 0x8f97fe408061f183, 0xa93f485c96f37b89, 0xeb84f608da56ef48, 0xda608834c40e603d, 0x8f97fe408061f183, 0xa93f485c96f37b89,
    0x6704e8ee8f18d563, 0xcee3e9ac1e072119, 0x510d0e65e2b470c1, 0xf6323f486b9038f0, 0x6704e8ee8f18d563, 0xcee3e9ac1e072119, 0x510d0e65e2b470c1, 0xf6323f486b9038f0,
    0xb508cdeffa5ceef, 0xf2417089e4fb3cbd, 0x60e75c2890d15730, 0xa6217d8bf660f29c, 0xb508cdeffa5ceef, 0xf2417089e4fb3cbd, 0x60e75c2890d15730, 0xa6217d8bf660f29c,
    0x7159cd30c3ac118e, 0x839b4e8fafead540, 0xd3f3e5e82920adc, 0x8f7d83bddee7bba8, 0x7159cd30c3ac118e, 0x839b4e8fafead540, 0xd3f3e5e82920adc, 0x8f7d83bddee7bba8,
    0x780f2243ea071d06, 0xeb915845f3de1634, 0xd19e120d26b6f386, 0x16ee53a7e5fecc6, 0x780f2243ea071d06, 0xeb915845f3de1634, 0xd19e120d26b6f386, 0x16ee53a7e5fecc6,
    0xcb5fd54e7933e477, 0xacb8417879fd449f, 0x9c22190be7f74732, 0x5d693c1ba3ba3621, 0xcb5fd54e7933e477, 0xacb8417879fd449f, 0x9c22190be7f74732, 0x5d693c1ba3ba3621,
    0xdcef0797c2b69ec7, 0x3d639263da827b13, 0xe273fd971bc8d0e7, 0x418f02702d227ed5, 0xdcef0797c2b69ec7, 0x3d639263da827b13, 0xe273fd971bc8d0e7, 0x418f02702d227ed5,
    0x8c25fda3b503038c, 0x2cbaed4daec8c07c, 0x5f58e6afcdd6ddc2, 0x284650ac5e1b0eba, 0x8c25fda3b503038c, 0x2cbaed4daec8c07c, 0x5f58e6afcdd6ddc2, 0x284650ac5e1b0eba,
    0x635b337ee819dab5, 0x9f9a036ed4f2d49f, 0xb93e260cae5c170e, 0xb0a7eae879ddb76d, 0x635b337ee819dab5, 0x9f9a036ed4f2d49f, 0xb93e260cae5c170e, 0xb0a7eae879ddb76d,
    0xd0762cbc8ca6570c, 0x34c6efb812b04bf5, 0x40bf0ab5fa14c112, 0xb6b570fc7c5740d3, 0xd0762cbc8ca6570c, 0x34c6efb812b04bf5, 0x40bf0ab5fa14c112, 0xb6b570fc7c5740d3,
    0x5a27b9002de33454, 0xb1a5b165b6d2b2d2, 0x8722e0ace9d1be22, 0x788ee3b37e5680fb, 0x5a27b9002de33454, 0xb1a5b165b6d2b2d2, 0x8722e0ace9d1be22, 0x788ee3b37e5680fb,
    0x14a726661551e284, 0x98b7672f9ef3b419, 0xbb93ae776bb30e3a, 0x28fd3b046380f850, 0x14a726661551e284, 0x98b7672f9ef3b419, 0xbb93ae776bb30e3a, 0x28fd3b046380f850,
    0x30a4680593258387, 0x337dc00c61bd9ce1, 0xd5eca244c7a4ff1d, 0x7762638264d279bd, 0x30a4680593258387, 0x337dc00c61bd9ce1, 0xd5eca244c7a4ff1d, 0x7762638264d279bd,
    0xc1e434bedeefd767, 0x299351a53b8ec22, 0xb2d456e4ad251b80, 0x3e9ed1fda49cea0b, 0xc1e434bedeefd767, 0x299351a53b8ec22, 0xb2d456e4ad251b80, 0x3e9ed1fda49cea0b,
    0x2972a92ba450bed8, 0x20216dd77be493de, 0xadffe8cf28449ec6, 0x1c4dbb1c4c27d243, 0x2972a92ba450bed8, 0x20216dd77be493de, 0xadffe8cf28449ec6, 0x1c4dbb1c4c27d243,
    0x15a16a8a8322d458, 0x388a128b7fd9a609, 0x2300e5d6baedf0fb, 0x2f63aa8647e15104, 0x15a16a8a8322d458, 0x388a128b7fd9a609, 0x2300e5d6baedf0fb, 0x2f63aa8647e15104,
    0xf1c36ce86ecec269, 0x27181125183970c9, 0xe584029370dca96d, 0x4d9bbc3e02f1cfb2, 0xf1c36ce86ecec269, 0x27181125183970c9, 0xe584029370dca96d, 0x4d9bbc3e02f1cfb2,
    0xea35bc29692af6f8, 0x18e21b4beabb4137, 0x1e3b9fc625b554f4, 0x25d64362697828fd, 0xea35bc29692af6f8, 0x18e21b4beabb4137, 0x1e3b9fc625b554f4, 0x25d64362697828fd,
    0x5a3f1bb1c53a9645, 0xdb7f023869fb8d38, 0xb462065911d4e1fc, 0x49c24ae4437d8030, 0x5a3f1bb1c53a9645, 0xdb7f023869fb8d38, 0xb462065911d4e1fc, 0x49c24ae4437d8030,
    0xd793862c112b0566, 0xaadd1106730d8feb, 0xc43b6e0e97b0d568, 0xe29024c18ee6fca2, 0xd793862c112b0566, 0xaadd1106730d8feb, 0xc43b6e0e97b0d568, 0xe29024c18ee6fca2,
    0x5e50c27535b88c66, 0x10383f20a4ff9a87, 0x38e8ee9d71a45af8, 0xdd5118375bf1a9b9, 0x5e50c27535b88c66, 0x10383f20a4ff9a87, 0x38e8ee9d71a45af8, 0xdd5118375bf1a9b9,
    0x775005982d74d7f7, 0x86ab99b4dde6c8b0, 0xb1204f603f51c080, 0xef61ac8470250ecf, 0x775005982d74d7f7, 0x86ab99b4dde6c8b0, 0xb1204f603f51c080, 0xef61ac8470250ecf,
    0x1bbcd90f132c603f, 0xcd1dabd964db557, 0x11a3ae5beb9d1ec9, 0xf755bfeea585d11d, 0x1bbcd90f132c603f, 0xcd1dabd964db557, 0x11a3ae5beb9d1ec9, 0xf755bfeea585d11d,
    0xa3b83250268ea4d7, 0x516306f4927c93af, 0xddb4ac49c9efa1da, 0x64bb6dec369d4418, 0xa3b83250268ea4d7, 0x516306f4927c93af, 0xddb4ac49c9efa1da, 0x64bb6dec369d4418,
    0xf9cc95c22b4c1fcc, 0x8d37f755f4ae9f6, 0xeec49b613478675b, 0xf143933aed25e0b0, 0xf9cc95c22b4c1fcc, 0x8d37f755f4ae9f6, 0xeec49b613478675b, 0xf143933aed25e0b0,
    0xe4c5dd8255dfc622, 0xe7ad7756f193198e, 0x92c2318b87fff9cb, 0x739c25f8fd73596d, 0xe4c5dd8255dfc622, 0xe7ad7756f193198e, 0x92c2318b87fff9cb, 0x739c25f8fd73596d,
    0x5636cac9f16dfed0, 0xdd8f909a938e0172, 0xc6401fe115063f5b, 0x8ad97b33f1ac1455, 0x5636cac9f16dfed0, 0xdd8f909a938e0172, 0xc6401fe115063f5b, 0x8ad97b33f1ac1455,
    0xc49366bb25e8513, 0x784d3d2f1698309, 0x530fb67ea1809a81, 0x410492299bb01f49, 0xc49366bb25e8513, 0x784d3d2f1698309, 0x530fb67ea1809a81, 0x410492299bb01f49,
    0x139542347424b9ac, 0x9cb0bd5ea1a1115e, 0x2e3f615c38f49a1, 0x985d4f4a9c5291ef, 0x139542347424b9ac, 0x9cb0bd5ea1a1115e, 0x2e3f615c38f49a1, 0x985d4f4a9c5291ef,
    0x775b9feafdcd26e7, 0x304265a6384f0f2d, 0x593664c39773012c, 0x4f0a2e5fb028f2ce, 0x775b9feafdcd26e7, 0x304265a6384f0f2d, 0x593664c39773012c, 0x4f0a2e5fb028f2ce,
    0xdd611f1000c17442, 0xd8185f9adfea4fd0, 0xef87139ca9a3ab1e, 0x3ba71336c34ee133, 0xdd611f1000c17442, 0xd8185f9adfea4fd0, 0xef87139ca9a3ab1e, 0x3ba71336c34ee133,
    0x7d3a455d56b70238, 0x660d32e130182684, 0x297a863f48cd1f43, 0x90e0a736a751ebb7, 0x7d3a455d56b70238, 0x660d32e130182684, 0x297a863f48cd1f43, 0x90e0a736a751ebb7,
    0x549f80ce550c4fd3, 0xf73b2922f38bd64, 0x16bf1f73fb7a9c3f, 0x6d1f5a59005bec17, 0x549f80ce550c4fd3, 0xf73b2922f38bd64, 0x16bf1f73fb7a9c3f, 0x6d1f5a59005bec17,
    0x2ff876fa5ef97c4, 0xc5cb72a2a51159b0, 0x8470f39d2d5c900e, 0x25abb3f1d39fcb76, 0x2ff876fa5ef97c4, 0xc5cb72a2a51159b0, 0x8470f39d2d5c900e, 0x25abb3f1d39fcb76,
    0x23eb8cc9b372442f, 0xd687ba55c64f6364, 0xda8d9e90fd8ff158, 0xe3cbdc7d2fe45ea7, 0x23eb8cc9b372442f, 0xd687ba55c64f6364, 0xda8d9e90fd8ff158, 0xe3cbdc7d2fe45ea7,
    0xb9a8c9b3aee52297, 0xc0d28a5c10960bd3, 0x45d7ac9b68f71a34, 0xeeb76e397069e804, 0xb9a8c9b3aee52297, 0xc0d28a5c10960bd3, 0x45d7ac9b68f71a34, 0xeeb76e397069e804,
    0x3d06c8bd1514e2d9, 0x9c9c98207cb10767, 0x65700b51aedfb5ef, 0x911f451539869408, 0x3d06c8bd1514e2d9, 0x9c9c98207cb10767, 0x65700b51aedfb5ef, 0x911f451539869408,
    0x7ae6849fbc3a0ec6, 0x3bb340eba06afe7e, 0xb46e9d8b682ea65e, 0x8dcf22f9a3b34356, 0x7ae6849fbc3a0ec6, 0x3bb340eba06afe7e, 0xb46e9d8b682ea65e, 0x8dcf22f9a3b34356,
    0x77bdaeda586257a7, 0xf19e400a5104d20d, 0xc368a348e46d950f, 0x9ef1cd60e679f284, 0x77bdaeda586257a7, 0xf19e400a5104d20d, 0xc368a348e46d950f, 0x9ef1cd60e679f284,
    0xe89cd854d5d01d33, 0x5cd377dc8bb882a2, 0xa7b0fb7883eee860, 0x7684403ec392950d, 0xe89cd854d5d01d33, 0x5cd377dc8bb882a2, 0xa7b0fb7883eee860, 0x7684403ec392950d,
    0x5fa3f06f4fed3b52, 0x8df57ac11bc04831, 0x2db01efa1e1e1897, 0x54846de4aadb9ca2, 0x5fa3f06f4fed3b52, 0x8df57ac11bc04831, 0x2db01efa1e1e1897, 0x54846de4aadb9ca2,
    0xba6745385893c784, 0x541d496344d2c75b, 0xe909678474e687fe, 0xdfe89923f6c9c2ff, 0xba6745385893c784, 0x541d496344d2c75b, 0xe909678474e687fe, 0xdfe89923f6c9c2ff,
    0xece5a71e0cfedc75, 0x5ff98fd5d51fe610, 0x83e8941918964615, 0x5922040b47f150c1, 0xece5a71e0cfedc75, 0x5ff98fd5d51fe610, 0x83e8941918964615, 0x5922040b47f150c1,
    0xf97d750e3dd94521, 0x5080d4c2b86f56d7, 0xa7de115b56c78d70, 0x6a9242ac87538194, 0xf97d750e3dd94521, 0x5080d4c2b86f56d7, 0xa7de115b56c78d70, 0x6a9242ac87538194,
    0xf7856ef7f9173e44, 0x2265fc92feb0dc09, 0x17dfc8e4f7ba8a57, 0x9001a64209f21db8, 0xf7856ef7f9173e44, 0x2265fc92feb0dc09, 0x17dfc8e4f7ba8a57, 0x9001a64209f21db8,
    0x90004c1371b893c5, 0xb932b7cf752e5545, 0xa0b1df81b6fe59fc, 0x8ef1dd26770af2c2, 0x90004c1371b893c5, 0xb932b7cf752e5545, 0xa0b1df81b6fe59fc, 0x8ef1dd26770af2c2,
    0x541a4f9cfbeed35, 0x9e61106178bfc530, 0xb3767e80935d8af2, 0x98d5782065af06, 0x541a4f9cfbeed35, 0x9e61106178bfc530, 0xb3767e80935d8af2, 0x98d5782065af06,
    0x31d191cd5c1466c7, 0x410fefafa319ac9d, 0xbdf8f242e316c4ab, 0x9e8cd55b57637ed0, 0x31d191cd5c1466c7, 0x410fefafa319ac9d, 0xbdf8f242e316c4ab, 0x9e8cd55b57637ed0,
    0xde122bebe9a39368, 0x4d001fd58f002526, 0xca6637000eb4a9f8, 0x2f2339d624f91f78, 0xde122bebe9a39368, 0x4d001fd58f002526, 0xca6637000eb4a9f8, 0x2f2339d624f91f78,
    0x6d1a7918c80df518, 0xdf9a4939342308e9, 0xebc2151ee6c8398c, 0x3cc2ba8a1116515, 0x6d1a7918c80df518, 0xdf9a4939342308e9, 0xebc2151ee6c8398c, 0x3cc2ba8a1116515,
    0xd341d037e840cf83, 0x387cb5d25af4afcc, 0xbba2515f22909e87, 0x7248fe7705f38e47, 0xd341d037e840cf83, 0x387cb5d25af4afcc, 0xbba2515f22909e87, 0x7248fe7705f38e47,
    0x4d61e56a525d225a, 0x262e963c8da05d3d, 0x59e89b094d220ec2, 0x55d5b52b78b9c5e, 0x4d61e56a525d225a, 0x262e963c8da05d3d, 0x59e89b094d220ec2, 0x55d5b52b78b9c5e,
    0x82b27eb33514ef99, 0xd30094ca96b7ce7b, 0xcf5cb381cd0a1535, 0xfeed4db6919e5a7c, 0x82b27eb33514ef99, 0xd30094ca96b7ce7b, 0xcf5cb381cd0a1535, 0xfeed4db6919e5a7c,
    0x41703f53753be59f, 0x5eeea940fcde8b6f, 0x4cd1f1b175100206, 0x4a20358574454ec0, 0x41703f53753be59f, 0x5eeea940fcde8b6f, 0x4cd1f1b175100206, 0x4a20358574454ec0,
    0x1478d361dbbf9fac, 0x6f02dc07d141875c, 0x296a202ed8e556a2, 0x2afd67999bf32ee5, 0x1478d361dbbf9fac, 0x6f02dc07d141875c, 0x296a202ed8e556a2, 0x2afd67999bf32ee5,
    0x7acfd96efa95491d, 0x6798ba0c0abb2c6d, 0x34c6f57b26c92122, 0x5736e1bad206b5de, 0x7acfd96efa95491d, 0x6798ba0c0abb2c6d, 0x34c6f57b26c92122, 0x5736e1bad206b5de,
    0x20057d2a0056521b, 0x3dea5bd5d0578bd7, 0x16e50d897d4634ac, 0x29bff3ecb9b7a6e3, 0x20057d2a0056521b, 0x3dea5bd5d0578bd7, 0x16e50d897d4634ac, 0x29bff3ecb9b7a6e3,
    0x475cd3205a3bdcde, 0x18a42105c31b7e88, 0x23e7414af663068, 0x15147108121967d7, 0x475cd3205a3bdcde, 0x18a42105c31b7e88, 0x23e7414af663068, 0x15147108121967d7,
    0xe4a3dff1d7d6fef9, 0x1a8d1a588085737, 0x11b4c74eda62beef, 0xe587cc0d69a73346, 0xe4a3dff1d7d6fef9, 0x1a8d1a588085737, 0x11b4c74eda62beef, 0xe587cc0d69a73346,
    0x1ff7327017aa2a6e, 0x594e29c42473d06b, 0xf6f31db1899b12d5, 0xc02ac5e47312d3ca, 0x1ff7327017aa2a6e, 0x594e29c42473d06b, 0xf6f31db1899b12d5, 0xc02ac5e47312d3ca,
    0xe70201e960cb78b8, 0x6f90ff3b6a65f108, 0x42747a7245e7fa84, 0xd1f507e43ab749b2, 0xe70201e960cb78b8, 0x6f90ff3b6a65f108, 0x42747a7245e7fa84, 0xd1f507e43ab749b2,
    0x1c86d265f15750cd, 0x3996ce73dd832c1c, 0x8e7fba02983224bd, 0xba0dec7103255dd4, 0x1c86d265f15750cd, 0x3996ce73dd832c1c, 0x8e7fba02983224bd, 0xba0dec7103255dd4,
    0x9e9cbd781628fc5b, 0xdae8645996edd6a5, 0xdebe0853b1a1d378, 0xa49229d24d014343, 0x9e9cbd781628fc5b, 0xdae8645996edd6a5, 0xdebe0853b1a1d378, 0xa49229d24d014343,
    0x7be5b9ffda905e1c, 0xa3c95eaec244aa30, 0x230bca8f4df0544, 0x4135c2bebfe148c6, 0x7be5b9ffda905e1c, 0xa3c95eaec244aa30, 0x230bca8f4df0544, 0x4135c2bebfe148c6,
    0x166fc0cc438a3c72, 0x3762b59a8ae83efa, 0xe8928a4c89114750, 0x2a440b51a4945ee5, 0x166fc0cc438a3c72, 0x3762b59a8ae83efa, 0xe8928a4c89114750, 0x2a440b51a4945ee5,
    0x80cefd2b7d99ff83, 0xbb9879c6e61fd62a, 0x6e7c8f1a84265034, 0x164bb2de1bbeddc8, 0x80cefd2b7d99ff83, 0xbb9879c6e61fd62a, 0x6e7c8f1a84265034, 0x164bb2de1bbeddc8,
    0xf3c12fe54d5c653b, 0x40b9e922ed9771e2, 0x551f5b0fbe7b1840, 0x25032aa7c4cb1811, 0xf3c12fe54d5c653b, 0x40b9e922ed9771e2, 0x551f5b0fbe7b1840, 0x25032aa7c4cb1811,
    0xaaed34074b164346, 0x8ffd96bbf9c9c81d, 0x70fc91eb5937085c, 0x7f795e2a5f915440, 0xaaed34074b164346, 0x8ffd96bbf9c9c81d, 0x70fc91eb5937085c, 0x7f795e2a5f915440,
    0x4543d9df5476d3cb, 0xf172d73e004fc90d, 0xdfd1c4febcc81238, 0xbc8dfb627fe558fc, 0x4543d9df5476d3cb, 0xf172d73e004fc90d, 0xdfd1c4febcc81238, 0xbc8dfb627fe558fc,
];

const FAST_PARTIAL_ROUND_VS_AVX512: [[u64; 24]; N_PARTIAL_ROUNDS] = [
    [
        0x0,
        0x94877900674181c3,
        0xc6c67cc37a2a2bbd,
        0xd667c2055387940f,
        0x0,
        0x94877900674181c3,
        0xc6c67cc37a2a2bbd,
        0xd667c2055387940f,
        0xba63a63e94b5ff0,
        0x99460cc41b8f079f,
        0x7ff02375ed524bb3,
        0xea0870b47a8caf0e,
        0xba63a63e94b5ff0,
        0x99460cc41b8f079f,
        0x7ff02375ed524bb3,
        0xea0870b47a8caf0e,
        0xabcad82633b7bc9d,
        0x3b8d135261052241,
        0xfb4515f5e5b0d539,
        0x3ee8011c2b37f77c,
        0xabcad82633b7bc9d,
        0x3b8d135261052241,
        0xfb4515f5e5b0d539,
        0x3ee8011c2b37f77c,
    ],
    [
        0x0,
        0xadef3740e71c726,
        0xa37bf67c6f986559,
        0xc6b16f7ed4fa1b00,
        0x0,
        0xadef3740e71c726,
        0xa37bf67c6f986559,
        0xc6b16f7ed4fa1b00,
        0x6a065da88d8bfc3c,
        0x4cabc0916844b46f,
        0x407faac0f02e78d1,
        0x7a786d9cf0852cf,
        0x6a065da88d8bfc3c,
        0x4cabc0916844b46f,
        0x407faac0f02e78d1,
        0x7a786d9cf0852cf,
        0x42433fb6949a629a,
        0x891682a147ce43b0,
        0x26cfd58e7b003b55,
        0x2bbf0ed7b657acb3,
        0x42433fb6949a629a,
        0x891682a147ce43b0,
        0x26cfd58e7b003b55,
        0x2bbf0ed7b657acb3,
    ],
    [
        0x0,
        0x481ac7746b159c67,
        0xe367de32f108e278,
        0x73f260087ad28bec,
        0x0,
        0x481ac7746b159c67,
        0xe367de32f108e278,
        0x73f260087ad28bec,
        0x5cfc82216bc1bdca,
        0xcaccc870a2663a0e,
        0xdb69cd7b4298c45d,
        0x7bc9e0c57243e62d,
        0x5cfc82216bc1bdca,
        0xcaccc870a2663a0e,
        0xdb69cd7b4298c45d,
        0x7bc9e0c57243e62d,
        0x3cc51c5d368693ae,
        0x366b4e8cc068895b,
        0x2bd18715cdabbca4,
        0xa752061c4f33b8cf,
        0x3cc51c5d368693ae,
        0x366b4e8cc068895b,
        0x2bd18715cdabbca4,
        0xa752061c4f33b8cf,
    ],
    [
        0x0,
        0xb22d2432b72d5098,
        0x9e18a487f44d2fe4,
        0x4b39e14ce22abd3c,
        0x0,
        0xb22d2432b72d5098,
        0x9e18a487f44d2fe4,
        0x4b39e14ce22abd3c,
        0x9e77fde2eb315e0d,
        0xca5e0385fe67014d,
        0xc2cb99bf1b6bddb,
        0x99ec1cd2a4460bfe,
        0x9e77fde2eb315e0d,
        0xca5e0385fe67014d,
        0xc2cb99bf1b6bddb,
        0x99ec1cd2a4460bfe,
        0x8577a815a2ff843f,
        0x7d80a6b4fd6518a5,
        0xeb6c67123eab62cb,
        0x8f7851650eca21a5,
        0x8577a815a2ff843f,
        0x7d80a6b4fd6518a5,
        0xeb6c67123eab62cb,
        0x8f7851650eca21a5,
    ],
    [
        0x0,
        0x11ba9a1b81718c2a,
        0x9f7d798a3323410c,
        0xa821855c8c1cf5e5,
        0x0,
        0x11ba9a1b81718c2a,
        0x9f7d798a3323410c,
        0xa821855c8c1cf5e5,
        0x535e8d6fac0031b2,
        0x404e7c751b634320,
        0xa729353f6e55d354,
        0x4db97d92e58bb831,
        0x535e8d6fac0031b2,
        0x404e7c751b634320,
        0xa729353f6e55d354,
        0x4db97d92e58bb831,
        0xb53926c27897bf7d,
        0x965040d52fe115c5,
        0x9565fa41ebd31fd7,
        0xaae4438c877ea8f4,
        0xb53926c27897bf7d,
        0x965040d52fe115c5,
        0x9565fa41ebd31fd7,
        0xaae4438c877ea8f4,
    ],
    [
        0x0,
        0x37f4e36af6073c6e,
        0x4edc0918210800e9,
        0xc44998e99eae4188,
        0x0,
        0x37f4e36af6073c6e,
        0x4edc0918210800e9,
        0xc44998e99eae4188,
        0x9f4310d05d068338,
        0x9ec7fe4350680f29,
        0xc5b2c1fdc0b50874,
        0xa01920c5ef8b2ebe,
        0x9f4310d05d068338,
        0x9ec7fe4350680f29,
        0xc5b2c1fdc0b50874,
        0xa01920c5ef8b2ebe,
        0x59fa6f8bd91d58ba,
        0x8bfc9eb89b515a82,
        0xbe86a7a2555ae775,
        0xcbb8bbaa3810babf,
        0x59fa6f8bd91d58ba,
        0x8bfc9eb89b515a82,
        0xbe86a7a2555ae775,
        0xcbb8bbaa3810babf,
    ],
    [
        0x0,
        0x577f9a9e7ee3f9c2,
        0x88c522b949ace7b1,
        0x82f07007c8b72106,
        0x0,
        0x577f9a9e7ee3f9c2,
        0x88c522b949ace7b1,
        0x82f07007c8b72106,
        0x8283d37c6675b50e,
        0x98b074d9bbac1123,
        0x75c56fb7758317c1,
        0xfed24e206052bc72,
        0x8283d37c6675b50e,
        0x98b074d9bbac1123,
        0x75c56fb7758317c1,
        0xfed24e206052bc72,
        0x26d7c3d1bc07dae5,
        0xf88c5e441e28dbb4,
        0x4fe27f9f96615270,
        0x514d4ba49c2b14fe,
        0x26d7c3d1bc07dae5,
        0xf88c5e441e28dbb4,
        0x4fe27f9f96615270,
        0x514d4ba49c2b14fe,
    ],
    [
        0x0,
        0xf02a3ac068ee110b,
        0xa3630dafb8ae2d7,
        0xce0dc874eaf9b55c,
        0x0,
        0xf02a3ac068ee110b,
        0xa3630dafb8ae2d7,
        0xce0dc874eaf9b55c,
        0x9a95f6cff5b55c7e,
        0x626d76abfed00c7b,
        0xa0c1cf1251c204ad,
        0xdaebd3006321052c,
        0x9a95f6cff5b55c7e,
        0x626d76abfed00c7b,
        0xa0c1cf1251c204ad,
        0xdaebd3006321052c,
        0x3d4bd48b625a8065,
        0x7f1e584e071f6ed2,
        0x720574f0501caed3,
        0xe3260ba93d23540a,
        0x3d4bd48b625a8065,
        0x7f1e584e071f6ed2,
        0x720574f0501caed3,
        0xe3260ba93d23540a,
    ],
    [
        0x0,
        0xab1cbd41d8c1e335,
        0x9322ed4c0bc2df01,
        0x51c3c0983d4284e5,
        0x0,
        0xab1cbd41d8c1e335,
        0x9322ed4c0bc2df01,
        0x51c3c0983d4284e5,
        0x94178e291145c231,
        0xfd0f1a973d6b2085,
        0xd427ad96e2b39719,
        0x8a52437fecaac06b,
        0x94178e291145c231,
        0xfd0f1a973d6b2085,
        0xd427ad96e2b39719,
        0x8a52437fecaac06b,
        0xdc20ee4b8c4c9a80,
        0xa2c98e9549da2100,
        0x1603fe12613db5b6,
        0xe174929433c5505,
        0xdc20ee4b8c4c9a80,
        0xa2c98e9549da2100,
        0x1603fe12613db5b6,
        0xe174929433c5505,
    ],
    [
        0x0,
        0x3d4eab2b8ef5f796,
        0xcfff421583896e22,
        0x4143cb32d39ac3d9,
        0x0,
        0x3d4eab2b8ef5f796,
        0xcfff421583896e22,
        0x4143cb32d39ac3d9,
        0x22365051b78a5b65,
        0x6f7fd010d027c9b6,
        0xd9dd36fba77522ab,
        0xa44cf1cb33e37165,
        0x22365051b78a5b65,
        0x6f7fd010d027c9b6,
        0xd9dd36fba77522ab,
        0xa44cf1cb33e37165,
        0x3fc83d3038c86417,
        0xc4588d418e88d270,
        0xce1320f10ab80fe2,
        0xdb5eadbbec18de5d,
        0x3fc83d3038c86417,
        0xc4588d418e88d270,
        0xce1320f10ab80fe2,
        0xdb5eadbbec18de5d,
    ],
    [
        0x0,
        0x1183dfce7c454afd,
        0x21cea4aa3d3ed949,
        0xfce6f70303f2304,
        0x0,
        0x1183dfce7c454afd,
        0x21cea4aa3d3ed949,
        0xfce6f70303f2304,
        0x19557d34b55551be,
        0x4c56f689afc5bbc9,
        0xa1e920844334f944,
        0xbad66d423d2ec861,
        0x19557d34b55551be,
        0x4c56f689afc5bbc9,
        0xa1e920844334f944,
        0xbad66d423d2ec861,
        0xf318c785dc9e0479,
        0x99e2032e765ddd81,
        0x400ccc9906d66f45,
        0xe1197454db2e0dd9,
        0xf318c785dc9e0479,
        0x99e2032e765ddd81,
        0x400ccc9906d66f45,
        0xe1197454db2e0dd9,
    ],
    [
        0x0,
        0x84d1ecc4d53d2ff1,
        0xd8af8b9ceb4e11b6,
        0x335856bb527b52f4,
        0x0,
        0x84d1ecc4d53d2ff1,
        0xd8af8b9ceb4e11b6,
        0x335856bb527b52f4,
        0xc756f17fb59be595,
        0xc0654e4ea5553a78,
        0x9e9a46b61f2ea942,
        0x14fc8b5b3b809127,
        0xc756f17fb59be595,
        0xc0654e4ea5553a78,
        0x9e9a46b61f2ea942,
        0x14fc8b5b3b809127,
        0xd7009f0f103be413,
        0x3e0ee7b7a9fb4601,
        0xa74e888922085ed7,
        0xe80a7cde3d4ac526,
        0xd7009f0f103be413,
        0x3e0ee7b7a9fb4601,
        0xa74e888922085ed7,
        0xe80a7cde3d4ac526,
    ],
    [
        0x0,
        0x238aa6daa612186d,
        0x9137a5c630bad4b4,
        0xc7db3817870c5eda,
        0x0,
        0x238aa6daa612186d,
        0x9137a5c630bad4b4,
        0xc7db3817870c5eda,
        0x217e4f04e5718dc9,
        0xcae814e2817bd99d,
        0xe3292e7ab770a8ba,
        0x7bb36ef70b6b9482,
        0x217e4f04e5718dc9,
        0xcae814e2817bd99d,
        0xe3292e7ab770a8ba,
        0x7bb36ef70b6b9482,
        0x3c7835fb85bca2d3,
        0xfe2cdf8ee3c25e86,
        0x61b3915ad7274b20,
        0xeab75ca7c918e4ef,
        0x3c7835fb85bca2d3,
        0xfe2cdf8ee3c25e86,
        0x61b3915ad7274b20,
        0xeab75ca7c918e4ef,
    ],
    [
        0x0,
        0xd6e15ffc055e154e,
        0xec67881f381a32bf,
        0xfbb1196092bf409c,
        0x0,
        0xd6e15ffc055e154e,
        0xec67881f381a32bf,
        0xfbb1196092bf409c,
        0xdc9d2e07830ba226,
        0x698ef3245ff7988,
        0x194fae2974f8b576,
        0x7a5d9bea6ca4910e,
        0xdc9d2e07830ba226,
        0x698ef3245ff7988,
        0x194fae2974f8b576,
        0x7a5d9bea6ca4910e,
        0x7aebfea95ccdd1c9,
        0xf9bd38a67d5f0e86,
        0xfa65539de65492d8,
        0xf0dfcbe7653ff787,
        0x7aebfea95ccdd1c9,
        0xf9bd38a67d5f0e86,
        0xfa65539de65492d8,
        0xf0dfcbe7653ff787,
    ],
    [
        0x0,
        0xbd87ad390420258,
        0xad8617bca9e33c8,
        0xc00ad377a1e2666,
        0x0,
        0xbd87ad390420258,
        0xad8617bca9e33c8,
        0xc00ad377a1e2666,
        0xac6fc58b3f0518f,
        0xc0cc8a892cc4173,
        0xc210accb117bc21,
        0xb73630dbb46ca18,
        0xac6fc58b3f0518f,
        0xc0cc8a892cc4173,
        0xc210accb117bc21,
        0xb73630dbb46ca18,
        0xc8be4920cbd4a54,
        0xbfe877a21be1690,
        0xae790559b0ded81,
        0xbf50db2f8d6ce31,
        0xc8be4920cbd4a54,
        0xbfe877a21be1690,
        0xae790559b0ded81,
        0xbf50db2f8d6ce31,
    ],
    [
        0x0,
        0xcf29427ff7c58,
        0xbd9b3cf49eec8,
        0xd1dc8aa81fb26,
        0x0,
        0xcf29427ff7c58,
        0xbd9b3cf49eec8,
        0xd1dc8aa81fb26,
        0xbc792d5c394ef,
        0xd2ae0b2266453,
        0xd413f12c496c1,
        0xc84128cfed618,
        0xbc792d5c394ef,
        0xd2ae0b2266453,
        0xd413f12c496c1,
        0xc84128cfed618,
        0xdb5ebd48fc0d4,
        0xd1b77326dcb90,
        0xbeb0ccc145421,
        0xd10e5b22b11d1,
        0xdb5ebd48fc0d4,
        0xd1b77326dcb90,
        0xbeb0ccc145421,
        0xd10e5b22b11d1,
    ],
    [
        0x0,
        0xe24c99adad8,
        0xcf389ed4bc8,
        0xe580cbf6966,
        0x0,
        0xe24c99adad8,
        0xcf389ed4bc8,
        0xe580cbf6966,
        0xcde5fd7e04f,
        0xe63628041b3,
        0xe7e81a87361,
        0xdabe78f6d98,
        0xcde5fd7e04f,
        0xe63628041b3,
        0xe7e81a87361,
        0xdabe78f6d98,
        0xefb14cac554,
        0xe5574743b10,
        0xd05709f42c1,
        0xe4690c96af1,
        0xefb14cac554,
        0xe5574743b10,
        0xd05709f42c1,
        0xe4690c96af1,
    ],
    [
        0x0,
        0xf7157bc98,
        0xe3006d948,
        0xfa65811e6,
        0x0,
        0xf7157bc98,
        0xe3006d948,
        0xfa65811e6,
        0xe0d127e2f,
        0xfc18bfe53,
        0xfd002d901,
        0xeed6461d8,
        0xe0d127e2f,
        0xfc18bfe53,
        0xfd002d901,
        0xeed6461d8,
        0x1068562754,
        0xfa0236f50,
        0xe3af13ee1,
        0xfa460f6d1,
        0x1068562754,
        0xfa0236f50,
        0xe3af13ee1,
        0xfa460f6d1,
    ],
    [
        0x0, 0x11131738, 0xf56d588, 0x11050f86, 0x0, 0x11131738, 0xf56d588, 0x11050f86, 0xf848f4f,
        0x111527d3, 0x114369a1, 0x106f2f38, 0xf848f4f, 0x111527d3, 0x114369a1, 0x106f2f38,
        0x11e2ca94, 0x110a29f0, 0xfa9f5c1, 0x10f625d1, 0x11e2ca94, 0x110a29f0, 0xfa9f5c1,
        0x10f625d1,
    ],
    [
        0x0, 0x11f718, 0x10b6c8, 0x134a96, 0x0, 0x11f718, 0x10b6c8, 0x134a96, 0x10cf7f, 0x124d03,
        0x13f8a1, 0x117c58, 0x10cf7f, 0x124d03, 0x13f8a1, 0x117c58, 0x132c94, 0x134fc0, 0x10a091,
        0x128961, 0x132c94, 0x134fc0, 0x10a091, 0x128961,
    ],
    [
        0x0, 0x1300, 0x1750, 0x114e, 0x0, 0x1300, 0x1750, 0x114e, 0x131f, 0x167b, 0x1371, 0x1230,
        0x131f, 0x167b, 0x1371, 0x1230, 0x182c, 0x1368, 0xf31, 0x15c9, 0x182c, 0x1368, 0xf31,
        0x15c9,
    ],
    [
        0x0, 0x14, 0x22, 0x12, 0x0, 0x14, 0x22, 0x12, 0x27, 0xd, 0xd, 0x1c, 0x27, 0xd, 0xd, 0x1c,
        0x2, 0x10, 0x29, 0xf, 0x2, 0x10, 0x29, 0xf,
    ],
];

#[allow(unused)]
#[inline(always)]
#[unroll_for_loops]
fn mds_partial_layer_init_avx<F>(state: &mut [F; SPONGE_WIDTH])
where
    F: PrimeField64,
{
    let mut result = [F::ZERO; SPONGE_WIDTH];
    let res0 = state[0];
    unsafe {
        let mut r0 = _mm512_loadu_epi64((&mut result[0..8]).as_mut_ptr().cast::<i64>());
        let mut r1 = _mm512_loadu_epi64((&mut result[4..12]).as_mut_ptr().cast::<i64>());

        for r in 1..12 {
            let sr512 = _mm512_set_epi64(
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
                state[r].to_canonical_u64() as i64,
            );
            let t0 = _mm512_loadu_epi64(
                (&FAST_PARTIAL_ROUND_INITIAL_MATRIX[r][0..8])
                    .as_ptr()
                    .cast::<i64>(),
            );
            let t1 = _mm512_loadu_epi64(
                (&FAST_PARTIAL_ROUND_INITIAL_MATRIX[r][4..12])
                    .as_ptr()
                    .cast::<i64>(),
            );
            let m0 = mult_avx512(&sr512, &t0);
            let m1 = mult_avx512(&sr512, &t1);
            r0 = add_avx512(&r0, &m0);
            r1 = add_avx512(&r1, &m1);
        }
        _mm512_storeu_epi64((state[0..8]).as_mut_ptr().cast::<i64>(), r0);
        _mm512_storeu_epi64((state[4..12]).as_mut_ptr().cast::<i64>(), r1);
        state[0] = res0;
    }
}

#[allow(unused)]
#[inline(always)]
#[unroll_for_loops]
fn partial_first_constant_layer_avx<F>(state: &mut [F; SPONGE_WIDTH])
where
    F: PrimeField64,
{
    unsafe {
        let c0 = _mm512_loadu_epi64(
            (&FAST_PARTIAL_FIRST_ROUND_CONSTANT[0..8])
                .as_ptr()
                .cast::<i64>(),
        );
        let c1 = _mm512_loadu_epi64(
            (&FAST_PARTIAL_FIRST_ROUND_CONSTANT[4..12])
                .as_ptr()
                .cast::<i64>(),
        );
        let mut s0 = _mm512_loadu_epi64((state[0..8]).as_ptr().cast::<i64>());
        let mut s1 = _mm512_loadu_epi64((state[4..12]).as_ptr().cast::<i64>());
        s0 = add_avx512(&s0, &c0);
        s1 = add_avx512(&s1, &c1);
        _mm512_storeu_epi64((state[0..8]).as_mut_ptr().cast::<i64>(), s0);
        _mm512_storeu_epi64((state[4..12]).as_mut_ptr().cast::<i64>(), s1);
    }
}

#[inline(always)]
fn sbox_monomial<F>(x: F) -> F
where
    F: PrimeField64,
{
    // x |--> x^7
    let x2 = x.square();
    let x4 = x2.square();
    let x3 = x * x2;
    x3 * x4
}

#[inline(always)]
unsafe fn fft2_real_avx512(x0: &__m512i, x1: &__m512i) -> (__m512i, __m512i) {
    let y0 = _mm512_add_epi64(*x0, *x1);
    let y1 = _mm512_sub_epi64(*x0, *x1);
    (y0, y1)
}

#[inline(always)]
unsafe fn fft4_real_avx512(
    x0: &__m512i,
    x1: &__m512i,
    x2: &__m512i,
    x3: &__m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
    let zeros = _mm512_xor_si512(*x0, *x0); // faster 0
    let (z0, z2) = fft2_real_avx512(x0, x2);
    let (z1, z3) = fft2_real_avx512(x1, x3);
    let y0 = _mm512_add_epi64(z0, z1);
    let y2 = _mm512_sub_epi64(z0, z1);
    let y3 = _mm512_sub_epi64(zeros, z3);
    (y0, z2, y3, y2)
}

#[inline(always)]
unsafe fn ifft2_real_unreduced_avx512(y0: &__m512i, y1: &__m512i) -> (__m512i, __m512i) {
    let x0 = _mm512_add_epi64(*y0, *y1);
    let x1 = _mm512_sub_epi64(*y0, *y1);
    (x0, x1)
}

#[inline(always)]
unsafe fn ifft4_real_unreduced_avx512(
    y: (__m512i, (__m512i, __m512i), __m512i),
) -> (__m512i, __m512i, __m512i, __m512i) {
    let zeros = _mm512_xor_si512(y.0, y.0); // faster 0
    let z0 = _mm512_add_epi64(y.0, y.2);
    let z1 = _mm512_sub_epi64(y.0, y.2);
    let z2 = y.1 .0;
    let z3 = _mm512_sub_epi64(zeros, y.1 .1);
    let (x0, x2) = ifft2_real_unreduced_avx512(&z0, &z2);
    let (x1, x3) = ifft2_real_unreduced_avx512(&z1, &z3);
    (x0, x1, x2, x3)
}

#[inline]
pub unsafe fn add64_no_carry_avx512(a: &__m512i, b: &__m512i) -> (__m512i, __m512i) {
    /*
     * a and b are signed 4 x i64. Suppose a and b represent only one i64, then:
     * - (test 1): if a < 2^63 and b < 2^63 (this means a >= 0 and b >= 0) => sum does not overflow => cout = 0
     * - if a >= 2^63 and b >= 2^63 => sum overflows so sum = a + b and cout = 1
     * - (test 2): if (a < 2^63 and b >= 2^63) or (a >= 2^63 and b < 2^63)
     *   - (test 3): if a + b < 2^64 (this means a + b is negative in signed representation) => no overflow so cout = 0
     *   - (test 3): if a + b >= 2^64 (this means a + b becomes positive in signed representation, that is, a + b >= 0) => there is overflow so cout = 1
     */
    let ones = _mm512_load_epi64(FC.ONE_V.as_ptr().cast::<i64>());
    let zeros = _mm512_xor_si512(*a, *a); // faster 0
    let r = _mm512_add_epi64(*a, *b);
    let ma = _mm512_cmpgt_epi64_mask(zeros, *a);
    let mb = _mm512_cmpgt_epi64_mask(zeros, *b);
    let mc = _mm512_cmpgt_epi64_mask(zeros, r);
    // let m = (ma & mb) | (!mc & ((!ma & mb) | (ma & !mb)));
    let m = (ma & mb) | (!mc & (ma ^ mb));
    let co = _mm512_mask_blend_epi64(m, zeros, ones);
    (r, co)
}

#[inline]
pub unsafe fn mul64_no_overflow_avx512(a: &__m512i, b: &__m512i) -> __m512i {
    /*
    // long version
    let r = _mm512_mul_epu32(*a, *b);
    let ah = _mm512_srli_epi64(*a, 32);
    let bh = _mm512_srli_epi64(*b, 32);
    let r1 = _mm512_mul_epu32(*a, bh);
    let r1 = _mm512_slli_epi64(r1, 32);
    let r = _mm512_add_epi64(r, r1);
    let r1 = _mm512_mul_epu32(ah, *b);
    let r1 = _mm512_slli_epi64(r1, 32);
    let r = _mm512_add_epi64(r, r1);
    r
    */
    // short version
    _mm512_mullo_epi64(*a, *b)
}

#[inline(always)]
unsafe fn block1_avx512(x: &__m512i, y: [i64; 3]) -> __m512i {
    let x0 = _mm512_permutex_epi64(*x, 0x0);
    let x1 = _mm512_permutex_epi64(*x, 0x55);
    let x2 = _mm512_permutex_epi64(*x, 0xAA);

    let f0 = _mm512_set_epi64(0, y[2], y[1], y[0], 0, y[2], y[1], y[0]);
    let f1 = _mm512_set_epi64(0, y[1], y[0], y[2], 0, y[1], y[0], y[2]);
    let f2 = _mm512_set_epi64(0, y[0], y[2], y[1], 0, y[0], y[2], y[1]);

    let t0 = mul64_no_overflow_avx512(&x0, &f0);
    let t1 = mul64_no_overflow_avx512(&x1, &f1);
    let t2 = mul64_no_overflow_avx512(&x2, &f2);

    let t0 = _mm512_add_epi64(t0, t1);
    _mm512_add_epi64(t0, t2)
}

#[allow(unused)]
#[inline(always)]
unsafe fn block2_avx512(xr: &__m512i, xi: &__m512i, y: [(i64, i64); 3]) -> (__m512i, __m512i) {
    let mut vxr: [i64; 8] = [0; 8];
    let mut vxi: [i64; 8] = [0; 8];
    _mm512_storeu_epi64(vxr.as_mut_ptr().cast::<i64>(), *xr);
    _mm512_storeu_epi64(vxi.as_mut_ptr().cast::<i64>(), *xi);
    let x1: [(i64, i64); 3] = [(vxr[0], vxi[0]), (vxr[1], vxi[1]), (vxr[2], vxi[2])];
    let x2: [(i64, i64); 3] = [(vxr[4], vxi[4]), (vxr[5], vxi[5]), (vxr[6], vxi[6])];
    let b1 = block2(x1, y);
    let b2 = block2(x2, y);
    vxr = [b1[0].0, b1[1].0, b1[2].0, 0, b2[0].0, b2[1].0, b2[2].0, 0];
    vxi = [b1[0].1, b1[1].1, b1[2].1, 0, b2[0].1, b2[1].1, b2[2].1, 0];
    let rr = _mm512_loadu_epi64(vxr.as_ptr().cast::<i64>());
    let ri = _mm512_loadu_epi64(vxi.as_ptr().cast::<i64>());
    (rr, ri)
}

#[allow(dead_code)]
#[inline(always)]
unsafe fn block2_full_avx512(xr: &__m512i, xi: &__m512i, y: [(i64, i64); 3]) -> (__m512i, __m512i) {
    let yr = _mm512_set_epi64(0, y[2].0, y[1].0, y[0].0, 0, y[2].0, y[1].0, y[0].0);
    let yi = _mm512_set_epi64(0, y[2].1, y[1].1, y[0].1, 0, y[2].1, y[1].1, y[0].1);
    let ys = _mm512_add_epi64(yr, yi);
    let xs = _mm512_add_epi64(*xr, *xi);

    // z0
    // z0r = dif2[0] + prod[1] - sum[1] + prod[2] - sum[2]
    // z0i = prod[0] - sum[0] + dif1[1] + dif1[2]
    let yy = _mm512_permutex_epi64(yr, 0x18);
    let mr_z0 = mul64_no_overflow_avx512(xr, &yy);
    let yy = _mm512_permutex_epi64(yi, 0x18);
    let mi_z0 = mul64_no_overflow_avx512(xi, &yy);
    let sum = _mm512_add_epi64(mr_z0, mi_z0);
    let dif1 = _mm512_sub_epi64(mi_z0, mr_z0);
    let dif2 = _mm512_sub_epi64(mr_z0, mi_z0);
    let yy = _mm512_permutex_epi64(ys, 0x18);
    let prod = mul64_no_overflow_avx512(&xs, &yy);
    let dif3 = _mm512_sub_epi64(prod, sum);
    let dif3perm1 = _mm512_permutex_epi64(dif3, 0x1);
    let dif3perm2 = _mm512_permutex_epi64(dif3, 0x2);
    let z0r = _mm512_add_epi64(dif2, dif3perm1);
    let z0r = _mm512_add_epi64(z0r, dif3perm2);
    let dif1perm1 = _mm512_permutex_epi64(dif1, 0x1);
    let dif1perm2 = _mm512_permutex_epi64(dif1, 0x2);
    let z0i = _mm512_add_epi64(dif3, dif1perm1);
    let z0i = _mm512_add_epi64(z0i, dif1perm2);
    let zeros = _mm512_xor_si512(z0r, z0r);
    let z0r = _mm512_mask_blend_epi64(0x11, zeros, z0r);
    let z0i = _mm512_mask_blend_epi64(0x11, zeros, z0i);

    // z1
    // z1r = dif2[0] + dif2[1] + prod[2] - sum[2];
    // z1i = prod[0] - sum[0] + prod[1] - sum[1] + dif1[2];
    let yy = _mm512_permutex_epi64(yr, 0x21);
    let mr_z1 = mul64_no_overflow_avx512(xr, &yy);
    let yy = _mm512_permutex_epi64(yi, 0x21);
    let mi_z1 = mul64_no_overflow_avx512(xi, &yy);
    let sum = _mm512_add_epi64(mr_z1, mi_z1);
    let dif1 = _mm512_sub_epi64(mi_z1, mr_z1);
    let dif2 = _mm512_sub_epi64(mr_z1, mi_z1);
    let yy = _mm512_permutex_epi64(ys, 0x21);
    let prod = mul64_no_overflow_avx512(&xs, &yy);
    let dif3 = _mm512_sub_epi64(prod, sum);
    let dif2perm = _mm512_permutex_epi64(dif2, 0x0);
    let dif3perm = _mm512_permutex_epi64(dif3, 0x8);
    let z1r = _mm512_add_epi64(dif2, dif2perm);
    let z1r = _mm512_add_epi64(z1r, dif3perm);
    let dif3perm = _mm512_permutex_epi64(dif3, 0x0);
    let dif1perm = _mm512_permutex_epi64(dif1, 0x8);
    let z1i = _mm512_add_epi64(dif3, dif3perm);
    let z1i = _mm512_add_epi64(z1i, dif1perm);
    let z1r = _mm512_mask_blend_epi64(0x22, zeros, z1r);
    let z1i = _mm512_mask_blend_epi64(0x22, zeros, z1i);

    // z2
    // z2r = dif2[0] + dif2[1] + dif2[2];
    // z2i = prod[0] - sum[0] + prod[1] - sum[1] + prod[2] - sum[2]
    let yy = _mm512_permutex_epi64(yr, 0x6);
    let mr_z2 = mul64_no_overflow_avx512(xr, &yy);
    let yy = _mm512_permutex_epi64(yi, 0x6);
    let mi_z2 = mul64_no_overflow_avx512(xi, &yy);
    let sum = _mm512_add_epi64(mr_z2, mi_z2);
    let dif2 = _mm512_sub_epi64(mr_z2, mi_z2);
    let yy = _mm512_permutex_epi64(ys, 0x6);
    let prod = mul64_no_overflow_avx512(&xs, &yy);
    let dif3 = _mm512_sub_epi64(prod, sum);
    let dif2perm1 = _mm512_permutex_epi64(dif2, 0x0);
    let dif2perm2 = _mm512_permutex_epi64(dif2, 0x10);
    let z2r = _mm512_add_epi64(dif2, dif2perm1);
    let z2r = _mm512_add_epi64(z2r, dif2perm2);
    let dif3perm1 = _mm512_permutex_epi64(dif3, 0x0);
    let dif3perm2 = _mm512_permutex_epi64(dif3, 0x10);
    let z2i = _mm512_add_epi64(dif3, dif3perm1);
    let z2i = _mm512_add_epi64(z2i, dif3perm2);
    let z2r = _mm512_mask_blend_epi64(0x44, zeros, z2r);
    let z2i = _mm512_mask_blend_epi64(0x44, zeros, z2i);

    let zr = _mm512_or_si512(z0r, z1r);
    let zr = _mm512_or_si512(zr, z2r);
    let zi = _mm512_or_si512(z0i, z1i);
    let zi = _mm512_or_si512(zi, z2i);
    (zr, zi)
}

#[inline(always)]
unsafe fn block3_avx512(x: &__m512i, y: [i64; 3]) -> __m512i {
    let x0 = _mm512_permutex_epi64(*x, 0x0);
    let x1 = _mm512_permutex_epi64(*x, 0x55);
    let x2 = _mm512_permutex_epi64(*x, 0xAA);

    let f0 = _mm512_set_epi64(0, y[2], y[1], y[0], 0, y[2], y[1], y[0]);
    let f1 = _mm512_set_epi64(0, y[1], y[0], -y[2], 0, y[1], y[0], -y[2]);
    let f2 = _mm512_set_epi64(0, y[0], -y[2], -y[1], 0, y[0], -y[2], -y[1]);

    let t0 = mul64_no_overflow_avx512(&x0, &f0);
    let t1 = mul64_no_overflow_avx512(&x1, &f1);
    let t2 = mul64_no_overflow_avx512(&x2, &f2);

    let t0 = _mm512_add_epi64(t0, t1);
    _mm512_add_epi64(t0, t2)
}

#[inline]
unsafe fn mds_multiply_freq_avx512(s0: &mut __m512i, s1: &mut __m512i, s2: &mut __m512i) {
    /*
    // Alternative code using store and set.
    let mut s: [i64; 12] = [0; 12];
    _mm256_storeu_si256(s[0..4].as_mut_ptr().cast::<__m256i>(), *s0);
    _mm256_storeu_si256(s[4..8].as_mut_ptr().cast::<__m256i>(), *s1);
    _mm256_storeu_si256(s[8..12].as_mut_ptr().cast::<__m256i>(), *s2);
    let f0 = _mm256_set_epi64x(0, s[2], s[1], s[0]);
    let f1 = _mm256_set_epi64x(0, s[5], s[4], s[3]);
    let f2 = _mm256_set_epi64x(0, s[8], s[7], s[6]);
    let f3 = _mm256_set_epi64x(0, s[11], s[10], s[9]);
    */

    // Alternative code using permute and blend (it is faster).
    let f0 = *s0;
    let f11 = _mm512_permutex_epi64(*s0, 0x3);
    let f12 = _mm512_permutex_epi64(*s1, 0x10);
    let f1 = _mm512_mask_blend_epi64(0x66, f11, f12);
    let f21 = _mm512_permutex_epi64(*s1, 0xE);
    let f22 = _mm512_permutex_epi64(*s2, 0x0);
    let f2 = _mm512_mask_blend_epi64(0x44, f21, f22);
    let f3 = _mm512_permutex_epi64(*s2, 0x39);

    let (u0, u1, u2, u3) = fft4_real_avx512(&f0, &f1, &f2, &f3);

    // let [v0, v4, v8] = block1_avx([u[0], u[1], u[2]], MDS_FREQ_BLOCK_ONE);
    // [u[0], u[1], u[2]] are all in u0
    let f0 = block1_avx512(&u0, MDS_FREQ_BLOCK_ONE);

    // let [v1, v5, v9] = block2([(u[0], v[0]), (u[1], v[1]), (u[2], v[2])], MDS_FREQ_BLOCK_TWO);
    // let (f1, f2) = block2_avx512(&u1, &u2, MDS_FREQ_BLOCK_TWO);
    let (f1, f2) = block2_full_avx512(&u1, &u2, MDS_FREQ_BLOCK_TWO);

    // let [v2, v6, v10] = block3_avx([u[0], u[1], u[2]], MDS_FREQ_BLOCK_ONE);
    // [u[0], u[1], u[2]] are all in u3
    let f3 = block3_avx512(&u3, MDS_FREQ_BLOCK_THREE);

    let (r0, r3, r6, r9) = ifft4_real_unreduced_avx512((f0, (f1, f2), f3));
    let t = _mm512_permutex_epi64(r3, 0x0);
    *s0 = _mm512_mask_blend_epi64(0x88, r0, t);
    let t1 = _mm512_permutex_epi64(r3, 0x9);
    let t2 = _mm512_permutex_epi64(r6, 0x40);
    *s1 = _mm512_mask_blend_epi64(0xCC, t1, t2);
    let t1 = _mm512_permutex_epi64(r6, 0x2);
    let t2 = _mm512_permutex_epi64(r9, 0x90);
    *s2 = _mm512_mask_blend_epi64(0xEE, t1, t2);
}

#[inline(always)]
#[unroll_for_loops]
unsafe fn mds_layer_avx512(s0: &mut __m512i, s1: &mut __m512i, s2: &mut __m512i) {
    let mask = _mm512_load_epi64(FC.P8_N_V.as_ptr().cast::<i64>());
    let mut sl0 = _mm512_and_si512(*s0, mask);
    let mut sl1 = _mm512_and_si512(*s1, mask);
    let mut sl2 = _mm512_and_si512(*s2, mask);
    let mut sh0 = _mm512_srli_epi64(*s0, 32);
    let mut sh1 = _mm512_srli_epi64(*s1, 32);
    let mut sh2 = _mm512_srli_epi64(*s2, 32);

    mds_multiply_freq_avx512(&mut sl0, &mut sl1, &mut sl2);
    mds_multiply_freq_avx512(&mut sh0, &mut sh1, &mut sh2);

    let shl0 = _mm512_slli_epi64(sh0, 32);
    let shl1 = _mm512_slli_epi64(sh1, 32);
    let shl2 = _mm512_slli_epi64(sh2, 32);
    let shh0 = _mm512_srli_epi64(sh0, 32);
    let shh1 = _mm512_srli_epi64(sh1, 32);
    let shh2 = _mm512_srli_epi64(sh2, 32);

    let (rl0, c0) = add64_no_carry_avx512(&sl0, &shl0);
    let (rh0, _) = add64_no_carry_avx512(&shh0, &c0);
    let r0 = reduce_avx512_96_64(&rh0, &rl0);

    let (rl1, c1) = add64_no_carry_avx512(&sl1, &shl1);
    let (rh1, _) = add64_no_carry_avx512(&shh1, &c1);
    *s1 = reduce_avx512_96_64(&rh1, &rl1);

    let (rl2, c2) = add64_no_carry_avx512(&sl2, &shl2);
    let (rh2, _) = add64_no_carry_avx512(&shh2, &c2);
    *s2 = reduce_avx512_96_64(&rh2, &rl2);

    let rl = _mm512_slli_epi64(*s0, 3); // * 8 (low part)
    let rh = _mm512_srli_epi64(*s0, 61); // * 8 (high part, only 3 bits)
    let rx = reduce_avx512_96_64(&rh, &rl);
    let rx = add_avx512(&r0, &rx);
    *s0 = _mm512_mask_blend_epi64(0x11, r0, rx);
}

#[unroll_for_loops]
unsafe fn mds_partial_layer_init_avx512<F>(s0: &mut __m512i, s1: &mut __m512i, s2: &mut __m512i)
where
    F: PrimeField64,
{
    let res0 = *s0;
    let mut r0 = _mm512_xor_epi64(res0, res0);
    let mut r1 = r0;
    let mut r2 = r0;
    for r in 1..12 {
        if r < 12 {
            let sr = match r {
                1 => _mm512_permutex_epi64(*s0, 0x55),
                2 => _mm512_permutex_epi64(*s0, 0xAA),
                3 => _mm512_permutex_epi64(*s0, 0xFF),
                4 => _mm512_permutex_epi64(*s1, 0x0),
                5 => _mm512_permutex_epi64(*s1, 0x55),
                6 => _mm512_permutex_epi64(*s1, 0xAA),
                7 => _mm512_permutex_epi64(*s1, 0xFF),
                8 => _mm512_permutex_epi64(*s2, 0x0),
                9 => _mm512_permutex_epi64(*s2, 0x55),
                10 => _mm512_permutex_epi64(*s2, 0xAA),
                11 => _mm512_permutex_epi64(*s2, 0xFF),
                _ => _mm512_permutex_epi64(*s0, 0x55),
            };
            let t0 = _mm512_loadu_epi64(
                (&FAST_PARTIAL_ROUND_INITIAL_MATRIX_AVX512[r][0..8])
                    .as_ptr()
                    .cast::<i64>(),
            );
            let t1 = _mm512_loadu_epi64(
                (&FAST_PARTIAL_ROUND_INITIAL_MATRIX_AVX512[r][8..16])
                    .as_ptr()
                    .cast::<i64>(),
            );
            let t2 = _mm512_loadu_epi64(
                (&FAST_PARTIAL_ROUND_INITIAL_MATRIX_AVX512[r][16..24])
                    .as_ptr()
                    .cast::<i64>(),
            );
            let m0 = mult_avx512(&sr, &t0);
            let m1 = mult_avx512(&sr, &t1);
            let m2 = mult_avx512(&sr, &t2);
            r0 = add_avx512(&r0, &m0);
            r1 = add_avx512(&r1, &m1);
            r2 = add_avx512(&r2, &m2);
        }
    }
    *s0 = _mm512_mask_blend_epi64(0x11, r0, res0);
    *s1 = r1;
    *s2 = r2;
}

#[inline(always)]
#[unroll_for_loops]
unsafe fn mds_partial_layer_fast_avx512<F>(
    s0: &mut __m512i,
    s1: &mut __m512i,
    s2: &mut __m512i,
    state: &mut [F; 2 * SPONGE_WIDTH],
    r: usize,
) where
    F: PrimeField64,
{
    let mut d_sum1 = (0u128, 0u32); // u160 accumulator
    let mut d_sum2 = (0u128, 0u32); // u160 accumulator
    for i in 1..4 {
        let t = FAST_PARTIAL_ROUND_W_HATS[r][i - 1] as u128;
        let si1 = state[i].to_noncanonical_u64() as u128;
        let si2 = state[i + 4].to_noncanonical_u64() as u128;
        d_sum1 = add_u160_u128(d_sum1, si1 * t);
        d_sum2 = add_u160_u128(d_sum2, si2 * t);
    }
    for i in 4..8 {
        let t = FAST_PARTIAL_ROUND_W_HATS[r][i - 1] as u128;
        let si1 = state[i + 4].to_noncanonical_u64() as u128;
        let si2 = state[i + 8].to_noncanonical_u64() as u128;
        d_sum1 = add_u160_u128(d_sum1, si1 * t);
        d_sum2 = add_u160_u128(d_sum2, si2 * t);
    }
    for i in 8..12 {
        let t = FAST_PARTIAL_ROUND_W_HATS[r][i - 1] as u128;
        let si1 = state[i + 8].to_noncanonical_u64() as u128;
        let si2 = state[i + 12].to_noncanonical_u64() as u128;
        d_sum1 = add_u160_u128(d_sum1, si1 * t);
        d_sum2 = add_u160_u128(d_sum2, si2 * t);
    }
    // 1st
    let x0_1 = state[0].to_noncanonical_u64() as u128;
    let mds0to0_1 = (MDS_MATRIX_CIRC[0] + MDS_MATRIX_DIAG[0]) as u128;
    d_sum1 = add_u160_u128(d_sum1, x0_1 * mds0to0_1);
    let d1 = reduce_u160::<F>(d_sum1);
    // 2nd
    let x0_2 = state[4].to_noncanonical_u64() as u128;
    let mds0to0_2 = (MDS_MATRIX_CIRC[0] + MDS_MATRIX_DIAG[0]) as u128;
    d_sum2 = add_u160_u128(d_sum2, x0_2 * mds0to0_2);
    let d2 = reduce_u160::<F>(d_sum2);

    // result = [d] concat [state[0] * v + state[shift up by 1]]
    let ss0 = _mm512_set_epi64(
        state[4].to_noncanonical_u64() as i64,
        state[4].to_noncanonical_u64() as i64,
        state[4].to_noncanonical_u64() as i64,
        state[4].to_noncanonical_u64() as i64,
        state[0].to_noncanonical_u64() as i64,
        state[0].to_noncanonical_u64() as i64,
        state[0].to_noncanonical_u64() as i64,
        state[0].to_noncanonical_u64() as i64,
    );
    let rc0 = _mm512_loadu_epi64(
        (&FAST_PARTIAL_ROUND_VS_AVX512[r][0..8])
            .as_ptr()
            .cast::<i64>(),
    );
    let rc1 = _mm512_loadu_epi64(
        (&FAST_PARTIAL_ROUND_VS_AVX512[r][8..16])
            .as_ptr()
            .cast::<i64>(),
    );
    let rc2 = _mm512_loadu_epi64(
        (&FAST_PARTIAL_ROUND_VS_AVX512[r][16..24])
            .as_ptr()
            .cast::<i64>(),
    );
    let (mh, ml) = mult_avx512_128(&ss0, &rc0);
    let m = reduce_avx512_128_64(&mh, &ml);
    let r0 = add_avx512(s0, &m);
    let d0 = _mm512_set_epi64(
        0,
        0,
        0,
        d2.to_canonical_u64() as i64,
        0,
        0,
        0,
        d1.to_canonical_u64() as i64,
    );
    *s0 = _mm512_mask_blend_epi64(0x11, r0, d0);

    let (mh, ml) = mult_avx512_128(&ss0, &rc1);
    let m = reduce_avx512_128_64(&mh, &ml);
    *s1 = add_avx512(s1, &m);

    let (mh, ml) = mult_avx512_128(&ss0, &rc2);
    let m = reduce_avx512_128_64(&mh, &ml);
    *s2 = add_avx512(s2, &m);

    _mm512_storeu_epi64((state[0..8]).as_mut_ptr().cast::<i64>(), *s0);
    _mm512_storeu_epi64((state[8..16]).as_mut_ptr().cast::<i64>(), *s1);
    _mm512_storeu_epi64((state[16..24]).as_mut_ptr().cast::<i64>(), *s2);
}

#[allow(unused)]
pub fn poseidon_avx512_single<F>(input: &[F; SPONGE_WIDTH]) -> [F; SPONGE_WIDTH]
where
    F: PrimeField64 + Poseidon,
{
    let mut state = &mut input.clone();
    let mut round_ctr = 0;

    unsafe {
        // Self::full_rounds(&mut state, &mut round_ctr);
        for _ in 0..HALF_N_FULL_ROUNDS {
            // load state
            let s0 = _mm512_loadu_epi64((&state[0..8]).as_ptr().cast::<i64>());
            let s1 = _mm512_loadu_epi64((&state[4..12]).as_ptr().cast::<i64>());

            let rc: &[u64; 12] = &ALL_ROUND_CONSTANTS[SPONGE_WIDTH * round_ctr..][..SPONGE_WIDTH]
                .try_into()
                .unwrap();
            let rc0 = _mm512_loadu_epi64((&rc[0..8]).as_ptr().cast::<i64>());
            let rc1 = _mm512_loadu_epi64((&rc[4..12]).as_ptr().cast::<i64>());
            let ss0 = add_avx512(&s0, &rc0);
            let ss1 = add_avx512(&s1, &rc1);
            let r0 = sbox_avx512_one(&ss0);
            let r1 = sbox_avx512_one(&ss1);

            // store state
            _mm512_storeu_epi64((state[0..8]).as_mut_ptr().cast::<i64>(), r0);
            _mm512_storeu_epi64((state[4..12]).as_mut_ptr().cast::<i64>(), r1);

            *state = <F as Poseidon>::mds_layer(&state);
            round_ctr += 1;
        }
        partial_first_constant_layer_avx(&mut state);
        mds_partial_layer_init_avx(&mut state);

        for i in 0..N_PARTIAL_ROUNDS {
            state[0] = sbox_monomial(state[0]);
            state[0] = state[0].add_canonical_u64(FAST_PARTIAL_ROUND_CONSTANTS[i]);
            *state = <F as Poseidon>::mds_partial_layer_fast(&state, i);
        }
        round_ctr += N_PARTIAL_ROUNDS;

        // Self::full_rounds(&mut state, &mut round_ctr);
        for _ in 0..HALF_N_FULL_ROUNDS {
            // load state
            let s0 = _mm512_loadu_epi64((&state[0..8]).as_ptr().cast::<i64>());
            let s1 = _mm512_loadu_epi64((&state[4..12]).as_ptr().cast::<i64>());

            let rc: &[u64; 12] = &ALL_ROUND_CONSTANTS[SPONGE_WIDTH * round_ctr..][..SPONGE_WIDTH]
                .try_into()
                .unwrap();
            let rc0 = _mm512_loadu_epi64((&rc[0..8]).as_ptr().cast::<i64>());
            let rc1 = _mm512_loadu_epi64((&rc[4..12]).as_ptr().cast::<i64>());
            let ss0 = add_avx512(&s0, &rc0);
            let ss1 = add_avx512(&s1, &rc1);
            let r0 = sbox_avx512_one(&ss0);
            let r1 = sbox_avx512_one(&ss1);

            // store state
            _mm512_storeu_epi64((state[0..8]).as_mut_ptr().cast::<i64>(), r0);
            _mm512_storeu_epi64((state[4..12]).as_mut_ptr().cast::<i64>(), r1);

            *state = <F as Poseidon>::mds_layer(&state);
            // mds_layer_avx::<F>(&mut s0, &mut s1, &mut s2);
            round_ctr += 1;
        }

        debug_assert_eq!(round_ctr, N_ROUNDS);
    };
    *state
}

pub fn poseidon_avx512_double<F>(input: &[F; 2 * SPONGE_WIDTH]) -> [F; 2 * SPONGE_WIDTH]
where
    F: PrimeField64 + Poseidon,
{
    let mut state: [F; 24] = input.clone();
    state[0..4].copy_from_slice(&input[0..4]);
    state[4..8].copy_from_slice(&input[12..16]);
    state[8..12].copy_from_slice(&input[4..8]);
    state[12..16].copy_from_slice(&input[16..20]);
    state[16..20].copy_from_slice(&input[8..12]);
    state[20..24].copy_from_slice(&input[20..24]);

    let mut round_ctr = 0;

    unsafe {
        // load state
        let mut s0 = _mm512_loadu_epi64((&state[0..8]).as_ptr().cast::<i64>());
        let mut s1 = _mm512_loadu_epi64((&state[8..16]).as_ptr().cast::<i64>());
        let mut s2 = _mm512_loadu_epi64((&state[16..24]).as_ptr().cast::<i64>());

        for _ in 0..HALF_N_FULL_ROUNDS {
            let rc: &[u64; 24] = &ALL_ROUND_CONSTANTS_AVX512[2 * SPONGE_WIDTH * round_ctr..]
                [..2 * SPONGE_WIDTH]
                .try_into()
                .unwrap();
            let rc0 = _mm512_loadu_epi64((&rc[0..8]).as_ptr().cast::<i64>());
            let rc1 = _mm512_loadu_epi64((&rc[8..16]).as_ptr().cast::<i64>());
            let rc2 = _mm512_loadu_epi64((&rc[16..24]).as_ptr().cast::<i64>());
            let ss0 = add_avx512(&s0, &rc0);
            let ss1 = add_avx512(&s1, &rc1);
            let ss2 = add_avx512(&s2, &rc2);
            s0 = sbox_avx512_one(&ss0);
            s1 = sbox_avx512_one(&ss1);
            s2 = sbox_avx512_one(&ss2);
            mds_layer_avx512(&mut s0, &mut s1, &mut s2);
            round_ctr += 1;
        }

        // this does partial_first_constant_layer_avx(&mut state);
        let c0 = _mm512_loadu_epi64(
            (&FAST_PARTIAL_FIRST_ROUND_CONSTANT_AVX512[0..8])
                .as_ptr()
                .cast::<i64>(),
        );
        let c1 = _mm512_loadu_epi64(
            (&FAST_PARTIAL_FIRST_ROUND_CONSTANT_AVX512[8..16])
                .as_ptr()
                .cast::<i64>(),
        );
        let c2 = _mm512_loadu_epi64(
            (&FAST_PARTIAL_FIRST_ROUND_CONSTANT_AVX512[16..24])
                .as_ptr()
                .cast::<i64>(),
        );
        s0 = add_avx512(&s0, &c0);
        s1 = add_avx512(&s1, &c1);
        s2 = add_avx512(&s2, &c2);

        mds_partial_layer_init_avx512::<F>(&mut s0, &mut s1, &mut s2);

        _mm512_storeu_epi64((state[0..8]).as_mut_ptr().cast::<i64>(), s0);
        _mm512_storeu_epi64((state[8..16]).as_mut_ptr().cast::<i64>(), s1);
        _mm512_storeu_epi64((state[16..24]).as_mut_ptr().cast::<i64>(), s2);

        for i in 0..N_PARTIAL_ROUNDS {
            state[0] = sbox_monomial(state[0]);
            state[0] = state[0].add_canonical_u64(FAST_PARTIAL_ROUND_CONSTANTS[i]);
            state[4] = sbox_monomial(state[4]);
            state[4] = state[4].add_canonical_u64(FAST_PARTIAL_ROUND_CONSTANTS[i]);
            mds_partial_layer_fast_avx512(&mut s0, &mut s1, &mut s2, &mut state, i);
        }
        round_ctr += N_PARTIAL_ROUNDS;

        // here state is already loaded in s0, s1, s2
        // Self::full_rounds(&mut state, &mut round_ctr);
        for _ in 0..HALF_N_FULL_ROUNDS {
            let rc: &[u64; 24] = &ALL_ROUND_CONSTANTS_AVX512[2 * SPONGE_WIDTH * round_ctr..]
                [..2 * SPONGE_WIDTH]
                .try_into()
                .unwrap();
            let rc0 = _mm512_loadu_epi64((&rc[0..8]).as_ptr().cast::<i64>());
            let rc1 = _mm512_loadu_epi64((&rc[8..16]).as_ptr().cast::<i64>());
            let rc2 = _mm512_loadu_epi64((&rc[16..24]).as_ptr().cast::<i64>());
            let ss0 = add_avx512(&s0, &rc0);
            let ss1 = add_avx512(&s1, &rc1);
            let ss2 = add_avx512(&s2, &rc2);
            s0 = sbox_avx512_one(&ss0);
            s1 = sbox_avx512_one(&ss1);
            s2 = sbox_avx512_one(&ss2);
            mds_layer_avx512(&mut s0, &mut s1, &mut s2);
            round_ctr += 1;
        }

        // store state
        _mm512_storeu_epi64((state[0..8]).as_mut_ptr().cast::<i64>(), s0);
        _mm512_storeu_epi64((state[8..16]).as_mut_ptr().cast::<i64>(), s1);
        _mm512_storeu_epi64((state[16..24]).as_mut_ptr().cast::<i64>(), s2);

        debug_assert_eq!(round_ctr, N_ROUNDS);
    };

    let mut new_state: [F; 24] = state.clone();
    new_state[0..4].copy_from_slice(&state[0..4]);
    new_state[4..8].copy_from_slice(&state[8..12]);
    new_state[8..12].copy_from_slice(&state[16..20]);
    new_state[12..16].copy_from_slice(&state[4..8]);
    new_state[16..20].copy_from_slice(&state[12..16]);
    new_state[20..24].copy_from_slice(&state[20..24]);
    new_state
}

pub fn hash_leaf_avx512<F>(inputs: &[F], leaf_size: usize) -> (Vec<F>, Vec<F>)
where
    F: RichField,
{
    // special case
    if leaf_size <= 4 {
        let mut h1 = vec![F::ZERO; 4];
        let mut h2 = vec![F::ZERO; 4];
        h1.copy_from_slice(&inputs[0..leaf_size]);
        h2.copy_from_slice(&inputs[leaf_size..2 * leaf_size]);
        return (h1, h2);
    }

    // general case
    let mut state: [F; 24] = [F::ZERO; 24];

    // absorb all input chunks of size SPONGE_RATE
    let mut idx1 = 0;
    let mut idx2 = leaf_size;
    let loops = if leaf_size % SPONGE_RATE == 0 {
        leaf_size / SPONGE_RATE
    } else {
        leaf_size / SPONGE_RATE + 1
    };
    for _ in 0..loops {
        let end1 = if idx1 + SPONGE_RATE >= leaf_size {
            leaf_size
        } else {
            idx1 + SPONGE_RATE
        };
        let end2 = if idx2 + SPONGE_RATE >= 2 * leaf_size {
            2 * leaf_size
        } else {
            idx2 + SPONGE_RATE
        };
        let end = end1 - idx1;
        state[0..end].copy_from_slice(&inputs[idx1..end1]);
        state[12..12 + end].copy_from_slice(&inputs[idx2..end2]);
        state = poseidon_avx512_double(&state);
        idx1 += SPONGE_RATE;
        idx2 += SPONGE_RATE;
    }

    // return 2 hashes of 4 elements each
    (
        vec![state[0], state[1], state[2], state[3]],
        vec![state[12], state[13], state[14], state[15]],
    )
}

pub fn hash_two_avx512<F>(h1: &Vec<F>, h2: &Vec<F>, h3: &Vec<F>, h4: &Vec<F>) -> (Vec<F>, Vec<F>)
where
    F: RichField,
{
    let mut state: [F; 24] = [F::ZERO; 24];
    state[0..4].copy_from_slice(&h1);
    state[4..8].copy_from_slice(&h2);
    state[12..16].copy_from_slice(&h3);
    state[16..20].copy_from_slice(&h4);
    state = poseidon_avx512_double(&state);
    (
        vec![state[0], state[1], state[2], state[3]],
        vec![state[12], state[13], state[14], state[15]],
    )
}
