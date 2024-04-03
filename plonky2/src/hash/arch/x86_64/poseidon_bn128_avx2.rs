#![cfg_attr(feature = "nightly", feature(stdsimd))]

use core::arch::asm;
use core::arch::x86_64::*;

use crate::hash::poseidon_bn128_ops::ElementBN128;

const C: [[u64; 4]; 100] = [
    [
        0x878a9569334498e4,
        0x4641e4a29d08274f,
        0xf2713820fea6f0c4,
        0x898c94bd2c76331,
    ],
    [
        0xd6dec67b3646bdbc,
        0x626a9e071b154f27,
        0x71a61cb1f9d90cbe,
        0x134dd09bc5dffaa7,
    ],
    [
        0xc24d9503f8682c8c,
        0x9cf5f5abe19fedff,
        0x125f8816cdb2d9f1,
        0x5954a7a4436fd78,
    ],
    [
        0xc306f8ed4ba6732d,
        0x5b187030689573d0,
        0xb0a9df5b5120771d,
        0x5513e9e64511461,
    ],
    [
        0x84b301dccd446ff0,
        0x59d0332079fd0d4c,
        0xcb69fbff03ebf775,
        0x1582477fe7736802,
    ],
    [
        0xd4cba791193dd512,
        0xc07dddce6dba21d5,
        0x79391672a0b6ecd2,
        0x2b13399d4308ec41,
    ],
    [
        0x3eb7b07418da854d,
        0x18df0397a244d7e3,
        0x983c1a1e41a858c5,
        0x14a4dc22dbbaf6f9,
    ],
    [
        0x317311a626a4e71c,
        0x8bfc4d5753b69402,
        0x8147d97f129bca1c,
        0x1779b47e3a5bfab,
    ],
    [
        0x969e97b2d9029781,
        0x6da6b49c2cc91cd2,
        0xf1779eeb56dc1b36,
        0x24e67809f1c36f1c,
    ],
    [
        0x96be623f30e5dab1,
        0x45d644353b9ff9af,
        0x5173775702777781,
        0x177bbab6eef5c2cc,
    ],
    [
        0x766c8f5d09003723,
        0xc35a793f1c4ef16d,
        0x1ccbcc21f8416aba,
        0xda62e07998b986d,
    ],
    [
        0x50d495b1c8b1cce2,
        0x8973e470121c3a76,
        0x1b4c8afdbe808a92,
        0x26cc2ec9d51be4d3,
    ],
    [
        0xfc8703f33d12bad0,
        0x6544a99005e01916,
        0x3e3149839d1a658e,
        0xd29900eab66c55b,
    ],
    [
        0x508ee4483fab787d,
        0x9557c4fb953244fa,
        0x150c0c527c7b0292,
        0x1c62a2de197849db,
    ],
    [
        0x9fc486f53774e8a,
        0xe31561abbcefb5cf,
        0x3d34f5bdbc156014,
        0x195b4ee8457027e1,
    ],
    [
        0xbd2efe1961e1b3c7,
        0x2b534ac8bc6e4ecb,
        0x501233ac87899cb3,
        0x1653eb7e017f9132,
    ],
    [
        0x77753115159d25f2,
        0x151334a5631b7216,
        0x16d209ff858f73eb,
        0x7c2d28738436409,
    ],
    [
        0x4ca3db03d44c586e,
        0x79fa80602ef44da6,
        0x5c89f9827c970f1d,
        0x2b2a0e263583b992,
    ],
    [
        0xe6ecdb864b836a8d,
        0x26eaf84ab4176c3c,
        0xa71eafecc6f4bc70,
        0xc1c26366f225554,
    ],
    [
        0x4239d17551cfdc1a,
        0x8d5f0841c18e1c3b,
        0x2fc1d2d58d15e408,
        0x1bae097b713e008e,
    ],
    [
        0x5a6dd19ca0fc3fa7,
        0xd51682cf90294707,
        0x18017b89769b9c26,
        0x9cd442dc7580835,
    ],
    [
        0xdb3686d6c12abc6,
        0x7c6b8216046da6eb,
        0xab8cba80ce31bd0f,
        0x723c7da54840864,
    ],
    [
        0xf9d771ae57719136,
        0xe871941125ccd77c,
        0x5877371eeb756c92,
        0x236db3bc6f7b868c,
    ],
    [
        0x1991f3a9ab1a8a1e,
        0xf1bc34fa7390b469,
        0x7a500a6fd5b5e601,
        0x47c1bff1d838d9a,
    ],
    [
        0x77d255c68da3371e,
        0x1c3d1d80af7c8849,
        0xf11e46404a393db7,
        0x244c9872ae7424f8,
    ],
    [
        0x54fc7ffde14dda4e,
        0x27f1da61c05e3c9b,
        0x2f48569498b2fc16,
        0x1bbead80194032f0,
    ],
    [
        0x5ed3dcfee7bbec96,
        0x4afe5ef835e3cbe8,
        0xc2616076444869a1,
        0x61ac6cf5b2a8fd4,
    ],
    [
        0x2792e1b29658268,
        0x45fe50fa6556f4b6,
        0xeb5a0b8c0f389162,
        0x2035c4acf7c80c91,
    ],
    [
        0x3d62bde5f5805cbf,
        0x530062cfd7a99c80,
        0xeb76c38ed028bf7e,
        0x19f568f69be678c4,
    ],
    [
        0x8aeb08756964c799,
        0x9a26e9a4a3f659ca,
        0x6dfff103f8546e1c,
        0x25952ae4d0199104,
    ],
    [
        0xcb5ba73726983c80,
        0xb63c2258f907c40f,
        0x234a8b16fa31bbe,
        0x65fb4ddea2dcdc3,
    ],
    [
        0xb7ca530b0c46cb0a,
        0x8b4b30ec102b9498,
        0x89232a1702f21d35,
        0x17b61f3f4482891a,
    ],
    [
        0x21ef0c8ef817f0b3,
        0x37e59fcfc295b10a,
        0xba190e69dcd371f0,
        0x8fc5d938192794a,
    ],
    [
        0x2490ca6341905e10,
        0x33a54aa9db48a5b6,
        0xe4f933be9632ef11,
        0x285221bf1f69c861,
    ],
    [
        0xb07dbb0743de3c5b,
        0xe81ab68bfa32ebf0,
        0x2a79628405d61365,
        0x3042b01a44f9123a,
    ],
    [
        0x941e8310bd2deb09,
        0x35e821e30b453bef,
        0xcee259921cbe111c,
        0xc06807889f9682,
    ],
    [
        0x66ff167e37db96ea,
        0xcb78ccd38eeeffdc,
        0x6f1e2690f1c90f59,
        0x1bed12cb4798cbdf,
    ],
    [
        0x70df149a76545cf3,
        0x9719c66a8f620da4,
        0x4cb01f0dc2c27a3,
        0x102abc29d3c94951,
    ],
    [
        0xfb61ac9547363bb6,
        0xb085490027d4e8eb,
        0xebb13083e19d63dd,
        0x12b63d6b9afd9a12,
    ],
    [
        0x39c5ef6deff4e0f7,
        0x38ae71161204910c,
        0xca0005aec936c88c,
        0x111cc546cbc138c5,
    ],
    [
        0x6d516cfdc4b4cd1e,
        0xf45f63ef6f96f930,
        0x83bea60478b1b76d,
        0x2f98a006fb13bcb5,
    ],
    [
        0x99dd58d1e9d818ff,
        0x7d060b0ed4121a0f,
        0x65252326690ab5e5,
        0x2e81a5ee340447bd,
    ],
    [
        0x1a1fc00d3e8c6709,
        0xbf8ae07ffb30ba1d,
        0x70e3044d7c1dbdbc,
        0x757ea27942cedfc,
    ],
    [
        0xaebdbb67c8ea3466,
        0x957ad9399fb6f98f,
        0xea7bde5d161c2f91,
        0x26cd837861787ed6,
    ],
    [
        0x43536d705f42cbd1,
        0x5ffaa914d6c6a273,
        0x23100f56d0b51cd2,
        0xd341ec358dcd032,
    ],
    [
        0x8783c40007d6d526,
        0xa4e28f55dbcc78b0,
        0x2a55553286590a39,
        0x2dc47e212043f68b,
    ],
    [
        0x818ce67df3bb08,
        0x39077bf1e2576b,
        0xd23b0bcf26fdd456,
        0x256a6d82275993a7,
    ],
    [
        0x14dbf2fcd6c39de1,
        0x5ddb69184942ab82,
        0x1767d433553e8d44,
        0xf8c85df551b034f,
    ],
    [
        0x86f32353e026f84c,
        0xddd0abee29343b82,
        0xf58b940b21fd9251,
        0xed6e71fa5025d70,
    ],
    [
        0xd4b29e5118ab86de,
        0x568b41bf5f9b4090,
        0xdb458bc7b18059ce,
        0x19e04bd20505ddb7,
    ],
    [
        0xaf52d4cbcea88c02,
        0x5286de716e60b0a3,
        0x4d508ff9fa524102,
        0x10b5f643ecbf138e,
    ],
    [
        0xc7fa4464b8e0f44d,
        0xe0b8537d6313e95e,
        0xf97182422dc7bc05,
        0x21ea8852d764b486,
    ],
    [
        0xeb27dfb49ab04f65,
        0xea9b98fcb4cc1f21,
        0xf746e8891fdacb02,
        0x270a5fd7486e7e0b,
    ],
    [
        0x20a199b1adac97d,
        0x68fcb2fb4e651e8b,
        0x58382af7cc3ec1b0,
        0x70f3ad69698f1b9,
    ],
    [
        0xa597882c006bd71a,
        0x551ef9f559c1d0a8,
        0xf5f5386fba24ceb,
        0x2f8375d466d028f4,
    ],
    [
        0xc7b6d1ffde7a0955,
        0xeec8e3f476fa8462,
        0xf65b7c76a5fb2c79,
        0x1dcf609a551f3770,
    ],
    [
        0xb0fa63e8782c4f49,
        0x48534388ab5d42d1,
        0xb337f2a513c9cc34,
        0xa6c8c859bf435d4,
    ],
    [
        0x1228104210e268a4,
        0x655b3afd73247f79,
        0x2691c84e84b36768,
        0x1f23ce44898ad585,
    ],
    [
        0xd3239dade1abbe9a,
        0xc52eddf7a1ece350,
        0x5c30277d426bc665,
        0x2f2920065a9d2af9,
    ],
    [
        0x70a36f47f2aff6c,
        0xfdb85e88e4550844,
        0x9627ed9978424e6a,
        0x21b6e1114fc62df0,
    ],
    [
        0x99935b823db6717e,
        0x9bc1842e83d4e5e3,
        0xe2189f767a11dc2,
        0xbe3b9ec8e2ca437,
    ],
    [
        0x45b7d695f757c594,
        0xb2a6d390594226d0,
        0xf9ebe18eb37f5259,
        0xa3769a486f411e7,
    ],
    [
        0x7026984e0085dc44,
        0x24da82e0203bb6cc,
        0x6766d00070e380c7,
        0xfbf3bfedf380cc1,
    ],
    [
        0x8d830bef3c888f69,
        0xe7837340530306,
        0xb93becc667d41b84,
        0x1ab671517c659364,
    ],
    [
        0x47413450eaa79fe9,
        0x43668c726d781e27,
        0xbfa1f603a306338f,
        0x181e1094e48f2844,
    ],
    [
        0xb2fd433efbc00cb6,
        0x799b6311847008a,
        0x4f93d06eb357bcc8,
        0x186a5be67a6d8a5e,
    ],
    [
        0xa6706ab88cc66075,
        0xa1df6bdd4a52a3db,
        0x16f12be46cefbfb,
        0x217f69e8c16f7db4,
    ],
    [
        0x841021ca9bd86897,
        0x22c8b22051a6d56d,
        0xf5cc67c845bfb18c,
        0x4df0afa22f4f3e5,
    ],
    [
        0x5dacef67b5c8145c,
        0xc33b3b9ad06681d7,
        0x3a7de3e9e0d45caf,
        0x1bbc33136e620c2f,
    ],
    [
        0x3e8327f808e15fb,
        0x9e49afcbc23c9572,
        0x7ae61434c7fecaf8,
        0x25c980ba66ee7aff,
    ],
    [
        0x61524ccb01ca9b9e,
        0xfce05819a5a0f2f4,
        0x28d68a3060c4dc3f,
        0x2c1b4cffec884b89,
    ],
    [
        0x88f7f5fed6230e90,
        0x1a311b161f04f8ee,
        0x412d5efe72759335,
        0x2b99045bdc52f7d0,
    ],
    [
        0x74410cc4ebe2e7c1,
        0xc5f3eb89c9781264,
        0xfb7297857779b613,
        0x27dc40000a0d3c73,
    ],
    [
        0x1c574f23928dbf5b,
        0x14d8a86f73ad59b9,
        0xffe26f3a19e8a96f,
        0xc5e96c0f330e5a2,
    ],
    [
        0xaf2690a49f664be7,
        0xa88a16989e37acb5,
        0x48eeb8b8e66860b0,
        0x17c44e9d385ff670,
    ],
    [
        0x155bddc3ff596b6f,
        0xa57aa664606d8a4b,
        0x5daaed84e6d0b6d1,
        0x2a4a43fd4227426,
    ],
    [
        0x125b97e5debe8526,
        0x71fe8d28ed156dec,
        0xaafd8804faba65cb,
        0x198f3617e3b5e480,
    ],
    [
        0x5aedecd0c58a1b4,
        0xf4ecdc5014f4ca99,
        0x2913aaa6ea41d52e,
        0x18b1799115d6d38d,
    ],
    [
        0x5f1869e3535ce098,
        0xa0aa7550b031d82b,
        0xcd093399c25e5e14,
        0x29ef9dcf6e2bef6f,
    ],
    [
        0x6fe70cce749aeef3,
        0x8afe7caafe4123e0,
        0x9fcd8b228f2344c9,
        0x5e600c65e33cdee,
    ],
    [
        0xa851c8caf84c9dce,
        0x79213f3a745fa85,
        0x561e877c4b7223b6,
        0xd7085ac8021cb56,
    ],
    [
        0xb85f4383ae556156,
        0x10d1afd8ac169c5,
        0xe40c294e792ade32,
        0x2522b4968520ccd9,
    ],
    [
        0x794be50448306390,
        0xcd33805ed0157321,
        0x9ddcafd32c22b4e6,
        0x119f97064698b03a,
    ],
    [
        0x30863344bed9e91f,
        0xa191078ba28f526f,
        0x220d96bb84ec07cd,
        0x24373331db348a2,
    ],
    [
        0xbbe999ed9c275bd8,
        0x2ebe3bb2c1ff920d,
        0xce1795c2cc4de1bf,
        0xdc18ee1fb665b16,
    ],
    [
        0x63267907e2dc6e73,
        0x7aef5ef2a5bad828,
        0xc360a5b6b54a39c2,
        0x2828b8c806976f,
    ],
    [
        0x88509c8adde817a6,
        0xf08e4d7764fbeda,
        0x3caff9074054604c,
        0x2835ff01d970e85f,
    ],
    [
        0x395e8f84d5f7fe8c,
        0x6552af119280cbc3,
        0x433cfe24cee250b6,
        0x24c1dbc8e4ca5479,
    ],
    [
        0xf93ba49aeb18e5d0,
        0xe4a0b527def68c97,
        0x15f69bc15e3c87bf,
        0xf6b491044f9bd36,
    ],
    [
        0xcba1c5843fef9248,
        0xe43ca0602cf98cd3,
        0x37610f6f73cc7c5,
        0x13e9643942b162ab,
    ],
    [
        0xd2e25df8630f6b7b,
        0xc55d2d12b8c7a34c,
        0xc63fe1090e68b8be,
        0x12c84fab76ec5b64,
    ],
    [
        0xadada4921aa4585,
        0x6a85fbc947dfe9dd,
        0xda29c744c80b0aa2,
        0x1cec39e596c498a,
    ],
    [
        0x1bca6e7ca22e3bd4,
        0xa66284b3e9f3dd54,
        0xab279edbaece0bf0,
        0x52670f132690604,
    ],
    [
        0xd366cac98bffbaf,
        0x3ec680e324945f78,
        0xf8e5e615b395de9f,
        0xaff7f3fd26f8df2,
    ],
    [
        0xb0f696b16cf1f098,
        0x9b6cbdda6e9e236f,
        0xa8c2de72d5fb302,
        0x2ff786d2518f5bf4,
    ],
    [
        0x2b232229c663a101,
        0x1b93fd6e28f5164c,
        0xe0be4257697a48c1,
        0x2f89bd024d6a2f9,
    ],
    [
        0xf73da0cfccbb8672,
        0xd389d107249c26e,
        0x30e4dac54b9c4e50,
        0x1d738b08ab28a9b2,
    ],
    [
        0xaf93bbae969c6038,
        0x50cd80c7820760a7,
        0xcf7460fd8dde6e02,
        0x2d862490c21d01a2,
    ],
    [
        0xb9f01904b48fa6dc,
        0xc3b7151b000e2bd5,
        0xa155d1cdb09c4a10,
        0xfa2d959db0307f5,
    ],
    [
        0xd87ddb4fc32fef17,
        0x12a63bb4a327005e,
        0x175e6fbfebdbba71,
        0x4bd0d58ae73a5ce,
    ],
];

const M: [[[u64; 4]; 5]; 5] = [
    [
        [
            0x77464b55cd95efca,
            0x68ba7a74ae0e5894,
            0xbd4dc1c2266c359d,
            0x2967c834940e37a0,
        ],
        [
            0x6907e36200995439,
            0xb9f80b5666c65169,
            0x7ba328f07ebc2640,
            0x152d921c334deb59,
        ],
        [
            0x235bc3071b88c57f,
            0x1edd9e8b512a928b,
            0x4eba9db9a285a5db,
            0x208c85cecd6e86b2,
        ],
        [
            0xd7e96fada4cc7131,
            0xe05eeb104bdd4f26,
            0xd629a31acc8b39c6,
            0x292e987009256cb4,
        ],
        [
            0x9337ce2160d27631,
            0xb7603b2e38f0d93e,
            0xba04b96b55dfec38,
            0x25c45b9bb527b189,
        ],
    ],
    [
        [
            0x9d7560eab0fe4046,
            0x35aebb7e1cbabfde,
            0x46f4c2b5ffaab98,
            0x10c9d5b18c43b9ea,
        ],
        [
            0x9de26ee0faaa6230,
            0x8b3cedd3678272c4,
            0xbf689106033676ec,
            0xa4f014b431ef663,
        ],
        [
            0x8b7a04145ef1d11a,
            0xed5ccb60d2f55df9,
            0xc0463074d5d84b7c,
            0xfc883bdcf417770,
        ],
        [
            0xc7a0f540e19091eb,
            0xd6b9fc0427f1efb4,
            0xd709082fce71505b,
            0x2c2f39bf3fb689c1,
        ],
        [
            0x570517f8d7bf3625,
            0x6f64bcced634daf,
            0x85747cad8e788981,
            0x240f49cb93d117d5,
        ],
    ],
    [
        [
            0xb866652e4f26da85,
            0xb9e2d4c767608cb5,
            0x7266982acf0812ff,
            0x1075bbdae372b70d,
        ],
        [
            0xabe2754c2279be8,
            0xf34d6acdb0ef8be1,
            0x638c985fb12509f5,
            0xce4a0756717cd0d,
        ],
        [
            0x16ef19d92023860d,
            0x97313a990cdaa693,
            0xfa536002a38deb76,
            0x157c584bf12b5fc2,
        ],
        [
            0x32ec79c4fa39b5e0,
            0x7e1d8f6dc66882f,
            0xdafcf6f32b1b7f1f,
            0xb80626e4af5efe5,
        ],
        [
            0x74572ba3822678b6,
            0x1178400143204c5f,
            0x46e8e28cd12c3a6f,
            0x10b1d99213e5666e,
        ],
    ],
    [
        [
            0x6190b23770183886,
            0x101d044302cb2858,
            0xecd03dccfbeaf617,
            0xb084598422035a5,
        ],
        [
            0x4ff66343628de773,
            0x8669e3967283e9d5,
            0xdbdb4492fd9478a1,
            0x2a172f4971297058,
        ],
        [
            0x26b36d6f81141445,
            0x46db4e5f5c0c0592,
            0x1c8ff6641950ef7f,
            0x3831bb3c0404ec0,
        ],
        [
            0x48268958c0294633,
            0xe32eaddae7cd0cfb,
            0x83f515af535c5f73,
            0xeb68faa42851083,
        ],
        [
            0x1c641486ade67a7a,
            0x4b50719a5e10222c,
            0x9f5dd44f4cc1d827,
            0x1b5b9eef181679f,
        ],
    ],
    [
        [
            0x71d451ca47c3e06f,
            0x1a4dc1da0d245f85,
            0x4812497a20f7afce,
            0x2d1c2ecb1969e4b,
        ],
        [
            0xa96b93484bd7274b,
            0xb6ffb6120bbc6f39,
            0x4f8cc3b20738a669,
            0x26d0dab233956299,
        ],
        [
            0xe693b6e9a4a622a4,
            0xd3c7b489ce3e9706,
            0x97a65d65e20440eb,
            0x1c50a5a391d3e7f,
        ],
        [
            0xde28a4428ec83e3a,
            0xc302d6eb2a211388,
            0x78e5ca7195aeb86e,
            0x1f159c9528951410,
        ],
        [
            0xfeb302a5110d9eb0,
            0xc251af52f6c4abc6,
            0xff454cd9ef575da7,
            0x1ab6f8eace913fdb,
        ],
    ],
];

#[inline]
unsafe fn set_zero() -> __m256i {
    _mm256_set_epi64x(0, 0, 0, 0)
}

#[inline]
unsafe fn set_one() -> __m256i {
    _mm256_set_epi64x(
        1011752739694698287i64,
        7381016538464732718i64,
        3962172157175319849i64,
        12436184717236109307u64 as i64,
    )
}

// cin is carry in and must be 0 or 1
#[inline]
unsafe fn add64(a: &__m256i, b: &__m256i, cin: &__m256i) -> (__m256i, __m256i) {
    let ones = _mm256_set_epi64x(1, 1, 1, 1);
    let r = _mm256_add_epi64(*a, *b);
    let m = _mm256_cmpgt_epi64(*a, r);
    let co = _mm256_and_si256(m, ones);
    let c = _mm256_add_epi64(r, *cin);
    let m = _mm256_cmpgt_epi64(r, c);
    let mo = _mm256_and_si256(m, ones);
    let co = _mm256_or_si256(co, mo);
    (c, co)
}

// bin is borrow in and must be 0 or 1
// TODO: revise
#[inline]
unsafe fn sub64(a: &__m256i, b: &__m256i, bin: &__m256i) -> (__m256i, __m256i) {
    let ones = _mm256_set_epi64x(1, 1, 1, 1);
    let r = _mm256_sub_epi64(*a, *b);
    let m = _mm256_cmpgt_epi64(r, *a);
    let co = _mm256_and_si256(m, ones);
    let c = _mm256_sub_epi64(r, *bin);
    let m = _mm256_cmpgt_epi64(c, r);
    let mo = _mm256_and_si256(m, ones);
    let co = _mm256_or_si256(co, mo);
    (c, co)
}

#[inline]
unsafe fn mul64(a: &__m256i, b: &__m256i) -> (__m256i, __m256i) {
    let mut av: [u64; 4] = [0; 4];
    let mut bv: [u64; 4] = [0; 4];
    let mut hv: [u64; 4] = [0; 4];
    let mut lv: [u64; 4] = [0; 4];
    _mm256_storeu_si256(av.as_mut_ptr().cast::<__m256i>(), *a);
    _mm256_storeu_si256(bv.as_mut_ptr().cast::<__m256i>(), *b);
    let c0 = (av[0] as u128) * (bv[0] as u128);
    let c1 = (av[1] as u128) * (bv[1] as u128);
    let c2 = (av[2] as u128) * (bv[2] as u128);
    let c3 = (av[3] as u128) * (bv[3] as u128);
    (hv[0], lv[0]) = ((c0 >> 64) as u64, c0 as u64);
    (hv[1], lv[1]) = ((c1 >> 64) as u64, c1 as u64);
    (hv[2], lv[2]) = ((c2 >> 64) as u64, c2 as u64);
    (hv[3], lv[3]) = ((c3 >> 64) as u64, c3 as u64);
    let h = _mm256_loadu_si256(hv.as_mut_ptr().cast::<__m256i>());
    let l = _mm256_loadu_si256(lv.as_mut_ptr().cast::<__m256i>());
    (h, l)
}

// madd0 hi = a*b + c (discards lo bits)
#[inline]
unsafe fn madd0(a: &__m256i, b: &__m256i, c: &__m256i) -> __m256i {
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let (hi, lo) = mul64(a, b);
    let (_, cr) = add64(&lo, c, &zeros);
    let (hi, _) = add64(&hi, &zeros, &cr);
    hi
}

// madd1 hi, lo = a*b + c
#[inline]
unsafe fn madd1(a: &__m256i, b: &__m256i, c: &__m256i) -> (__m256i, __m256i) {
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let (hi, lo) = mul64(a, b);
    let (lo, cr) = add64(&lo, c, &zeros);
    let (hi, _) = add64(&hi, &zeros, &cr);
    (hi, lo)
}

// madd2 hi, lo = a*b + c + d
#[inline]
unsafe fn madd2(a: &__m256i, b: &__m256i, c: &__m256i, d: &__m256i) -> (__m256i, __m256i) {
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let (hi, lo) = mul64(a, b);
    let (c, cr) = add64(c, d, &zeros);
    let (hi, _) = add64(&hi, &zeros, &cr);
    let (lo, cr) = add64(&lo, &c, &zeros);
    let (hi, _) = add64(&hi, &zeros, &cr);
    (hi, lo)
}

#[inline]
unsafe fn madd3(
    a: &__m256i,
    b: &__m256i,
    c: &__m256i,
    d: &__m256i,
    e: &__m256i,
) -> (__m256i, __m256i) {
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let (hi, lo) = mul64(a, b);
    let (c, cr) = add64(c, d, &zeros);
    let (hi, _) = add64(&hi, &zeros, &cr);
    let (lo, cr) = add64(&lo, &c, &zeros);
    let (hi, _) = add64(&hi, e, &cr);
    (hi, lo)
}

#[inline]
pub unsafe fn _mm256_mullo_epi64(a: __m256i, b: __m256i) -> __m256i {
    let mut av: [u64; 4] = [0; 4];
    let mut bv: [u64; 4] = [0; 4];
    _mm256_storeu_si256(av.as_mut_ptr().cast::<__m256i>(), a);
    _mm256_storeu_si256(bv.as_mut_ptr().cast::<__m256i>(), b);
    /*
    asm!(
        "mov rax, [rdi]",
        "mov rdx, [rsi]",
        "mul rdx",
        "mov [rdi], rax",
        "mov rax, [rdi+8]",
        "mov rdx, [rsi+8]",
        "mul rdx",
        "mov [rdi+8], rax",
        "mov rax, [rdi+16]",
        "mov rdx, [rsi+16]",
        "mul rdx",
        "mov [rdi+16], rax",
        "mov rax, [rdi+24]",
        "mov rdx, [rsi+24]",
        "mul rdx",
        "mov [rdi+24], rax",
        in("rdi") &av,
        in("rsi") &bv,
    );
    */
    for i in 0..4 {
        av[i] = ((av[i] as u128) * (bv[i] as u128)) as u64;
    }
    _mm256_load_si256(av.as_ptr().cast::<__m256i>())
}

#[inline]
unsafe fn _mulGeneric(x: [__m256i; 4], y: [__m256i; 4]) -> [__m256i; 4] {
    let mut z: [__m256i; 4] = [_mm256_set_epi64x(0, 0, 0, 0); 4];
    let mut t: [__m256i; 4] = [_mm256_set_epi64x(0, 0, 0, 0); 4];
    let mut c: [__m256i; 3] = [_mm256_set_epi64x(0, 0, 0, 0); 3];

    let ct0 = _mm256_set_epi64x(
        4891460686036598785i64,
        4891460686036598785i64,
        4891460686036598785i64,
        4891460686036598785i64,
    );
    let ct1 = _mm256_set_epi64x(
        2896914383306846353i64,
        2896914383306846353i64,
        2896914383306846353i64,
        2896914383306846353i64,
    );
    let ct2 = _mm256_set_epi64x(
        13281191951274694749u64 as i64,
        13281191951274694749u64 as i64,
        13281191951274694749u64 as i64,
        13281191951274694749u64 as i64,
    );
    let ct3 = _mm256_set_epi64x(
        3486998266802970665i64,
        3486998266802970665i64,
        3486998266802970665i64,
        3486998266802970665i64,
    );
    let ct4 = _mm256_set_epi64x(
        14042775128853446655u64 as i64,
        14042775128853446655u64 as i64,
        14042775128853446655u64 as i64,
        14042775128853446655u64 as i64,
    );
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);

    // round 0
    let mut v = x[0];
    (c[1], c[0]) = mul64(&v, &y[0]);
    let m = _mm256_mullo_epi64(c[0], ct4);
    c[2] = madd0(&m, &ct0, &c[0]);
    (c[1], c[0]) = madd1(&v, &y[1], &c[1]);
    (c[2], t[0]) = madd2(&m, &ct1, &c[2], &c[0]);
    (c[1], c[0]) = madd1(&v, &y[2], &c[1]);
    (c[2], t[1]) = madd2(&m, &ct2, &c[2], &c[0]);
    (c[1], c[0]) = madd1(&v, &y[3], &c[1]);
    (t[3], t[2]) = madd3(&m, &ct3, &c[0], &c[2], &c[1]);

    // round 1
    v = x[1];
    (c[1], c[0]) = madd1(&v, &y[0], &t[0]);
    let m = _mm256_mullo_epi64(c[0], ct4);
    c[2] = madd0(&m, &ct0, &c[0]);
    (c[1], c[0]) = madd2(&v, &y[1], &c[1], &t[1]);
    (c[2], t[0]) = madd2(&m, &ct1, &c[2], &c[0]);
    (c[1], c[0]) = madd2(&v, &y[2], &c[1], &t[2]);
    (c[2], t[1]) = madd2(&m, &ct2, &c[2], &c[0]);
    (c[1], c[0]) = madd2(&v, &y[3], &c[1], &t[3]);
    (t[3], t[2]) = madd3(&m, &ct3, &c[0], &c[2], &c[1]);

    // round 2
    v = x[2];
    (c[1], c[2]) = madd1(&v, &y[0], &t[0]);
    let m = _mm256_mullo_epi64(c[0], ct4);
    c[2] = madd0(&m, &ct0, &c[0]);
    (c[1], c[2]) = madd2(&v, &y[1], &c[1], &t[1]);
    (c[2], t[0]) = madd2(&m, &ct1, &c[2], &c[0]);
    (c[1], c[0]) = madd2(&v, &y[2], &c[1], &t[2]);
    (c[2], t[1]) = madd2(&m, &ct2, &c[2], &c[0]);
    (c[1], c[0]) = madd2(&v, &y[3], &c[1], &t[3]);
    (t[3], t[2]) = madd3(&m, &ct3, &c[0], &c[2], &c[1]);

    // round 3
    v = x[3];
    (c[1], c[0]) = madd1(&v, &y[0], &t[0]);
    let m = _mm256_mullo_epi64(c[0], ct4);
    c[2] = madd0(&m, &ct0, &c[0]);
    (c[1], c[2]) = madd2(&v, &y[1], &c[1], &t[1]);
    (c[2], z[0]) = madd2(&m, &ct1, &c[2], &c[0]);
    (c[1], c[0]) = madd2(&v, &y[2], &c[1], &t[2]);
    (c[2], z[1]) = madd2(&m, &ct2, &c[2], &c[0]);
    (c[1], c[0]) = madd2(&v, &y[3], &c[1], &t[3]);
    (z[3], z[2]) = madd3(&m, &ct3, &c[0], &c[2], &c[1]);

    // if z > q --> z -= q
    let cmp0 = _mm256_cmpgt_epi64(ct0, z[0]);
    let cmp1 = _mm256_cmpeq_epi64(ct1, z[1]);
    let cmp0 = _mm256_and_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpgt_epi64(ct1, z[1]);
    let cmp0 = _mm256_or_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpeq_epi64(ct2, z[2]);
    let cmp0 = _mm256_and_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpgt_epi64(ct2, z[2]);
    let cmp0 = _mm256_or_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpeq_epi64(ct3, z[3]);
    let cmp0 = _mm256_and_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpgt_epi64(ct3, z[3]);
    let cmp0 = _mm256_or_si256(cmp0, cmp1);
    let st0 = _mm256_andnot_si256(cmp0, ct0);
    let st1 = _mm256_andnot_si256(cmp0, ct1);
    let st2 = _mm256_andnot_si256(cmp0, ct2);
    let st3 = _mm256_andnot_si256(cmp0, ct3);
    let mut b = zeros;
    (z[0], b) = sub64(&z[0], &st0, &zeros);
    (z[1], b) = sub64(&z[1], &st1, &b);
    (z[2], b) = sub64(&z[2], &st2, &b);
    (z[3], _) = sub64(&z[3], &st3, &b);

    z
}

#[inline]
fn exp5state(state: &mut [__m256i; 5]) {
    let s: [__m256i; 4] = [state[0], state[1], state[2], state[3]];
    unsafe {
        let s2 = _mulGeneric(s, s);
        let s4 = _mulGeneric(s2, s2);
        let s5 = _mulGeneric(s, s4);
        state[0] = s5[0];
        state[1] = s5[1];
        state[2] = s5[2];
        state[3] = s5[3];
    }
    let s: [__m256i; 4] = [state[4], state[4], state[4], state[4]];
    unsafe {
        let s2 = _mulGeneric(s, s);
        let s4 = _mulGeneric(s2, s2);
        let s5 = _mulGeneric(s, s4);
        state[4] = s5[0];
    }
}

#[inline]
unsafe fn _addGeneric(x: [__m256i; 4], y: [__m256i; 4]) -> [__m256i; 4] {
    let mut z: [__m256i; 4] = [_mm256_set_epi64x(0, 0, 0, 0); 4];
    let mut cr = _mm256_set_epi64x(0, 0, 0, 0);

    (z[0], cr) = add64(&x[0], &y[0], &cr);
    (z[1], cr) = add64(&x[1], &y[1], &cr);
    (z[2], cr) = add64(&x[2], &y[2], &cr);
    (z[3], _) = add64(&x[3], &y[3], &cr);

    // if z > q --> z -= q
    // note: this is NOT constant time
    let ct0 = _mm256_set_epi64x(
        4891460686036598785i64,
        4891460686036598785i64,
        4891460686036598785i64,
        4891460686036598785i64,
    );
    let ct1 = _mm256_set_epi64x(
        2896914383306846353i64,
        2896914383306846353i64,
        2896914383306846353i64,
        2896914383306846353i64,
    );
    let ct2 = _mm256_set_epi64x(
        13281191951274694749u64 as i64,
        13281191951274694749u64 as i64,
        13281191951274694749u64 as i64,
        13281191951274694749u64 as i64,
    );
    let ct3 = _mm256_set_epi64x(
        3486998266802970665i64,
        3486998266802970665i64,
        3486998266802970665i64,
        3486998266802970665i64,
    );
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);

    // if z > q --> z -= q
    let cmp0 = _mm256_cmpgt_epi64(ct0, z[0]);
    let cmp1 = _mm256_cmpeq_epi64(ct1, z[1]);
    let cmp0 = _mm256_and_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpgt_epi64(ct1, z[1]);
    let cmp0 = _mm256_or_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpeq_epi64(ct2, z[2]);
    let cmp0 = _mm256_and_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpgt_epi64(ct2, z[2]);
    let cmp0 = _mm256_or_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpeq_epi64(ct3, z[3]);
    let cmp0 = _mm256_and_si256(cmp0, cmp1);
    let cmp1 = _mm256_cmpgt_epi64(ct3, z[3]);
    let cmp0 = _mm256_or_si256(cmp0, cmp1);
    let st0 = _mm256_andnot_si256(cmp0, ct0);
    let st1 = _mm256_andnot_si256(cmp0, ct1);
    let st2 = _mm256_andnot_si256(cmp0, ct2);
    let st3 = _mm256_andnot_si256(cmp0, ct3);
    let mut b = zeros;
    (z[0], b) = sub64(&z[0], &st0, &zeros);
    (z[1], b) = sub64(&z[1], &st1, &b);
    (z[2], b) = sub64(&z[2], &st2, &b);
    (z[3], _) = sub64(&z[3], &st3, &b);

    z
}

#[inline]
unsafe fn to_mont(a: [__m256i; 4]) -> [__m256i; 4] {
    /*
    let rSquare = _mm256_set_epi64x(
        1997599621687373223u64 as i64,
        6052339484930628067u64 as i64,
        10108755138030829701u64 as i64,
        150537098327114917u64 as i64,
    );
    */
    let rSquare = _mm256_set_epi64x(
        150537098327114917u64 as i64,
        10108755138030829701u64 as i64,
        6052339484930628067u64 as i64,
        1997599621687373223u64 as i64,
    );
    let r: [__m256i; 4] = [rSquare, rSquare, rSquare, rSquare];
    _mulGeneric(a, r)
}

#[inline]
unsafe fn ark(state: &mut [__m256i; 5], c: [[u64; 4]; 100], it: usize) {
    let mut cc: [__m256i; 4] = [
        _mm256_set_epi64x(
            c[it][0] as i64,
            c[it][1] as i64,
            c[it][2] as i64,
            c[it][3] as i64,
        ),
        _mm256_set_epi64x(
            c[it + 1][0] as i64,
            c[it + 1][1] as i64,
            c[it + 1][2] as i64,
            c[it + 1][3] as i64,
        ),
        _mm256_set_epi64x(
            c[it + 2][0] as i64,
            c[it + 2][1] as i64,
            c[it + 2][2] as i64,
            c[it + 2][3] as i64,
        ),
        _mm256_set_epi64x(
            c[it + 3][0] as i64,
            c[it + 3][1] as i64,
            c[it + 3][2] as i64,
            c[it + 3][3] as i64,
        ),
    ];
    // first 4 elems
    let mut ss: [__m256i; 4] = [state[0], state[1], state[2], state[3]];
    ss = _addGeneric(ss, cc);
    state[0] = ss[0];
    state[1] = ss[1];
    state[2] = ss[2];
    state[3] = ss[3];

    // 5th elem
    ss[0] = state[4];
    cc[0] = _mm256_set_epi64x(
        c[it + 4][0] as i64,
        c[it + 4][1] as i64,
        c[it + 4][2] as i64,
        c[it + 4][3] as i64,
    );
    ss = _addGeneric(ss, cc);
    state[4] = ss[0];
}

#[inline]
unsafe fn mix(state: &mut [__m256i; 5], m: [[[u64; 4]; 5]; 5]) {
    let mut newState: [ElementBN128; 5] = [ElementBN128::Zero(); 5];
    let mut mul = ElementBN128::Zero();
    for i in 0..5 {
        newState[i].SetUint64(0);
        for j in 0..5 {
            let mm = ElementBN128::New(m[j][i]);
            // mul.Mul(mm, state[j]);
            newState[i].Add(newState[i], mul);
        }
    }
    //for i in 0..5 {
    //    state[i] = newState[i];
    //}
}

fn print_state4(state: &[__m256i; 4]) {
    let mut a: [u64; 4] = [0; 4];
    println!("State4:");
    unsafe {
        _mm256_storeu_si256(a.as_mut_ptr().cast::<__m256i>(), state[0]);
        println!("{:?}", a);
        _mm256_storeu_si256(a.as_mut_ptr().cast::<__m256i>(), state[1]);
        println!("{:?}", a);
        _mm256_storeu_si256(a.as_mut_ptr().cast::<__m256i>(), state[2]);
        println!("{:?}", a);
        _mm256_storeu_si256(a.as_mut_ptr().cast::<__m256i>(), state[3]);
        println!("{:?}", a);
    }
}

pub fn permute_bn128_avx(input: [u64; 12]) -> [u64; 12] {
    let st64: Vec<i64> = input
        .into_iter()
        .map(|x| x as i64)
        .collect();

    const NROUNDSF: usize = 8;

    unsafe {
        // permute
        // load states
        let mut inp: [__m256i; 4] = [
            _mm256_set_epi64x(st64[11], st64[8], st64[5], st64[2]),
            _mm256_set_epi64x(st64[10], st64[7], st64[4], st64[1]),
            _mm256_set_epi64x(st64[9], st64[6], st64[3], st64[0]),
            _mm256_set_epi64x(0i64, 0i64, 0i64, 0i64),
        ];
        print_state4(&inp);
        // to mont
        let inp = to_mont(inp);
        print_state4(&inp);
    }
    // save state
    let out: [u64; 12] = [0; 12];
    out
}
