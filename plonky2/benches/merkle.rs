mod allocator;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::hash::hash_types::RichField;
use plonky2::hash::keccak::KeccakHash;
use plonky2::hash::merkle_tree::MerkleTree;
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::hash::poseidon2::Poseidon2Hash;
use plonky2::hash::poseidon_bn128::PoseidonBN128Hash;
use plonky2::plonk::config::Hasher;
use tynm::type_name;

const ELEMS_PER_LEAF: usize = 135;

pub(crate) fn bench_merkle_tree<F: RichField, H: Hasher<F>>(c: &mut Criterion) {
    let mut group = c.benchmark_group(&format!(
        "merkle-tree<{}, {}>",
        type_name::<F>(),
        type_name::<H>()
    ));
    group.sample_size(10);

    for size_log in [13, 14, 15] {
        let size = 1 << size_log;
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            // let leaves = vec![F::rand_vec(ELEMS_PER_LEAF); size];
            // b.iter(|| MerkleTree::<F, H>::new_from_2d(leaves.clone(), 0));
            let mut leaves = Vec::with_capacity(size * ELEMS_PER_LEAF);
            for _ in 0..size {
                leaves.append(&mut F::rand_vec(ELEMS_PER_LEAF));
            }
            b.iter(|| MerkleTree::<F, H>::new_from_1d(leaves.clone(), ELEMS_PER_LEAF, 2));
        });
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_merkle_tree::<GoldilocksField, PoseidonHash>(c);
    bench_merkle_tree::<GoldilocksField, Poseidon2Hash>(c);
    bench_merkle_tree::<GoldilocksField, KeccakHash<25>>(c);
    bench_merkle_tree::<GoldilocksField, PoseidonBN128Hash>(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
