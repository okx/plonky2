mod allocator;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
#[cfg(feature = "cuda")]
use cryptography_cuda::init_cuda_degree_rs;
use plonky2::field::extension::Extendable;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::polynomial::PolynomialCoeffs;
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::timing::TimingTree;
use tynm::type_name;

pub(crate) fn bench_batch_lde<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    c: &mut Criterion,
) {
    const RATE_BITS: usize = 3;

    let mut group = c.benchmark_group(&format!("lde<{}>", type_name::<F>()));

    #[cfg(feature = "cuda")]
    init_cuda_degree_rs(16);

    for size_log in [13, 14, 15] {
        let orig_size = 1 << (size_log - RATE_BITS);
        let lde_size = 1 << size_log;
        let batch_size = 1 << 4;

        group.bench_with_input(BenchmarkId::from_parameter(lde_size), &lde_size, |b, _| {
            let polynomials: Vec<PolynomialCoeffs<F>> = (0..batch_size)
                .into_iter()
                .map(|_i| PolynomialCoeffs::new(F::rand_vec(orig_size)))
                .collect();
            let mut timing = TimingTree::new("lde", log::Level::Error);
            b.iter(|| {
                PolynomialBatch::<F, C, D>::from_coeffs(
                    polynomials.clone(),
                    RATE_BITS,
                    false,
                    1,
                    &mut timing,
                    None,
                )
            });
        });
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_batch_lde::<GoldilocksField, PoseidonGoldilocksConfig, 2>(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
