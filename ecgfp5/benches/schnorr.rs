use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use plonky2_ecgfp5::gadgets::schnorr::{schnorr_keygen, schnorr_sign, F};
use plonky2_field::types::Field;
use rand::thread_rng;

pub fn bench_schnorr(c: &mut Criterion) {
    let mut group = c.benchmark_group("schnorr");
    group.bench_function("keygen", |b| {
        b.iter(|| {
            let mut rng = thread_rng();
            let _ = schnorr_keygen(&mut rng);
        })
    });
    group.bench_function("sign", |b| {
        b.iter_batched(
            || {
                let mut rng = thread_rng();
                let (_pk, sk) = schnorr_keygen(&mut rng);
                let message = [F::TWO; 32];
                (sk, message)
            },
            |(sk, message)| {
                let mut rng = thread_rng();
                let _ = schnorr_sign(&message, &sk, &mut rng);
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_schnorr);
criterion_main!(benches);
