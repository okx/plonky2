# Description

This repo is a fork of https://github.com/0xPolygonZero/plonky2. To boost speed, several optimizations were implemented:

# Optimizations
- Precompute FFT twiddle factors.
- CUDA implementation of Goldilocks Field NTT (feature `cuda`).
- CUDA implementation of Poseidon (Goldilocks) and Poseidon (BN 128) (feature `cuda`).
- Fixed the AVX implementation for Poseidon (Goldilocks) (target CPU must support AVX2).
- CUDA implementation of Merkle Tree building (feature `cuda`).
- Change Merkle Tree structure from recursive to iterative (1-dimensional vector).

# Dependencies

```
git submodule update --init --recursive
```

## Benchmarking Merkle Tree building with Poseison hash

Set the latest Rust nightly:
```
rustup update
rustup override set nightly-x86_64-unknown-linux-gnu
```

CPU, no AVX: ``cargo bench merkle``

CPU with AVX2: ``RUSTFLAGS="-C target-feature=+avx2" cargo bench merkle``

CPU with AVX512: ``RUSTFLAGS="-C target-feature=+avx512dq" cargo bench merkle``

GPU (CUDA): ``cargo bench merkle --features=cuda``

### Results

The results in the table below represent the build time of a Merkle Tree with the indicated number of leaves (first row) using the hashing method indicated in the first column. The systems used for benchmarking are:

- first three columns: AMD Ryzen Threadripper PRO 5975WX 32-Cores (only AVX2) +
NVIDIA RTX 4090

- last three columns: AMD Ryzen 9 7950X 16-Core (AVX2 and AVX512DQ)


| Number of MT Leaves | 2^13  | 2^14  | 2^15  |   | 2^13  | 2^14  | 2^15 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Poseidon (no AVX)                     | 12.4  | 23.4  | 46.6  |                                       | 12.8  | 25.2  | 50.3  |
| Poseidon (AVX)                        | 11.4  | 21.3  | 39.2  |                                       | 10.3  | 20.3  | 40.2  |
| Poseidon (AVX512)                     |  -     |  -     | -   |                                       | 12.3  | 24.1  | 47.8  |
| Poseidon (GPU)                        | 8     | 14.3  | 26.5  |                                       |    -   | -      |  -     |
| Poseidon BN 128 (no AVX)              | 111.9 | 223   | 446.3 |                                       | 176.9 | 351   | 699.1 |
| Poseidon BN 128 (AVX)                 | 146.8 | 291.7 | 581.8 |                                       | 220.1 | 433.5 | 858.8 |
| Poseidon BN 128 (AVX512)              |    -   |    -   |   -    |                                       | WIP   | WIP   | WIP   |
| Poseidon BN 128 (GPU)                 | 37.5  | 57.6  | 92.9  |                                        | - | - | - |

## Running

To see recursion performance, one can run this bench, which generates a chain of three recursion proofs:

```sh
RUSTFLAGS=-Ctarget-cpu=native cargo run --release --example bench_recursion -- -vv
```

## Jemalloc

Plonky2 prefers the [Jemalloc](http://jemalloc.net) memory allocator due to its superior performance. To use it, include `jemallocator = "0.5.0"` in your `Cargo.toml` and add the following lines
to your `main.rs`:

```rust
use jemallocator::Jemalloc;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;
```

Jemalloc is known to cause crashes when a binary compiled for x86 is run on an Apple silicon-based Mac under [Rosetta 2](https://support.apple.com/en-us/HT211861). If you are experiencing crashes on your Apple silicon Mac, run `rustc --print target-libdir`. The output should contain `aarch64-apple-darwin`. If the output contains `x86_64-apple-darwin`, then you are running the Rust toolchain for x86; we recommend switching to the native ARM version.

## Contributing guidelines

See [CONTRIBUTING.md](./CONTRIBUTING.md).

## Licenses

All crates of this monorepo are licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.


## Security

This code has not yet been audited, and should not be used in any production systems.

While Plonky2 is configurable, its defaults generally target 100 bits of security. The default FRI configuration targets 100 bits of *conjectured* security based on the conjecture in [ethSTARK](https://eprint.iacr.org/2021/582).

Plonky2's default hash function is Poseidon, configured with 8 full rounds, 22 partial rounds, a width of 12 field elements (each ~64 bits), and an S-box of `x^7`. [BBLP22](https://tosc.iacr.org/index.php/ToSC/article/view/9850) suggests that this configuration may have around 95 bits of security, falling a bit short of our 100 bit target.


## Links

#### Actively maintained

- [Polygon Zero's zkEVM](https://github.com/0xPolygonZero/zk_evm), an efficient Type 1 zkEVM built on top of Starky and plonky2

#### No longer maintained

- [System Zero](https://github.com/0xPolygonZero/system-zero), a zkVM built on top of Starky
- [Waksman](https://github.com/0xPolygonZero/plonky2-waksman), Plonky2 gadgets for permutation checking using Waksman networks
- [Insertion](https://github.com/0xPolygonZero/plonky2-insertion), Plonky2 gadgets for insertion into a list
- [u32](https://github.com/0xPolygonZero/plonky2-u32), Plonky2 gadgets for u32 arithmetic
- [ECDSA](https://github.com/0xPolygonZero/plonky2-ecdsa), Plonky2 gadgets for the ECDSA algorithm
