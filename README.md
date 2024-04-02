# description
this repo is a fork of https://github.com/0xPolygonZero/plonky2. several optimizations were implemented to boost the computation speed.

# optimizations
- precompute of fft twiddle factors
- cuda implementation of Goldilocks Field NTT (feature `cuda`)

# dependencies
```
git submodule update --init --recursive
```

# run examples
- cuda NTT
```
cargo run --release -p plonky2_field --features=cuda --example fft
```

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
