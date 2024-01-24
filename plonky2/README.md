# Plonky2

Plonky2 is a SNARK implementation based on techniques from PLONK and FRI. It is the successor of [Plonky](https://github.com/0xPolygonZero/plonky), which was based on PLONK and Halo.

Plonky2 is built for speed, and features a highly efficient recursive circuit. On a Macbook Pro, recursive proofs can be generated in about 170 ms.

# Rust

To use a nightly toolchain for Plonky2 by default, you can run

```
rustup override set nightly
```

# Plonky2 on GPU

## Poseidon Hash on GPU (CUDA)

Build the shared library

```
cd cryptography_cuda/cuda/merkle
make lib
make libgpu
```

Run tests (in plonky2 folder)

```
export LD_LIBRARY_PATH=<path-to cryptography_cuda/cuda/merkle>
# CPU-only
cargo test -- --nocapture merkle_trees
# GPU
cargo test --features=cuda -- --nocapture merkle_trees
```

Run benchmarks
```
# CPU
cargo bench merkle
# GPU
cargo bench --features=cuda merkle
```

Run microbenchmarks

```
cd cryptography_cuda/cuda/merkle
./run-benchmark.sh
```

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.


### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
