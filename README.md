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

# Rust

To use a nightly toolchain for Plonky2 by default, you can run

```
rustup override set nightly
```
