## Benchmark
Machine: Apple M1 Pro 10 cores + 32GB

### Secp256k1
Gate Number: 98039
Prove Time: 9s
```
    Finished release [optimized] target(s) in 4.30s
    Running `target/release/examples/ecdsa_secp256k1`
[2024-03-06T10:42:35Z DEBUG plonky2::util::context_tree] 98039 gates to root
[2024-03-06T10:42:35Z DEBUG plonky2::plonk::circuit_builder] Total gate counts:
[2024-03-06T10:42:35Z DEBUG plonky2::plonk::circuit_builder] - 16 instances of BaseSumGate { num_limbs: 16 } + Base: 4
[2024-03-06T10:42:35Z DEBUG plonky2::plonk::circuit_builder] - 1264 instances of U32AddManyGate { num_addends: 13, num_ops: 4, _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }
[2024-03-06T10:42:35Z DEBUG plonky2::plonk::circuit_builder] - 3121 instances of U32RangeCheckGate { num_input_limbs: 8, _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }
[2024-03-06T10:42:35Z DEBUG plonky2::plonk::circuit_builder] - 3 instances of U32AddManyGate { num_addends: 0, num_ops: 6, _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }
[2024-03-06T10:42:35Z DEBUG plonky2::plonk::circuit_builder] - 1 instances of U32RangeCheckGate { num_input_limbs: 0, _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }
[2024-03-06T10:42:35Z DEBUG plonky2::plonk::circuit_builder] - 1012 instances of U32AddManyGate { num_addends: 5, num_ops: 5, _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }
[2024-03-06T10:42:35Z DEBUG plonky2::plonk::circuit_builder] - 1264 instances of U32AddManyGate { num_addends: 11, num_ops: 4, _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }
[2024-03-06T10:42:35Z DEBUG plonky2::plonk::circuit_builder] - 1737 instances of U32SubtractionGate { num_ops: 6, _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }
[2024-03-06T10:42:36Z DEBUG plonky2::plonk::circuit_builder] - 55530 instances of U32ArithmeticGate { num_ops: 3, _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }
[2024-03-06T10:42:36Z DEBUG plonky2::plonk::circuit_builder] - 10546 instances of U32AddManyGate { num_addends: 3, num_ops: 5, _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }
[2024-03-06T10:42:36Z DEBUG plonky2::plonk::circuit_builder] - 1907 instances of ArithmeticGate { num_ops: 20 }
[2024-03-06T10:42:36Z DEBUG plonky2::plonk::circuit_builder] - 16912 instances of ComparisonGate { num_bits: 32, num_chunks: 16, _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }<D=2>
[2024-03-06T10:42:36Z DEBUG plonky2::plonk::circuit_builder] - 1686 instances of U32AddManyGate { num_addends: 15, num_ops: 3, _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }
[2024-03-06T10:42:36Z DEBUG plonky2::plonk::circuit_builder] - 512 instances of RandomAccessGate { bits: 4, num_copies: 4, num_extra_constants: 2, _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }<D=2>
[2024-03-06T10:42:36Z DEBUG plonky2::plonk::circuit_builder] - 1264 instances of U32AddManyGate { num_addends: 7, num_ops: 4, _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }
[2024-03-06T10:42:36Z DEBUG plonky2::plonk::circuit_builder] - 1264 instances of U32AddManyGate { num_addends: 9, num_ops: 4, _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }
[2024-03-06T10:42:36Z DEBUG plonky2::plonk::circuit_builder] Degree before blinding & padding: 105434
[2024-03-06T10:42:36Z DEBUG plonky2::plonk::circuit_builder] Degree after blinding & padding: 131072
[2024-03-06T10:42:43Z DEBUG plonky2::plonk::circuit_builder] Building circuit took 7.3354907s
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] 9.0794s to prove
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | 0.5859s to run 339849 generators
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | 0.0920s to compute full witness
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | 0.0037s to compute wire polynomials
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | 3.7082s to compute wires commitment
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | | 0.0545s to IFFT
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | | 0.4327s to FFT + blinding
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | | 0.4273s to transpose LDEs
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | | 2.7703s to build Merkle tree
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | 0.1239s to compute partial products
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | 0.7268s to commit to partial products and Z's
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | | 0.0083s to IFFT
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | | 0.0734s to FFT + blinding
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | | 0.0724s to transpose LDEs
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | | 0.5669s to build Merkle tree
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | 2.2916s to compute quotient polys
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | 0.0006s to split up quotient polys
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | 0.5671s to commit to quotient polys
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | | 0.0544s to FFT + blinding
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | | 0.0598s to transpose LDEs
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | | 0.4480s to build Merkle tree
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | 0.0426s to construct the opening set
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | 0.2691s to compute opening proofs
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | | 0.1275s to reduce batch of 258 polynomials
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | | 0.0009s to reduce batch of 2 polynomials
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | | 0.0681s to perform final FFT 1048576
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | | 0.0604s to fold codewords in the commitment phase
[2024-03-06T10:42:53Z DEBUG plonky2::util::timing] | | 0.0038s to find proof-of-work witness
```

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.


### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
