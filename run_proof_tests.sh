#!/bin/sh -e
cargo test --package plonky2 --features=cuda,batch --release --test fibonacci_test -- test_fibonacci_proof --exact --nocapture
cargo test --package plonky2 --features=cuda,batch --release --test range_check_test -- test_range_check_proof --exact --nocapture
cargo test --package plonky2 --features=cuda,batch --release --test factorial_test -- test_factorial_proof --exact --nocapture
