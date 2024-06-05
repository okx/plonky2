#[cfg(feature = "cuda")]
pub fn init_cuda() {
    use cryptography_cuda::{get_number_of_gpus_rs, init_coset_rs, init_twiddle_factors_rs};
    use plonky2_field::goldilocks_field::GoldilocksField;
    use plonky2_field::types::{Field, PrimeField64};

    let num_of_gpus = get_number_of_gpus_rs();
    println!("num of gpus: {:?}", num_of_gpus);
    std::env::set_var("NUM_OF_GPUS", num_of_gpus.to_string());

    let log_ns: Vec<usize> = (2..22).collect();

    let mut device_id = 0;
    while device_id < num_of_gpus {
        init_coset_rs(
            device_id,
            24,
            GoldilocksField::coset_shift().to_canonical_u64(),
        );
        for log_n in &log_ns {
            // println!("{:?}", log_n);
            init_twiddle_factors_rs(device_id, *log_n);
        }
        device_id = device_id + 1;
    }
}
