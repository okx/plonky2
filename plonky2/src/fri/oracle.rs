#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};

#[cfg(feature = "cuda")]
use cryptography_cuda::{
    device::memory::HostOrDeviceSlice, lde_batch, lde_batch_multi_gpu, transpose_rev_batch,
    types::*,
};

use itertools::Itertools;
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::packed::PackedField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::FriProof;
use crate::fri::prover::fri_proof;
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo};
use crate::fri::FriParams;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::MerkleTree;
use crate::iop::challenger::Challenger;
use crate::plonk::config::GenericConfig;
use crate::timed;
use crate::util::reducing::ReducingFactor;
use crate::util::timing::TimingTree;
use crate::util::{log2_strict, reverse_bits, reverse_index_bits_in_place, transpose};

/// Four (~64 bit) field elements gives ~128 bit security.
pub const SALT_SIZE: usize = 4;

/// Represents a FRI oracle, i.e. a batch of polynomials which have been Merklized.
#[derive(Eq, PartialEq, Debug)]
pub struct PolynomialBatch<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub polynomials: Vec<PolynomialCoeffs<F>>,
    pub merkle_tree: MerkleTree<F, C::Hasher>,
    pub degree_log: usize,
    pub rate_bits: usize,
    pub blinding: bool,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Default
    for PolynomialBatch<F, C, D>
{
    fn default() -> Self {
        PolynomialBatch {
            polynomials: Vec::new(),
            merkle_tree: MerkleTree::default(),
            degree_log: 0,
            rate_bits: 0,
            blinding: false,
        }
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PolynomialBatch<F, C, D>
{
    /// Creates a list polynomial commitment for the polynomials interpolating the values in `values`.
    pub fn from_values(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        // #[cfg(any(not(feature = "cuda"), not(feature = "batch")))]
        let coeffs = timed!(
            timing,
            "IFFT",
            values.into_par_iter().map(|v| v.ifft()).collect::<Vec<_>>()
        );

        // #[cfg(all(feature = "cuda", feature = "batch"))]
        // let degree = values[0].len();
        // #[cfg(all(feature = "cuda", feature = "batch"))]
        // let log_n = log2_strict(degree);

        // #[cfg(all(feature = "cuda", feature = "batch"))]
        // let num_gpus: usize = std::env::var("NUM_OF_GPUS")
        //     .expect("NUM_OF_GPUS should be set")
        //     .parse()
        //     .unwrap();
        // // let num_gpus = 1;
        // #[cfg(all(feature = "cuda", feature = "batch"))]
        // let total_num_of_fft = values.len();
        // #[cfg(all(feature = "cuda", feature = "batch"))]
        // let per_device_batch = total_num_of_fft.div_ceil(num_gpus);

        // #[cfg(all(feature = "cuda", feature = "batch"))]
        // let chunk_size = total_num_of_fft.div_ceil(num_gpus);
        // #[cfg(all(feature = "cuda", feature = "batch"))]
        // println!(
        //     "invoking intt_batch, total_nums: {:?}, log_n: {:?}, num_gpus: {:?}",
        //     total_num_of_fft, log_n, num_gpus
        // );

        // #[cfg(all(feature = "cuda", feature = "batch"))]
        // let coeffs = timed!(
        //     timing,
        //     "IFFT",
        //     values
        //         .par_chunks(chunk_size)
        //         .enumerate()
        //         .flat_map(|(id, poly_chunk)| {
        //             let mut polys_values: Vec<F> =
        //                 poly_chunk.iter().flat_map(|p| p.values.clone()).collect();
        //             let mut ntt_cfg = NTTConfig::default();
        //             ntt_cfg.batches = per_device_batch as u32;

        //             intt_batch(id, polys_values.as_mut_ptr(), log_n, ntt_cfg);
        //             polys_values
        //                 .chunks(1 << log_n)
        //                 .map(|buffer| PolynomialCoeffs::new(buffer.to_vec()))
        //                 .collect::<Vec<PolynomialCoeffs<F>>>()
        //         })
        //         .collect()
        // );

        Self::from_coeffs(
            coeffs,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }

    /// Creates a list polynomial commitment for the polynomials `polynomials`.
    pub fn from_coeffs_cpu(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let degree = polynomials[0].len();

        let lde_values = timed!(
            timing,
            "FFT + blinding",
            Self::lde_values(&polynomials, rate_bits, blinding, fft_root_table)
        );

        let mut leaves = timed!(timing, "transpose LDEs", transpose(&lde_values));
        reverse_index_bits_in_place(&mut leaves);
        let merkle_tree = timed!(
            timing,
            "build Merkle tree",
            MerkleTree::new_from_2d(leaves, cap_height)
        );

        Self {
            polynomials,
            merkle_tree,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        Self::from_coeffs_cpu(
            polynomials,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }

    #[cfg(feature = "cuda")]
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let degree = polynomials[0].len();
        let log_n = log2_strict(degree);

        if log_n + rate_bits > 1 && polynomials.len() > 0 {
            let _num_gpus: usize = std::env::var("NUM_OF_GPUS")
                .expect("NUM_OF_GPUS should be set")
                .parse()
                .unwrap();

            let merkle_tree = Self::from_coeffs_gpu(
                &polynomials,
                rate_bits,
                blinding,
                cap_height,
                timing,
                fft_root_table,
                log_n,
                degree,
            );

            return Self {
                polynomials,
                merkle_tree,
                degree_log: log2_strict(degree),
                rate_bits,
                blinding,
            };
        } else {
            Self::from_coeffs_cpu(
                polynomials,
                rate_bits,
                blinding,
                cap_height,
                timing,
                fft_root_table,
            )
        }
    }

    #[cfg(feature = "cuda")]
    pub fn from_coeffs_gpu(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        _blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        _fft_root_table: Option<&FftRootTable<F>>,
        log_n: usize,
        _degree: usize,
    ) -> MerkleTree<F, <C as GenericConfig<D>>::Hasher> {
        // let salt_size = if blinding { SALT_SIZE } else { 0 };
        // println!("salt_size: {:?}", salt_size);
        let output_domain_size = log_n + rate_bits;

        let num_gpus: usize = std::env::var("NUM_OF_GPUS")
            .expect("NUM_OF_GPUS should be set")
            .parse()
            .unwrap();
        // let num_gpus: usize = 1;
        // println!("get num of gpus: {:?}", num_gpus);
        let total_num_of_fft = polynomials.len();
        // println!("total_num_of_fft: {:?}", total_num_of_fft);

        let total_num_input_elements = total_num_of_fft * (1 << log_n);
        let total_num_output_elements = total_num_of_fft * (1 << output_domain_size);

        let mut gpu_input: Vec<F> = polynomials
            .into_iter()
            .flat_map(|v| v.coeffs.iter().cloned())
            .collect();

        let mut cfg_lde = NTTConfig::default();
        cfg_lde.batches = total_num_of_fft as u32;
        cfg_lde.extension_rate_bits = rate_bits as u32;
        cfg_lde.are_inputs_on_device = false;
        cfg_lde.are_outputs_on_device = true;
        cfg_lde.with_coset = true;
        cfg_lde.is_multi_gpu = true;

        let mut device_output_data: HostOrDeviceSlice<'_, F> =
            HostOrDeviceSlice::cuda_malloc(0 as i32, total_num_output_elements).unwrap();
        if num_gpus == 1 {
            let _ = timed!(
                timing,
                "LDE on 1 GPU",
                lde_batch(
                    0,
                    device_output_data.as_mut_ptr(),
                    gpu_input.as_mut_ptr(),
                    log_n,
                    cfg_lde.clone()
                )
            );
        } else {
            let _ = timed!(
                timing,
                "LDE on multi GPU",
                lde_batch_multi_gpu::<F>(
                    device_output_data.as_mut_ptr(),
                    gpu_input.as_mut_ptr(),
                    num_gpus,
                    cfg_lde.clone(),
                    log_n,
                    total_num_input_elements,
                    total_num_output_elements,
                )
            );
        }

        let mut cfg_trans = TransposeConfig::default();
        cfg_trans.batches = total_num_of_fft as u32;
        cfg_trans.are_inputs_on_device = true;
        cfg_trans.are_outputs_on_device = true;

        let mut device_transpose_data: HostOrDeviceSlice<'_, F> =
            HostOrDeviceSlice::cuda_malloc(0 as i32, total_num_output_elements).unwrap();

        let _ = timed!(
            timing,
            "transpose",
            transpose_rev_batch(
                0 as i32,
                device_transpose_data.as_mut_ptr(),
                device_output_data.as_mut_ptr(),
                output_domain_size,
                cfg_trans
            )
        );

        let mt = timed!(
            timing,
            "Merkle tree with GPU data",
            MerkleTree::new_from_gpu_leaves(
                &device_transpose_data,
                1 << output_domain_size,
                total_num_of_fft,
                cap_height
            )
        );
        mt
    }

    fn lde_values(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Vec<Vec<F>> {
        let degree = polynomials[0].len();
        #[cfg(all(feature = "cuda", feature = "batch"))]
        let log_n = log2_strict(degree) + rate_bits;

        // If blinding, salt with two random elements to each leaf vector.
        let salt_size = if blinding { SALT_SIZE } else { 0 };
        // println!("salt_size: {:?}", salt_size);

        #[cfg(all(feature = "cuda", feature = "batch"))]
        let num_gpus: usize = std::env::var("NUM_OF_GPUS")
            .expect("NUM_OF_GPUS should be set")
            .parse()
            .unwrap();
        // let num_gpus: usize = 1;
        #[cfg(all(feature = "cuda", feature = "batch"))]
        println!("get num of gpus: {:?}", num_gpus);
        #[cfg(all(feature = "cuda", feature = "batch"))]
        let total_num_of_fft = polynomials.len();
        // println!("total_num_of_fft: {:?}", total_num_of_fft);
        #[cfg(all(feature = "cuda", feature = "batch"))]
        let per_device_batch = total_num_of_fft.div_ceil(num_gpus);

        #[cfg(all(feature = "cuda", feature = "batch"))]
        let chunk_size = total_num_of_fft.div_ceil(num_gpus);

        #[cfg(all(feature = "cuda", feature = "batch"))]
        if log_n > 10 && polynomials.len() > 0 {
            println!("log_n: {:?}", log_n);
            let start_lde = std::time::Instant::now();

            // let poly_chunk = polynomials;
            // let id = 0;
            let ret = polynomials
                .par_chunks(chunk_size)
                .enumerate()
                .flat_map(|(id, poly_chunk)| {
                    println!(
                        "invoking ntt_batch, device_id: {:?}, per_device_batch: {:?}",
                        id, per_device_batch
                    );

                    let start = std::time::Instant::now();

                    let input_domain_size = 1 << log2_strict(degree);
                    let device_input_data: HostOrDeviceSlice<'_, F> =
                        HostOrDeviceSlice::cuda_malloc(
                            id as i32,
                            input_domain_size * polynomials.len(),
                        )
                        .unwrap();
                    let device_input_data = std::sync::RwLock::new(device_input_data);

                    poly_chunk.par_iter().enumerate().for_each(|(i, p)| {
                        // println!("copy for index: {:?}", i);
                        let _guard = device_input_data.read().unwrap();
                        let _ = _guard.copy_from_host_offset(
                            p.coeffs.as_slice(),
                            input_domain_size * i,
                            input_domain_size,
                        );
                    });

                    println!("data transform elapsed: {:?}", start.elapsed());
                    let mut cfg_lde = NTTConfig::default();
                    cfg_lde.batches = per_device_batch as u32;
                    cfg_lde.extension_rate_bits = rate_bits as u32;
                    cfg_lde.are_inputs_on_device = true;
                    cfg_lde.are_outputs_on_device = true;
                    cfg_lde.with_coset = true;
                    println!(
                        "start cuda_malloc with elements: {:?}",
                        (1 << log_n) * per_device_batch
                    );
                    let mut device_output_data: HostOrDeviceSlice<'_, F> =
                        HostOrDeviceSlice::cuda_malloc(id as i32, (1 << log_n) * per_device_batch)
                            .unwrap();

                    let start = std::time::Instant::now();
                    lde_batch::<F>(
                        id,
                        device_output_data.as_mut_ptr(),
                        device_input_data.read().unwrap().as_ptr(),
                        log2_strict(degree),
                        cfg_lde,
                    );
                    println!("real lde_batch elapsed: {:?}", start.elapsed());
                    let start = std::time::Instant::now();
                    let nums: Vec<usize> = (0..poly_chunk.len()).collect();
                    let r = nums
                        .par_iter()
                        .map(|i| {
                            let mut host_data: Vec<F> = vec![F::ZERO; 1 << log_n];
                            let _ = device_output_data.copy_to_host_offset(
                                host_data.as_mut_slice(),
                                (1 << log_n) * i,
                                1 << log_n,
                            );
                            PolynomialValues::new(host_data).values
                        })
                        .collect::<Vec<Vec<F>>>();
                    println!("collect data from gpu used: {:?}", start.elapsed());
                    r
                })
                // .chain(
                //     (0..salt_size)
                //         .into_par_iter()
                //         .map(|_| F::rand_vec(degree << rate_bits)),
                // )
                .collect();
            println!("real lde elapsed: {:?}", start_lde.elapsed());
            return ret;
        }

        let ret = polynomials
            .par_iter()
            .map(|p| {
                assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                p.lde(rate_bits)
                    .coset_fft_with_options(F::coset_shift(), Some(rate_bits), fft_root_table)
                    .values
            })
            .chain(
                (0..salt_size)
                    .into_par_iter()
                    .map(|_| F::rand_vec(degree << rate_bits)),
            )
            .collect();
        return ret;
    }

    /// Fetches LDE values at the `index * step`th point.
    pub fn get_lde_values(&self, index: usize, step: usize) -> &[F] {
        let index = index * step;
        let index = reverse_bits(index, self.degree_log + self.rate_bits);
        let slice = &self.merkle_tree.get(index);
        &slice[..slice.len() - if self.blinding { SALT_SIZE } else { 0 }]
    }

    /// Like `get_lde_values`, but fetches LDE values from a batch of `P::WIDTH` points, and returns
    /// packed values.
    pub fn get_lde_values_packed<P>(&self, index_start: usize, step: usize) -> Vec<P>
    where
        P: PackedField<Scalar = F>,
    {
        let row_wise = (0..P::WIDTH)
            .map(|i| self.get_lde_values(index_start + i, step))
            .collect_vec();

        // This is essentially a transpose, but we will not use the generic transpose method as we
        // want inner lists to be of type P, not Vecs which would involve allocation.
        let leaf_size = row_wise[0].len();
        (0..leaf_size)
            .map(|j| {
                let mut packed = P::ZEROS;
                packed
                    .as_slice_mut()
                    .iter_mut()
                    .zip(&row_wise)
                    .for_each(|(packed_i, row_i)| *packed_i = row_i[j]);
                packed
            })
            .collect_vec()
    }

    /// Produces a batch opening proof.
    pub fn prove_openings(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        assert!(D > 1, "Not implemented for D=1.");
        let alpha = challenger.get_extension_challenge::<D>();
        let mut alpha = ReducingFactor::new(alpha);

        // Final low-degree polynomial that goes into FRI.
        let mut final_poly = PolynomialCoeffs::empty();

        // Each batch `i` consists of an opening point `z_i` and polynomials `{f_ij}_j` to be opened at that point.
        // For each batch, we compute the composition polynomial `F_i = sum alpha^j f_ij`,
        // where `alpha` is a random challenge in the extension field.
        // The final polynomial is then computed as `final_poly = sum_i alpha^(k_i) (F_i(X) - F_i(z_i))/(X-z_i)`
        // where the `k_i`s are chosen such that each power of `alpha` appears only once in the final sum.
        // There are usually two batches for the openings at `zeta` and `g * zeta`.
        // The oracles used in Plonky2 are given in `FRI_ORACLES` in `plonky2/src/plonk/plonk_common.rs`.
        for FriBatchInfo { point, polynomials } in &instance.batches {
            // Collect the coefficients of all the polynomials in `polynomials`.
            let polys_coeff = polynomials.iter().map(|fri_poly| {
                &oracles[fri_poly.oracle_index].polynomials[fri_poly.polynomial_index]
            });
            let composition_poly = timed!(
                timing,
                &format!("reduce batch of {} polynomials", polynomials.len()),
                alpha.reduce_polys_base(polys_coeff)
            );
            let quotient = composition_poly.divide_by_linear(*point);
            // quotient.coeffs.push(F::Extension::ZERO); // pad back to power of two
            alpha.shift_poly(&mut final_poly);
            final_poly += quotient;
        }
        // NOTE: circom_compatability
        // Multiply the final polynomial by `X`, so that `final_poly` has the maximum degree for
        // which the LDT will pass. See github.com/mir-protocol/plonky2/pull/436 for details.
        final_poly.coeffs.insert(0, F::Extension::ZERO);

        let lde_final_poly = final_poly.lde(fri_params.config.rate_bits);
        let lde_final_values = timed!(
            timing,
            &format!("perform final FFT {}", lde_final_poly.len()),
            lde_final_poly.coset_fft(F::coset_shift().into())
        );

        let fri_proof = fri_proof::<F, C, D>(
            &oracles
                .par_iter()
                .map(|c| &c.merkle_tree)
                .collect::<Vec<_>>(),
            lde_final_poly,
            lde_final_values,
            challenger,
            fri_params,
            timing,
        );

        fri_proof
    }
}
