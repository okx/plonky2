#[cfg(feature = "cuda")]
use alloc::sync::Arc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::mem::MaybeUninit;
use core::slice;
use std::collections::HashSet;
#[cfg(feature = "cuda")]
use std::sync::Mutex;
use std::time::Instant;

#[cfg(feature = "cuda")]
use cryptography_cuda::device::memory::HostOrDeviceSlice;
#[cfg(feature = "cuda")]
use cryptography_cuda::device::stream::CudaStream;
#[cfg(feature = "cuda")]
use cryptography_cuda::{
    fill_digests_buf_linear_gpu_with_gpu_ptr, fill_digests_buf_linear_multigpu_with_gpu_ptr,
};
use num::range;
#[cfg(feature = "cuda")]
use once_cell::sync::Lazy;
use plonky2_maybe_rayon::*;
use serde::{Deserialize, Serialize};

#[cfg(all(target_feature = "avx2", target_feature = "avx512dq"))]
use crate::hash::arch::x86_64::poseidon_goldilocks_avx512::{hash_leaf_avx512, hash_two_avx512};
use crate::hash::hash_types::RichField;
#[cfg(feature = "cuda")]
use crate::hash::hash_types::NUM_HASH_OUT_ELTS;
use crate::hash::merkle_proofs::MerkleProof;
#[cfg(feature = "cuda")]
use crate::plonk::config::HasherType;
use crate::plonk::config::{GenericHashOut, Hasher};
use crate::util::log2_strict;

#[cfg(feature = "cuda")]
pub static GPU_ID: Lazy<Arc<Mutex<u64>>> = Lazy::new(|| Arc::new(Mutex::new(0)));

#[cfg(feature = "cuda_timing")]
fn print_time(now: Instant, msg: &str) {
    println!("Time {} {} ms", msg, now.elapsed().as_millis());
}

#[cfg(not(feature = "cuda_timing"))]
fn print_time(_now: Instant, _msg: &str) {}

#[cfg(feature = "cuda")]
const FORCE_SINGLE_GPU: bool = true;

/// The Merkle cap of height `h` of a Merkle tree is the `h`-th layer (from the root) of the tree.
/// It can be used in place of the root to verify Merkle paths, which are `h` elements shorter.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(bound = "")]
// TODO: Change H to GenericHashOut<F>, since this only cares about the hash, not the hasher.
pub struct MerkleCap<F: RichField, H: Hasher<F>>(pub Vec<H::Hash>);

impl<F: RichField, H: Hasher<F>> Default for MerkleCap<F, H> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<F: RichField, H: Hasher<F>> MerkleCap<F, H> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn height(&self) -> usize {
        log2_strict(self.len())
    }

    pub fn flatten(&self) -> Vec<F> {
        self.0.iter().flat_map(|&h| h.to_vec()).collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MerkleTree<F: RichField, H: Hasher<F>> {
    /// The data in the leaves of the Merkle tree.
    // pub leaves: Vec<Vec<F>>,
    leaves: Vec<F>,

    pub leaf_size: usize,

    /// The digests in the tree. Consists of `cap.len()` sub-trees, each corresponding to one
    /// element in `cap`. Each subtree is contiguous and located at
    /// `digests[digests.len() / cap.len() * i..digests.len() / cap.len() * (i + 1)]`.
    /// Within each subtree, siblings are stored next to each other. The layout is,
    /// left_child_subtree || left_child_digest || right_child_digest || right_child_subtree, where
    /// left_child_digest and right_child_digest are H::Hash and left_child_subtree and
    /// right_child_subtree recurse. Observe that the digest of a node is stored by its _parent_.
    /// Consequently, the digests of the roots are not stored here (they can be found in `cap`).
    pub digests: Vec<H::Hash>,

    /// The Merkle cap.
    pub cap: MerkleCap<F, H>,
}

impl<F: RichField, H: Hasher<F>> Default for MerkleTree<F, H> {
    fn default() -> Self {
        Self {
            leaf_size: 0,
            leaves: Vec::new(),
            digests: Vec::new(),
            cap: MerkleCap::default(),
        }
    }
}

fn capacity_up_to_mut<T>(v: &mut Vec<T>, len: usize) -> &mut [MaybeUninit<T>] {
    assert!(v.capacity() >= len);
    let v_ptr = v.as_mut_ptr().cast::<MaybeUninit<T>>();
    unsafe {
        // SAFETY: `v_ptr` is a valid pointer to a buffer of length at least `len`. Upon return, the
        // lifetime will be bound to that of `v`. The underlying memory will not be deallocated as
        // we hold the sole mutable reference to `v`. The contents of the slice may be
        // uninitialized, but the `MaybeUninit` makes it safe.
        slice::from_raw_parts_mut(v_ptr, len)
    }
}

fn fill_subtree<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[F],
    leaf_size: usize,
) -> H::Hash {
    let leaves_count = leaves.len() / leaf_size;

    // if one leaf => return it hash
    if leaves_count == 1 {
        let hash = H::hash_or_noop(leaves);
        digests_buf[0].write(hash);
        return hash;
    }
    // if two leaves => return their concat hash
    if leaves_count == 2 {
        let (leaf1, leaf2) = leaves.split_at(leaf_size);
        let hash_left = H::hash_or_noop(leaf1);
        let hash_right = H::hash_or_noop(leaf2);
        digests_buf[0].write(hash_left);
        digests_buf[1].write(hash_right);
        return H::two_to_one(hash_left, hash_right);
    }

    assert_eq!(leaves_count, digests_buf.len() / 2 + 1);

    // leaves first - we can do all in parallel
    let (_, digests_leaves) = digests_buf.split_at_mut(digests_buf.len() - leaves_count);
    digests_leaves
        .into_par_iter()
        .enumerate()
        .for_each(|(leaf_idx, digest)| {
            let (_, r) = leaves.split_at(leaf_idx * leaf_size);
            let (leaf, _) = r.split_at(leaf_size);
            digest.write(H::hash_or_noop(leaf));
        });

    // internal nodes - we can do in parallel per level
    let mut last_index = digests_buf.len() - leaves_count;

    for level_log in range(1, log2_strict(leaves_count)).rev() {
        let level_size = 1 << level_log;
        let (_, digests_slice) = digests_buf.split_at_mut(last_index - level_size);
        let (digests_slice, next_digests) = digests_slice.split_at_mut(level_size);

        digests_slice
            .into_par_iter()
            .zip(last_index - level_size..last_index)
            .for_each(|(digest, idx)| {
                let left_idx = 2 * (idx + 1) - last_index;
                let right_idx = left_idx + 1;

                unsafe {
                    let left_digest = next_digests[left_idx].assume_init();
                    let right_digest = next_digests[right_idx].assume_init();
                    digest.write(H::two_to_one(left_digest, right_digest));
                }
            });
        last_index -= level_size;
    }

    // return cap hash
    let hash: <H as Hasher<F>>::Hash;
    unsafe {
        let left_digest = digests_buf[0].assume_init();
        let right_digest = digests_buf[1].assume_init();
        hash = H::two_to_one(left_digest, right_digest);
    }
    hash
}

fn fill_digests_buf<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &Vec<F>,
    leaf_size: usize,
    cap_height: usize,
) {
    // Special case of a tree that's all cap. The usual case will panic because we'll try to split
    // an empty slice into chunks of `0`. (We would not need this if there was a way to split into
    // `blah` chunks as opposed to chunks _of_ `blah`.)
    let leaves_count = leaves.len() / leaf_size;

    if digests_buf.is_empty() {
        debug_assert_eq!(cap_buf.len(), leaves_count);
        cap_buf
            .par_iter_mut()
            .enumerate()
            .for_each(|(leaf_idx, cap_buf)| {
                let (_, r) = leaves.split_at(leaf_idx * leaf_size);
                let (leaf, _) = r.split_at(leaf_size);
                cap_buf.write(H::hash_or_noop(leaf));
            });
        return;
    }

    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = leaves_count >> cap_height;
    let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
    let leaves_chunks = leaves.par_chunks_exact(subtree_leaves_len * leaf_size);
    assert_eq!(digests_chunks.len(), cap_buf.len());
    assert_eq!(digests_chunks.len(), leaves_chunks.len());
    digests_chunks.zip(cap_buf).zip(leaves_chunks).for_each(
        |((subtree_digests, subtree_cap), subtree_leaves)| {
            // We have `1 << cap_height` sub-trees, one for each entry in `cap`. They are totally
            // independent, so we schedule one task for each. `digests_buf` and `leaves` are split
            // into `1 << cap_height` slices, one for each sub-tree.
            subtree_cap.write(fill_subtree::<F, H>(
                subtree_digests,
                subtree_leaves,
                leaf_size,
            ));
        },
    );

    // TODO - debug code - to remove in future
    /*
    let digests_count: u64 = digests_buf.len().try_into().unwrap();
    let leaves_count: u64 = leaves.len().try_into().unwrap();
    let cap_height: u64  = cap_height.try_into().unwrap();
    let leaf_size: u64 = leaves[0].len().try_into().unwrap();
    let fname = format!("cpu-{}-{}-{}-{}.txt", digests_count, leaves_count, leaf_size, cap_height);
    let mut file = File::create(fname).unwrap();
    for digest in digests_buf {
        unsafe {
            let hash = digest.assume_init().to_vec();
            for x in hash {
                let str = format!("{} ", x.to_canonical_u64());
                file.write_all(str.as_bytes());
            }
        }
        file.write_all(b"\n");
    }
    */
}

#[cfg(all(target_feature = "avx2", target_feature = "avx512dq"))]
fn fill_subtree_poseidon_avx512<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[F],
    leaf_size: usize,
) -> H::Hash {
    let leaves_count = leaves.len() / leaf_size;

    // if one leaf => return its hash
    if leaves_count == 1 {
        let hash = H::hash_or_noop(leaves);
        digests_buf[0].write(hash);
        return hash;
    }
    // if two leaves => return their concat hash
    if leaves_count == 2 {
        let (h1, h2) = hash_leaf_avx512(leaves, leaf_size);
        let hash_left = H::Hash::from_vec(&h1);
        let hash_right = H::Hash::from_vec(&h2);
        digests_buf[0].write(hash_left);
        digests_buf[1].write(hash_right);
        return H::two_to_one(hash_left, hash_right);
    }

    assert_eq!(leaves_count, digests_buf.len() / 2 + 1);

    // leaves first - we can do all in parallel
    let (_, digests_leaves) = digests_buf.split_at_mut(digests_buf.len() - leaves_count);
    digests_leaves
        .par_chunks_mut(2)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, digests)| {
            let (_, r) = leaves.split_at(2 * chunk_idx * leaf_size);
            let (leaves2, _) = r.split_at(2 * leaf_size);
            let (h1, h2) = hash_leaf_avx512(leaves2, leaf_size);
            let h1 = H::Hash::from_vec(&h1);
            let h2 = H::Hash::from_vec(&h2);
            digests[0].write(h1);
            digests[1].write(h2);
        });

    // internal nodes - we can do in parallel per level
    let mut last_index = digests_buf.len() - leaves_count;

    for level_log in range(1, log2_strict(leaves_count)).rev() {
        let level_size = 1 << level_log;
        let (_, digests_slice) = digests_buf.split_at_mut(last_index - level_size);
        let (digests_slice, next_digests) = digests_slice.split_at_mut(level_size);

        digests_slice
            .par_chunks_mut(2)
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_idx, digests)| {
                let idx = last_index - level_size + 2 * chunk_idx;
                let left_idx1 = 2 * (idx + 1) - last_index;
                let right_idx1 = left_idx1 + 1;
                let left_idx2 = right_idx1 + 1;
                let right_idx2 = left_idx2 + 1;

                unsafe {
                    let left_digest1 = next_digests[left_idx1].assume_init().to_vec();
                    let right_digest1 = next_digests[right_idx1].assume_init().to_vec();
                    let left_digest2 = next_digests[left_idx2].assume_init().to_vec();
                    let right_digest2 = next_digests[right_idx2].assume_init().to_vec();

                    let (h1, h2) = hash_two_avx512(
                        &left_digest1,
                        &right_digest1,
                        &left_digest2,
                        &right_digest2,
                    );
                    let h1 = H::Hash::from_vec(&h1);
                    let h2 = H::Hash::from_vec(&h2);
                    digests[0].write(h1);
                    digests[1].write(h2);
                }
            });
        last_index -= level_size;
    }

    // return cap hash
    let hash: <H as Hasher<F>>::Hash;
    unsafe {
        let left_digest = digests_buf[0].assume_init();
        let right_digest = digests_buf[1].assume_init();
        hash = H::two_to_one(left_digest, right_digest);
    }
    hash
}

#[cfg(all(target_feature = "avx2", target_feature = "avx512dq"))]
fn fill_digests_buf_poseidon_avx515<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &Vec<F>,
    leaf_size: usize,
    cap_height: usize,
) {
    let leaves_count = leaves.len() / leaf_size;
    if digests_buf.is_empty() {
        debug_assert_eq!(cap_buf.len(), leaves_count);
        cap_buf
            .par_chunks_mut(2)
            .into_par_iter()
            .enumerate()
            .for_each(|(leaf_idx, cap_buf)| {
                let (_, r) = leaves.split_at(2 * leaf_idx * leaf_size);
                let (lv, _) = r.split_at(2 * leaf_size);
                let (h1, h2) = hash_leaf_avx512(lv, leaf_size);
                cap_buf[0].write(H::Hash::from_vec(&h1));
                cap_buf[1].write(H::Hash::from_vec(&h2));
            });
        return;
    }

    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = leaves_count >> cap_height;
    let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
    let leaves_chunks = leaves.par_chunks_exact(subtree_leaves_len * leaf_size);
    assert_eq!(digests_chunks.len(), cap_buf.len());
    assert_eq!(digests_chunks.len(), leaves_chunks.len());
    digests_chunks.zip(cap_buf).zip(leaves_chunks).for_each(
        |((subtree_digests, subtree_cap), subtree_leaves)| {
            subtree_cap.write(fill_subtree_poseidon_avx512::<F, H>(
                subtree_digests,
                subtree_leaves,
                leaf_size,
            ));
        },
    );
}

#[cfg(feature = "cuda")]
fn fill_digests_buf_gpu<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &Vec<F>,
    leaf_size: usize,
    cap_height: usize,
) {
    let leaves_count = leaves.len() / leaf_size;

    let num_gpus: usize = std::env::var("NUM_OF_GPUS")
        .expect("NUM_OF_GPUS should be set")
        .parse()
        .unwrap();

    let mut gpu_id_lock = GPU_ID.lock().unwrap();
    let gpu_id = *gpu_id_lock;
    *gpu_id_lock += 1;
    if *gpu_id_lock >= num_gpus as u64 {
        *gpu_id_lock = 0;
    }

    let now = Instant::now();
    let mut gpu_leaves_buf: HostOrDeviceSlice<'_, F> =
        HostOrDeviceSlice::cuda_malloc(gpu_id as i32, leaves.len()).unwrap();
    print_time(now, "alloc gpu leaves buffer");

    let now = Instant::now();
    let _ = gpu_leaves_buf.copy_from_host(leaves.as_slice());
    print_time(now, "leaves copy to gpu");

    let now = Instant::now();
    fill_digests_buf_gpu_ptr::<F, H>(
        digests_buf,
        cap_buf,
        gpu_leaves_buf.as_mut_ptr(),
        leaves_count,
        leaf_size,
        cap_height,
        gpu_id,
    );
    print_time(now, "fill_digests_buf_gpu_ptr");
}

#[cfg(feature = "cuda")]
fn fill_digests_buf_gpu_ptr<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves_ptr: *const F,
    leaves_len: usize,
    leaf_len: usize,
    cap_height: usize,
    gpu_id: u64,
) {
    let digests_count: u64 = digests_buf.len().try_into().unwrap();
    let leaves_count: u64 = leaves_len.try_into().unwrap();
    let caps_count: u64 = cap_buf.len().try_into().unwrap();
    let cap_height: u64 = cap_height.try_into().unwrap();
    let leaf_size: u64 = leaf_len.try_into().unwrap();

    let now = Instant::now();
    // if digests_buf is empty (size 0), just allocate a few bytes to avoid errors
    let digests_size = if digests_buf.len() == 0 {
        NUM_HASH_OUT_ELTS
    } else {
        digests_buf.len() * NUM_HASH_OUT_ELTS
    };
    let caps_size = if cap_buf.len() == 0 {
        NUM_HASH_OUT_ELTS
    } else {
        cap_buf.len() * NUM_HASH_OUT_ELTS
    };

    let mut gpu_digests_buf: HostOrDeviceSlice<'_, F> =
        HostOrDeviceSlice::cuda_malloc(gpu_id as i32, digests_size).unwrap();
    let mut gpu_cap_buf: HostOrDeviceSlice<'_, F> =
        HostOrDeviceSlice::cuda_malloc(gpu_id as i32, caps_size).unwrap();

    unsafe {
        let num_gpus: usize = std::env::var("NUM_OF_GPUS")
            .expect("NUM_OF_GPUS should be set")
            .parse()
            .unwrap();
        if !FORCE_SINGLE_GPU
            && leaves_count >= (1 << 12)
            && cap_height > 0
            && num_gpus > 1
            && H::HASHER_TYPE == HasherType::PoseidonBN128
        {
            // println!("Multi GPU");
            fill_digests_buf_linear_multigpu_with_gpu_ptr(
                gpu_digests_buf.as_mut_ptr() as *mut core::ffi::c_void,
                gpu_cap_buf.as_mut_ptr() as *mut core::ffi::c_void,
                leaves_ptr as *mut core::ffi::c_void,
                digests_count,
                caps_count,
                leaves_count,
                leaf_size,
                cap_height,
                H::HASHER_TYPE as u64,
            );
        } else {
            // println!("Single GPU");
            fill_digests_buf_linear_gpu_with_gpu_ptr(
                gpu_digests_buf.as_mut_ptr() as *mut core::ffi::c_void,
                gpu_cap_buf.as_mut_ptr() as *mut core::ffi::c_void,
                leaves_ptr as *mut core::ffi::c_void,
                digests_count,
                caps_count,
                leaves_count,
                leaf_size,
                cap_height,
                H::HASHER_TYPE as u64,
                gpu_id,
            );
        }
    }
    print_time(now, "fill init");

    let stream1 = CudaStream::create().unwrap();
    let stream2 = CudaStream::create().unwrap();

    if digests_buf.len() != 0 {
        gpu_digests_buf
            .copy_to_host_ptr_async(
                digests_buf.as_mut_ptr() as *mut core::ffi::c_void,
                digests_size,
                &stream1,
            )
            .expect("copy digests");
    }
    if cap_buf.len() != 0 {
        gpu_cap_buf
            .copy_to_host_ptr_async(
                cap_buf.as_mut_ptr() as *mut core::ffi::c_void,
                caps_size,
                &stream2,
            )
            .expect("copy caps");
    }
    stream1.synchronize().expect("cuda sync");
    stream2.synchronize().expect("cuda sync");
    stream1.destroy().expect("cuda stream destroy");
    stream2.destroy().expect("cuda stream destroy");

    let now = Instant::now();

    print_time(now, "copy results");
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn fill_digests_buf_cpu<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &Vec<F>,
    leaf_size: usize,
    cap_height: usize,
) {
    use cryptography_cuda::fill_digests_buf_linear_cpu;

    let leaves_count = (leaves.len() / leaf_size) as u64;
    let digests_count: u64 = digests_buf.len().try_into().unwrap();
    let caps_count: u64 = cap_buf.len().try_into().unwrap();
    let cap_height: u64 = cap_height.try_into().unwrap();

    unsafe {
        fill_digests_buf_linear_cpu(
            digests_buf.as_mut_ptr() as *mut core::ffi::c_void,
            cap_buf.as_mut_ptr() as *mut core::ffi::c_void,
            leaves.as_ptr() as *const core::ffi::c_void,
            digests_count,
            caps_count,
            leaves_count,
            leaf_size as u64,
            cap_height,
            H::HASHER_TYPE as u64,
        );
    }
}

#[cfg(feature = "cuda")]
fn fill_digests_buf_meta<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &Vec<F>,
    leaf_size: usize,
    cap_height: usize,
) {
    // if the input is small or if it Keccak hashing, just do the hashing on CPU
    if leaf_size <= H::HASH_SIZE / 8 || H::HASHER_TYPE == HasherType::Keccak {
        fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves, leaf_size, cap_height);
    } else {
        fill_digests_buf_gpu::<F, H>(digests_buf, cap_buf, leaves, leaf_size, cap_height);
        // fill_digests_buf_cpu::<F, H>(digests_buf, cap_buf, leaves, leaf_size, cap_height);
    }
}

#[cfg(all(
    not(feature = "cuda"),
    not(all(target_feature = "avx2", target_feature = "avx512dq"))
))]
fn fill_digests_buf_meta<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &Vec<F>,
    leaf_size: usize,
    cap_height: usize,
) {
    fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves, leaf_size, cap_height);
}

#[cfg(all(target_feature = "avx2", target_feature = "avx512dq"))]
fn fill_digests_buf_meta<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &Vec<F>,
    leaf_size: usize,
    cap_height: usize,
) {
    use crate::plonk::config::HasherType;

    if leaf_size <= H::HASH_SIZE / 8 || H::HASHER_TYPE != HasherType::Poseidon {
        fill_digests_buf::<F, H>(digests_buf, cap_buf, leaves, leaf_size, cap_height);
    } else {
        fill_digests_buf_poseidon_avx515::<F, H>(
            digests_buf,
            cap_buf,
            leaves,
            leaf_size,
            cap_height,
        );
    }
}

impl<F: RichField, H: Hasher<F>> MerkleTree<F, H> {
    pub fn new_from_1d(leaves_1d: Vec<F>, leaf_size: usize, cap_height: usize) -> Self {
        let leaves_len = leaves_1d.len() / leaf_size;
        let log2_leaves_len = log2_strict(leaves_len);
        assert!(
            cap_height <= log2_leaves_len,
            "cap_height={} should be at most log2(leaves.len())={}",
            cap_height,
            log2_leaves_len
        );

        let num_digests = 2 * (leaves_len - (1 << cap_height));
        let mut digests: Vec<<H as Hasher<F>>::Hash> = Vec::with_capacity(num_digests);

        let len_cap = 1 << cap_height;
        let mut cap: Vec<<H as Hasher<F>>::Hash> = Vec::with_capacity(len_cap);

        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
        let now = Instant::now();
        fill_digests_buf_meta::<F, H>(digests_buf, cap_buf, &leaves_1d, leaf_size, cap_height);
        print_time(now, "fill digests buffer");

        unsafe {
            // SAFETY: `fill_digests_buf` and `cap` initialized the spare capacity up to
            // `num_digests` and `len_cap`, resp.
            digests.set_len(num_digests);
            cap.set_len(len_cap);
        }
        /*
        println!{"Digest Buffer"};
        for dg in &digests {
            println!("{:?}", dg);
        }
        println!{"Cap Buffer"};
        for dg in &cap {
            println!("{:?}", dg);
        }
        */
        Self {
            leaves: leaves_1d,
            leaf_size,
            digests,
            cap: MerkleCap(cap),
        }
    }

    pub fn new_from_2d(leaves_2d: Vec<Vec<F>>, cap_height: usize) -> Self {
        let leaf_size = leaves_2d[0].len();
        let leaves_count = leaves_2d.len();
        let zeros = vec![F::from_canonical_u64(0); leaf_size];
        let mut leaves_1d: Vec<F> = Vec::with_capacity(leaves_count * leaf_size);
        for idx in 0..leaves_count {
            if leaves_2d[idx].len() == 0 {
                leaves_1d.extend(zeros.clone());
            } else {
                leaves_1d.extend(leaves_2d[idx].clone());
            }
        }
        Self::new_from_1d(leaves_1d, leaf_size, cap_height)
    }

    pub fn new_from_fields(
        leaves_1d: Vec<F>,
        leaf_size: usize,
        digests: Vec<H::Hash>,
        cap: MerkleCap<F, H>,
    ) -> Self {
        Self {
            leaves: leaves_1d,
            leaf_size,
            digests,
            cap,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn new_from_gpu_leaves(
        leaves_gpu_ptr: &HostOrDeviceSlice<'_, F>,
        leaves_len: usize,
        leaf_len: usize,
        cap_height: usize,
    ) -> Self {
        // special case
        if leaf_len <= H::HASH_SIZE / 8 || H::HASHER_TYPE == HasherType::Keccak {
            let mut host_leaves: Vec<F> = vec![F::ZERO; leaves_len * leaf_len];
            leaves_gpu_ptr
                .copy_to_host(host_leaves.as_mut_slice(), leaves_len * leaf_len)
                .expect("copy to host error");
            return Self::new_from_1d(host_leaves, leaf_len, cap_height);
        }

        // general case
        let log2_leaves_len = log2_strict(leaves_len);
        assert!(
            cap_height <= log2_leaves_len,
            "cap_height={} should be at most log2(leaves.len())={}",
            cap_height,
            log2_leaves_len
        );

        // copy data from GPU in async mode
        let mut host_leaves: Vec<F> = vec![F::ZERO; leaves_len * leaf_len];
        let stream_copy = CudaStream::create().unwrap();

        let start = std::time::Instant::now();
        leaves_gpu_ptr
            .copy_to_host_async(host_leaves.as_mut_slice(), &stream_copy)
            .expect("copy to host error");
        print_time(start, "copy leaves from GPU async");

        let num_digests = 2 * (leaves_len - (1 << cap_height));
        let mut digests = Vec::with_capacity(num_digests);

        let len_cap = 1 << cap_height;
        let mut cap = Vec::with_capacity(len_cap);

        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
        let now = Instant::now();
        let gpu_id = 0;
        fill_digests_buf_gpu_ptr::<F, H>(
            digests_buf,
            cap_buf,
            leaves_gpu_ptr.as_ptr(),
            leaves_len,
            leaf_len,
            cap_height,
            gpu_id,
        );
        print_time(now, "fill digests buffer");

        unsafe {
            // SAFETY: `fill_digests_buf` and `cap` initialized the spare capacity up to
            // `num_digests` and `len_cap`, resp.
            digests.set_len(num_digests);
            cap.set_len(len_cap);
        }
        /*
        println!{"Digest Buffer"};
        for dg in &digests {
            println!("{:?}", dg);
        }
        println!{"Cap Buffer"};
        for dg in &cap {
            println!("{:?}", dg);
        }
        */
        let _ = stream_copy.synchronize();
        let _ = stream_copy.destroy();

        Self {
            leaves: host_leaves,
            leaf_size: leaf_len,
            digests,
            cap: MerkleCap(cap),
        }
    }

    pub fn get(&self, i: usize) -> &[F] {
        let (_, v) = self.leaves.split_at(i * self.leaf_size);
        let (v, _) = v.split_at(self.leaf_size);
        v
    }

    pub fn get_leaves_1d(&self) -> Vec<F> {
        self.leaves.clone()
    }

    pub fn get_leaves_2d(&self) -> Vec<Vec<F>> {
        let v2d: Vec<Vec<F>> = self
            .leaves
            .chunks_exact(self.leaf_size)
            .map(|leaf| leaf.to_vec())
            .collect();
        v2d
    }

    pub fn get_leaves_count(&self) -> usize {
        self.leaves.len() / self.leaf_size
    }

    pub fn change_leaf_and_update(&mut self, leaf: Vec<F>, leaf_index: usize) {
        assert_eq!(leaf.len(), self.leaf_size);
        let leaves_count = self.leaves.len() / self.leaf_size;
        assert!(leaf_index < leaves_count);

        let cap_height = log2_strict(self.cap.len());
        let mut leaves = self.leaves.clone();
        let start = leaf_index * self.leaf_size;
        let leaf_copy = leaf.clone();
        leaf.into_iter()
            .enumerate()
            .for_each(|(i, el)| leaves[start + i] = el);

        let digests_len = self.digests.len();
        let cap_len = self.cap.0.len();
        let digests_buf = capacity_up_to_mut(&mut self.digests, digests_len);
        let cap_buf = capacity_up_to_mut(&mut self.cap.0, cap_len);
        self.leaves = leaves;
        if digests_buf.is_empty() {
            cap_buf[leaf_index].write(H::hash_or_noop(leaf_copy.as_slice()));
        } else {
            let subtree_leaves_len = leaves_count >> cap_height;
            let subtree_idx = leaf_index / subtree_leaves_len;
            let subtree_digests_len = digests_buf.len() >> cap_height;
            let subtree_offset = subtree_idx * subtree_digests_len;
            let idx_in_subtree =
                subtree_digests_len - subtree_leaves_len + leaf_index % subtree_leaves_len;
            if subtree_leaves_len == 2 {
                digests_buf[subtree_offset + idx_in_subtree]
                    .write(H::hash_or_noop(leaf_copy.as_slice()));
            } else {
                assert!(subtree_leaves_len > 2);
                let idx = subtree_offset + idx_in_subtree;
                digests_buf[idx].write(H::hash_or_noop(leaf_copy.as_slice()));
                let mut child_idx: i64 = idx_in_subtree as i64;
                let mut parent_idx: i64 = child_idx / 2 - 1;
                while child_idx > 1 {
                    unsafe {
                        let mut left_idx = subtree_offset + child_idx as usize;
                        let mut right_idx = subtree_offset + child_idx as usize + 1;
                        if child_idx & 1 == 1 {
                            left_idx = subtree_offset + child_idx as usize - 1;
                            right_idx = subtree_offset + child_idx as usize;
                        }
                        let left_digest = digests_buf[left_idx].assume_init();
                        let right_digest = digests_buf[right_idx].assume_init();
                        digests_buf[subtree_offset + parent_idx as usize]
                            .write(H::two_to_one(left_digest, right_digest));
                    }
                    child_idx = parent_idx;
                    parent_idx = child_idx / 2 - 1;
                }
            }
            unsafe {
                let left_digest = digests_buf[subtree_offset].assume_init();
                let right_digest = digests_buf[subtree_offset + 1].assume_init();
                cap_buf[subtree_idx].write(H::two_to_one(left_digest, right_digest));
            }
        }
    }

    pub fn change_leaves_in_range_and_update(
        &mut self,
        new_leaves: Vec<Vec<F>>,
        start_index: usize,
        end_index: usize,
    ) {
        assert_eq!(new_leaves.len(), end_index - start_index);
        assert_eq!(new_leaves[0].len(), self.leaf_size);

        let tree_leaves_count = self.leaves.len() / self.leaf_size;
        assert!(start_index < end_index);
        assert!(end_index < tree_leaves_count);

        let cap_height = log2_strict(self.cap.len());
        let mut leaves = self.leaves.clone();

        leaves[start_index * self.leaf_size..end_index * self.leaf_size]
            .chunks_exact_mut(self.leaf_size)
            .zip(new_leaves.clone())
            .for_each(|(x, y)| {
                for j in 0..self.leaf_size {
                    x[j] = y[j];
                }
            });

        let digests_len = self.digests.len();
        let cap_len = self.cap.0.len();
        let digests_buf = capacity_up_to_mut(&mut self.digests, digests_len);
        let cap_buf = capacity_up_to_mut(&mut self.cap.0, cap_len);
        self.leaves = leaves;
        if digests_buf.is_empty() {
            cap_buf[start_index..end_index]
                .par_iter_mut()
                .zip(new_leaves)
                .for_each(|(cap, leaf)| {
                    cap.write(H::hash_or_noop(leaf.as_slice()));
                });
        } else {
            let subtree_leaves_len = tree_leaves_count >> cap_height;
            let subtree_digests_len = digests_buf.len() >> cap_height;

            let mut positions: Vec<usize> = (start_index..end_index)
                .map(|idx| {
                    let subtree_idx = idx / subtree_leaves_len;
                    let subtree_offset = subtree_idx * subtree_digests_len;
                    let idx_in_subtree =
                        subtree_digests_len - subtree_leaves_len + idx % subtree_leaves_len;
                    subtree_offset + idx_in_subtree
                })
                .collect();

            // TODO change to parallel loop
            for i in 0..positions.len() {
                digests_buf[positions[i]].write(H::hash_or_noop(new_leaves[i].as_slice()));
            }

            if subtree_digests_len > 2 {
                let rounds = log2_strict(tree_leaves_count) - cap_height - 1;
                for _ in 0..rounds {
                    let mut parent_indexes: HashSet<usize> = HashSet::new();
                    let parents: Vec<usize> = positions
                        .par_iter()
                        .map(|pos| {
                            let subtree_offset = pos / subtree_digests_len;
                            let idx_in_subtree = pos % subtree_digests_len;
                            let mut parent_idx = 0;
                            if idx_in_subtree > 1 {
                                parent_idx = idx_in_subtree / 2 - 1;
                            }
                            subtree_offset * subtree_digests_len + parent_idx
                        })
                        .collect();
                    for p in parents {
                        parent_indexes.insert(p);
                    }
                    positions = parent_indexes.into_iter().collect();

                    // TODO change to parallel loop
                    for i in 0..positions.len() {
                        let subtree_offset = positions[i] / subtree_digests_len;
                        let idx_in_subtree = positions[i] % subtree_digests_len;
                        let digest_idx =
                            subtree_offset * subtree_digests_len + 2 * (idx_in_subtree + 1);
                        unsafe {
                            let left_digest = digests_buf[digest_idx].assume_init();
                            let right_digest = digests_buf[digest_idx + 1].assume_init();
                            digests_buf[positions[i]]
                                .write(H::two_to_one(left_digest, right_digest));
                        }
                    }
                }
            }

            let mut cap_indexes: HashSet<usize> = HashSet::new();
            for idx in start_index..end_index {
                cap_indexes.insert(idx / subtree_leaves_len);
            }

            unsafe {
                for idx in cap_indexes {
                    let digest_idx = idx * subtree_digests_len;
                    let left_digest = digests_buf[digest_idx].assume_init();
                    let right_digest = digests_buf[digest_idx + 1].assume_init();
                    cap_buf[idx].write(H::two_to_one(left_digest, right_digest));
                }
            }
        }
    }

    /// Create a Merkle proof from a leaf index.
    pub fn prove(&self, leaf_index: usize) -> MerkleProof<F, H> {
        let cap_height = log2_strict(self.cap.len());
        let num_layers = log2_strict(self.get_leaves_count()) - cap_height;
        let subtree_digest_size = (1 << (num_layers + 1)) - 2; // 2 ^ (k+1) - 2
        let subtree_idx = leaf_index / (1 << num_layers);

        let siblings: Vec<<H as Hasher<F>>::Hash> = Vec::with_capacity(num_layers);
        if num_layers == 0 {
            return MerkleProof { siblings };
        }

        // digests index where we start
        let idx = subtree_digest_size - (1 << num_layers) + (leaf_index % (1 << num_layers));

        let siblings = (0..num_layers)
            .map(|i| {
                // relative index
                let rel_idx = (idx + 2 - (1 << i + 1)) / (1 << i);
                // absolute index
                let mut abs_idx = subtree_idx * subtree_digest_size + rel_idx;
                if (rel_idx & 1) == 1 {
                    abs_idx -= 1;
                } else {
                    abs_idx += 1;
                }
                self.digests[abs_idx]
            })
            .collect();

        MerkleProof { siblings }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;
    use crate::field::extension::Extendable;
    use crate::hash::merkle_proofs::verify_merkle_proof_to_cap;
    use crate::hash::poseidon_bn128::PoseidonBN128GoldilocksConfig;
    use crate::plonk::config::{
        GenericConfig, KeccakGoldilocksConfig, Poseidon2GoldilocksConfig, PoseidonGoldilocksConfig,
    };

    fn random_data_2d<F: RichField>(n: usize, k: usize) -> Vec<Vec<F>> {
        (0..n).map(|_| F::rand_vec(k)).collect()
    }

    #[allow(unused)]
    fn random_data_1d<F: RichField>(n: usize, k: usize) -> Vec<F> {
        F::rand_vec(k * n)
    }

    fn verify_all_leaves<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        leaves: Vec<Vec<F>>,
        cap_height: usize,
    ) -> Result<()> {
        let tree = MerkleTree::<F, C::Hasher>::new_from_2d(leaves.clone(), cap_height);
        for (i, leaf) in leaves.into_iter().enumerate() {
            let proof = tree.prove(i);
            verify_merkle_proof_to_cap(leaf, i, &tree.cap, &proof)?;
        }
        Ok(())
    }

    fn verify_change_leaf_and_update(log_n: usize, cap_h: usize) {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let n = 1 << log_n;
        let k = 7;
        let mut leaves = random_data_2d::<F>(n, k);

        let mut mt1 =
            MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new_from_2d(leaves.clone(), cap_h);

        let tmp = random_data_2d::<F>(1, k);
        leaves[0] = tmp[0].clone();
        let mt2 = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new_from_2d(leaves, cap_h);

        mt1.change_leaf_and_update(tmp[0].clone(), 0);

        /*
        println!("Tree 1");
        mt1.digests.into_iter().for_each(
            |x| {
                println!("{:?}", x);
            }
        );
        println!("Tree 2");
        mt2.digests.into_iter().for_each(
            |x| {
                println!("{:?}", x);
            }
        );
        */

        mt1.digests
            .into_par_iter()
            .zip(mt2.digests)
            .for_each(|(d1, d2)| {
                assert_eq!(d1, d2);
            });

        mt1.cap
            .0
            .into_par_iter()
            .zip(mt2.cap.0)
            .for_each(|(d1, d2)| {
                assert_eq!(d1, d2);
            });
    }

    fn verify_change_leaf_and_update_range_one_by_one(
        leaves_count: usize,
        leaf_size: usize,
        cap_height: usize,
        start_index: usize,
        end_index: usize,
    ) {
        use plonky2_field::types::Field;

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let raw_leaves: Vec<Vec<F>> = random_data_2d::<F>(leaves_count, leaf_size);
        let vals: Vec<Vec<F>> = random_data_2d::<F>(end_index - start_index, leaf_size);

        let mut leaves1_1d: Vec<F> = raw_leaves.into_iter().flatten().collect();
        let leaves2_1d: Vec<F> = leaves1_1d.clone();

        let mut tree2 = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new_from_1d(
            leaves2_1d, leaf_size, cap_height,
        );

        // v1
        let now = Instant::now();
        for i in start_index..end_index {
            for j in 0..leaf_size {
                leaves1_1d[i * leaf_size + j] = vals[i - start_index][j];
            }
        }
        let tree1 = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new_from_1d(
            leaves1_1d, leaf_size, cap_height,
        );
        println!("Time V1: {} ms", now.elapsed().as_millis());

        // v2
        let now = Instant::now();
        for idx in start_index..end_index {
            let mut leaf: Vec<F> = vec![F::from_canonical_u64(0); leaf_size];
            for j in 0..leaf_size {
                leaf[j] = vals[idx - start_index][j];
            }
            tree2.change_leaf_and_update(leaf, idx);
        }
        println!("Time V2: {} ms", now.elapsed().as_millis());

        // compare leaves
        let t2leaves = tree2.get_leaves_1d();
        tree1
            .get_leaves_1d()
            .chunks_exact(leaf_size)
            .enumerate()
            .for_each(|(i, x)| {
                let mut ok = true;
                for j in 0..leaf_size {
                    if x[j] != t2leaves[i * leaf_size + j] {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    println!("Leaves different at index {:?}", i);
                }
                assert!(ok);
            });

        // compare trees
        tree1.digests.into_iter().enumerate().for_each(|(i, x)| {
            let y = tree2.digests[i];
            if x != y {
                println!("Digests different at index {:?}", i);
            }
            assert_eq!(x, y);
        });
        tree1.cap.0.into_iter().enumerate().for_each(|(i, x)| {
            let y = tree2.cap.0[i];
            if x != y {
                println!("Cap different at index {:?}", i);
            }
            assert_eq!(x, y);
        });
    }

    fn verify_change_leaf_and_update_range(
        leaves_count: usize,
        leaf_size: usize,
        cap_height: usize,
        start_index: usize,
        end_index: usize,
    ) {
        // use plonky2_field::types::Field;

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let raw_leaves: Vec<Vec<F>> = random_data_2d::<F>(leaves_count, leaf_size);
        let vals: Vec<Vec<F>> = random_data_2d::<F>(end_index - start_index, leaf_size);

        let mut leaves1_1d: Vec<F> = raw_leaves.into_iter().flatten().collect();
        let leaves2_1d: Vec<F> = leaves1_1d.clone();

        let mut tree2 = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new_from_1d(
            leaves2_1d, leaf_size, cap_height,
        );

        // v1
        let now = Instant::now();
        for i in start_index..end_index {
            for j in 0..leaf_size {
                leaves1_1d[i * leaf_size + j] = vals[i - start_index][j];
            }
        }
        let tree1 = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new_from_1d(
            leaves1_1d, leaf_size, cap_height,
        );
        println!("Time V1: {} ms", now.elapsed().as_millis());

        // v2
        let now = Instant::now();
        /*
        for idx in start_index..end_index {
            let mut leaf: Vec<F> = vec![F::from_canonical_u64(0); leaf_size];
            for j in 0..leaf_size {
                leaf[j] = vals[idx - start_index][j];
            }
            tree2.change_leaf_and_update(leaf, idx);
        }
        */
        tree2.change_leaves_in_range_and_update(vals, start_index, end_index);
        println!("Time V2: {} ms", now.elapsed().as_millis());

        // compare leaves
        let t2leaves = tree2.get_leaves_1d();
        tree1
            .get_leaves_1d()
            .chunks_exact(leaf_size)
            .enumerate()
            .for_each(|(i, x)| {
                let mut ok = true;
                for j in 0..leaf_size {
                    if x[j] != t2leaves[i * leaf_size + j] {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    println!("Leaves different at index {:?}", i);
                }
                assert!(ok);
            });

        // compare trees
        tree1.digests.into_iter().enumerate().for_each(|(i, x)| {
            let y = tree2.digests[i];
            if x != y {
                println!("Digests different at index {:?}", i);
            }
            assert_eq!(x, y);
        });
        tree1.cap.0.into_iter().enumerate().for_each(|(i, x)| {
            let y = tree2.cap.0[i];
            if x != y {
                println!("Cap different at index {:?}", i);
            }
            assert_eq!(x, y);
        });
    }

    #[test]
    #[should_panic]
    fn test_cap_height_too_big() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let cap_height = log_n + 1; // Should panic if `cap_height > len_n`.

        let leaves = random_data_2d::<F>(1 << log_n, 7);
        let _ = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new_from_2d(leaves, cap_height);
    }

    #[test]
    fn test_cap_height_eq_log2_len() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data_2d::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, log_n)?;

        Ok(())
    }

    #[test]
    fn test_change_leaf_and_update() -> Result<()> {
        // small tree, 1 cap
        verify_change_leaf_and_update(3, 0);
        // small tree, 2 cap
        verify_change_leaf_and_update(3, 1);
        // small tree, 4 cap
        verify_change_leaf_and_update(3, 2);
        // small tree, all cap
        verify_change_leaf_and_update(3, 3);

        // big tree
        verify_change_leaf_and_update(12, 3);

        Ok(())
    }

    #[test]
    fn test_change_leaf_and_update_range() -> Result<()> {
        for h in 0..11 {
            println!(
                "Run verify_change_leaf_and_update_range_one_by_one() for height {:?}",
                h
            );
            verify_change_leaf_and_update_range_one_by_one(1024, 68, h, 32, 48);
            println!(
                "Run verify_change_leaf_and_update_range() for height {:?}",
                h
            );
            verify_change_leaf_and_update_range(1024, 68, h, 32, 48);
        }

        Ok(())
    }

    #[test]
    fn test_merkle_trees_poseidon_g64() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        // GPU warmup
        #[cfg(feature = "cuda")]
        let _x: HostOrDeviceSlice<'_, F> = HostOrDeviceSlice::cuda_malloc(0, 64).unwrap();

        let log_n = 12;
        let n = 1 << log_n;
        let leaves = random_data_2d::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, 1)?;

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_merkle_trees_cuda_poseidon_g64() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 14;
        let n = 1 << log_n;
        let leaves = random_data_2d::<F>(n, 7);
        let leaves_1d: Vec<F> = leaves.into_iter().flatten().collect();

        let mut gpu_data: HostOrDeviceSlice<'_, F> =
            HostOrDeviceSlice::cuda_malloc(0, n * 7).unwrap();
        gpu_data
            .copy_from_host(leaves_1d.as_slice())
            .expect("copy data to gpu");

        MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new_from_gpu_leaves(&gpu_data, n, 7, 1);

        Ok(())
    }

    #[test]
    fn test_merkle_trees_poseidon2_g64() -> Result<()> {
        const D: usize = 2;
        type C = Poseidon2GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 12;
        let n = 1 << log_n;
        let leaves = random_data_2d::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, 1)?;

        Ok(())
    }

    #[test]
    fn test_merkle_trees_keccak() -> Result<()> {
        const D: usize = 2;
        type C = KeccakGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 12;
        let n = 1 << log_n;
        let leaves = random_data_2d::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, 1)?;

        Ok(())
    }

    #[test]
    fn test_merkle_trees_poseidon_bn128() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonBN128GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 12;
        let n = 1 << log_n;
        let leaves = random_data_2d::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, 1)?;

        Ok(())
    }

    #[cfg(all(target_feature = "avx2", target_feature = "avx512dq"))]
    fn check_consistency<F: RichField, H: Hasher<F>>(
        leaves: &Vec<F>,
        leaves_len: usize,
        leaf_size: usize,
        cap_height: usize,
    ) {
        println!("Check for height: {:?}", cap_height);
        // no AVX
        let num_digests = 2 * (leaves_len - (1 << cap_height));
        let mut digests: Vec<<H as Hasher<F>>::Hash> = Vec::with_capacity(num_digests);
        let len_cap = 1 << cap_height;
        let mut cap: Vec<<H as Hasher<F>>::Hash> = Vec::with_capacity(len_cap);
        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
        fill_digests_buf::<F, H>(digests_buf, cap_buf, &leaves.clone(), leaf_size, cap_height);
        unsafe {
            digests.set_len(num_digests);
            cap.set_len(len_cap);
        }
        // AVX512
        let mut digests_avx512: Vec<<H as Hasher<F>>::Hash> = Vec::with_capacity(num_digests);
        let mut cap_avx512: Vec<<H as Hasher<F>>::Hash> = Vec::with_capacity(len_cap);
        let digests_buf_avx512 = capacity_up_to_mut(&mut digests_avx512, num_digests);
        let cap_buf_avx512 = capacity_up_to_mut(&mut cap_avx512, len_cap);
        fill_digests_buf_poseidon_avx515::<F, H>(
            digests_buf_avx512,
            cap_buf_avx512,
            &leaves,
            leaf_size,
            cap_height,
        );
        unsafe {
            digests_avx512.set_len(num_digests);
            cap_avx512.set_len(len_cap);
        }

        digests
            .into_iter()
            .zip(digests_avx512)
            .for_each(|(d1, d2)| {
                assert_eq!(d1, d2);
            });
        cap.into_iter().zip(cap_avx512).for_each(|(d1, d2)| {
            assert_eq!(d1, d2);
        });
    }

    #[cfg(all(target_feature = "avx2", target_feature = "avx512dq"))]
    #[test]
    fn test_merkle_trees_poseidon_g64_avx512_consistency() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        type H = <C as GenericConfig<D>>::Hasher;

        let leaf_size = 16;
        let log_n = 10;
        let leaves_len = 1 << log_n;
        let leaves: Vec<F> = random_data_1d::<F>(leaves_len, leaf_size);

        for cap_height in 0..log_n {
            check_consistency::<F, H>(&leaves, leaves_len, leaf_size, cap_height);
        }

        Ok(())
    }
}
