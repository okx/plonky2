#[cfg(feature = "cuda")]
use alloc::sync::Arc;
use alloc::vec::Vec;
use num::range;
use core::mem::MaybeUninit;
use core::slice;
#[cfg(feature = "cuda")]
use std::os::raw::c_void;
#[cfg(feature = "cuda")]
use std::sync::Mutex;

#[cfg(feature = "cuda")]
use cryptography_cuda::device::memory::HostOrDeviceSlice;
#[cfg(feature = "cuda")]
use once_cell::sync::Lazy;
use plonky2_maybe_rayon::*;
use serde::{Deserialize, Serialize};

use crate::hash::hash_types::RichField;
#[cfg(feature = "cuda")]
use crate::hash::hash_types::NUM_HASH_OUT_ELTS;
use crate::hash::merkle_proofs::MerkleProof;
use crate::plonk::config::{GenericHashOut, Hasher};
use crate::util::log2_strict;
#[cfg(feature = "cuda")]
use cryptography_cuda::merkle::bindings::{
    fill_delete, fill_digests_buf_linear_gpu, fill_digests_buf_linear_gpu_with_gpu_ptr, fill_init, get_cap_ptr, get_digests_ptr,
    get_leaves_ptr,
};
#[cfg(feature = "cuda")]
use cryptography_cuda::device::stream::CudaStream;

use std::time::Instant;

#[cfg(feature = "cuda")]
static gpu_lock: Lazy<Arc<Mutex<i32>>> = Lazy::new(|| Arc::new(Mutex::new(0)));

#[cfg(feature = "cuda_timing")]
fn print_time(now: Instant, msg: &str)
{
    println!("Time {} {} ms", msg, now.elapsed().as_millis());
}

#[cfg(not(feature = "cuda_timing"))]
fn print_time(_now: Instant, _msg: &str)
{
}

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
    leaves: &[Vec<F>],
) -> H::Hash {
    // if one leaf => return it hash
    if leaves.len() == 1 {
        let hash = H::hash_or_noop(&leaves[0]);
        digests_buf[0].write(hash);
        return hash;
    }
    // if two leaves => return their concat hash
    if leaves.len() == 2 {
        let hash_left = H::hash_or_noop(&leaves[0]);
        let hash_right = H::hash_or_noop(&leaves[1]);
        digests_buf[0].write(hash_left);
        digests_buf[1].write(hash_right);
        return H::two_to_one(hash_left, hash_right);
    }

    assert_eq!(leaves.len(), digests_buf.len() / 2 + 1);

    // leaves first - we can do all in parallel
    let (_, digests_leaves) = digests_buf.split_at_mut(digests_buf.len() - leaves.len());
    digests_leaves
        .into_par_iter()
        .zip(leaves)
        .for_each(|(digest, leaf)| {
            digest.write(H::hash_or_noop(leaf));
        });

    // internal nodes - we can do in parallel per level
    let mut last_index = digests_buf.len() - leaves.len();

    log2_strict(leaves.len());
    for level_log in range(1, log2_strict(leaves.len())).rev() {
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
    leaves: &[Vec<F>],
    cap_height: usize,
) {
    // Special case of a tree that's all cap. The usual case will panic because we'll try to split
    // an empty slice into chunks of `0`. (We would not need this if there was a way to split into
    // `blah` chunks as opposed to chunks _of_ `blah`.)
    if digests_buf.is_empty() {
        debug_assert_eq!(cap_buf.len(), leaves.len());
        cap_buf
            .par_iter_mut()
            .zip(leaves)
            .for_each(|(cap_buf, leaf)| {
                cap_buf.write(H::hash_or_noop(leaf));
            });
        return;
    }

    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = leaves.len() >> cap_height;
    let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
    let leaves_chunks = leaves.par_chunks_exact(subtree_leaves_len);
    assert_eq!(digests_chunks.len(), cap_buf.len());
    assert_eq!(digests_chunks.len(), leaves_chunks.len());
    digests_chunks.zip(cap_buf).zip(leaves_chunks).for_each(
        |((subtree_digests, subtree_cap), subtree_leaves)| {
            // We have `1 << cap_height` sub-trees, one for each entry in `cap`. They are totally
            // independent, so we schedule one task for each. `digests_buf` and `leaves` are split
            // into `1 << cap_height` slices, one for each sub-tree.
            subtree_cap.write(fill_subtree::<F, H>(subtree_digests, subtree_leaves));
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

#[cfg(feature = "cuda")]
#[repr(C)]
union U8U64 {
    f1: [u8; 32],
    f2: [u64; 4],
}

#[cfg(feature = "cuda")]
fn fill_digests_buf_gpu_v1<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
    cap_height: usize,
) {
    let digests_count: u64 = digests_buf.len().try_into().unwrap();
    let leaves_count: u64 = leaves.len().try_into().unwrap();
    let caps_count: u64 = cap_buf.len().try_into().unwrap();
    let cap_height: u64 = cap_height.try_into().unwrap();
    let leaf_size: u64 = leaves[0].len().try_into().unwrap();
    let hash_size: u64 = H::HASH_SIZE.try_into().unwrap();

    let _lock = gpu_lock.lock().unwrap();

    unsafe {
        let now = Instant::now();
        fill_init(
            digests_count,
            leaves_count,
            caps_count,
            leaf_size,
            hash_size,
            H::HASHER_TYPE as u64,
        );
        print_time(now, "fill init");
        let now = Instant::now();

        // copy data to C
        let mut pd: *mut u64 = get_digests_ptr();
        let mut pl: *mut u64 = get_leaves_ptr();
        let mut pc: *mut u64 = get_cap_ptr();

        for leaf in leaves {
            for elem in leaf {
                let val = &elem.to_canonical_u64();
                *pl = *val;
                pl = pl.add(1);
            }
        }

        /*
        let lc = leaves.len();
        leaves.into_iter().enumerate().for_each(
            |(i, leaf)| {
                let mut p = pl;
                p = p.add(i);
                for elem in leaf {
                    let val = &elem.to_canonical_u64();
                    *p = *val;
                    p = p.add(lc);
                }
            }
        );
        */
        print_time(now, "copy data to C");
        let now = Instant::now();

        // println!("Digest size {}, Leaves {}, Leaf size {}, Cap H {}", digests_count, leaves_count, leaf_size, cap_height);
        fill_digests_buf_linear_gpu(
            digests_count,
            caps_count,
            leaves_count,
            leaf_size,
            cap_height,
        );

        print_time(now, "kernel");
        let now = Instant::now();

        // TODO - debug code - to remove in future
        // let mut pd : *mut u64 = get_digests_ptr();
        /*
        println!("*** Digests");
        for i in 0..leaves.len() {
            for j in 0..leaf_size {
                print!("{} ", *pd);
                pd = pd.add(1);
            }
            println!();
        }
        pd = get_digests_ptr();
        */
        /*
        let fname = format!("gpu-{}-{}-{}-{}.txt", digests_count, leaves_count, leaf_size, cap_height);
        let mut file = File::create(fname).unwrap();
        for _i in 0..digests_count {
            for _j in 0..4 {
                let str = format!("{} ", *pd);
                file.write_all(str.as_bytes());
                pd = pd.add(1);
            }
            file.write_all(b"\n");
        }
        pd = get_digests_ptr();
        */

        // copy data from C
        for dg in digests_buf {
            let mut parts = U8U64 { f1: [0; 32] };
            // copy hash from pd to digests_buf
            for i in 0..4 {
                parts.f2[i] = *pd;
                pd = pd.add(1);
            }
            let (slice, _) = parts.f1.split_at(H::HASH_SIZE);
            let h: H::Hash = H::Hash::from_bytes(slice);
            dg.write(h);
        }
        for cp in cap_buf {
            let mut parts = U8U64 { f1: [0; 32] };
            // copy hash from pc to cap_buf
            for i in 0..4 {
                parts.f2[i] = *pc;
                pc = pc.add(1);
            }
            let (slice, _) = parts.f1.split_at(H::HASH_SIZE);
            let h: H::Hash = H::Hash::from_bytes(slice);
            cp.write(h);
        }

        print_time(now, "copy results");
        let now = Instant::now();

        fill_delete();
        print_time(now, "fill delete");
    }
}

#[allow(dead_code)]
#[cfg(feature = "cuda")]
fn fill_digests_buf_gpu_v2<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
    cap_height: usize,
) {
    let _lock = gpu_lock.lock().unwrap();

    let digests_count: u64 = digests_buf.len().try_into().unwrap();
    let leaves_count: u64 = leaves.len().try_into().unwrap();
    let caps_count: u64 = cap_buf.len().try_into().unwrap();
    let cap_height: u64 = cap_height.try_into().unwrap();
    let leaf_size: u64 = leaves[0].len().try_into().unwrap();

    let leaves_size = leaves.len() * leaves[0].len();

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

    // println!("{} {} {} {} {:?}", leaves_count, leaf_size, digests_count, caps_count, H::HASHER_TYPE);

    let mut gpu_leaves_buf: HostOrDeviceSlice<'_, F> =
        HostOrDeviceSlice::cuda_malloc(0, leaves_size).unwrap();
    let mut gpu_digests_buf: HostOrDeviceSlice<'_, F> =
        HostOrDeviceSlice::cuda_malloc(0, digests_size).unwrap();
    let mut gpu_caps_buf: HostOrDeviceSlice<'_, F> =
        HostOrDeviceSlice::cuda_malloc(0, caps_size).unwrap();
    print_time(now, "alloc gpu ds");
    let now = Instant::now();

    // Note: flatten() is very slow, so we use a naive nested for loop
    // let leaves1 = leaves.to_vec().into_iter().flatten().collect::<Vec<F>>();

    // v1: use 2 for loops - better than flatten()
    let mut leaves1 = Vec::with_capacity(leaves.len() * leaves[0].len());
    for leaf in leaves {
        for el in leaf {
            leaves1.push(el.clone());
        }
    }
    /*
    // v2: use par chunks - same performance
    let mut leaves1 = vec![F::ZERO; leaves.len() * leaves[0].len()];
    leaves1.par_chunks_exact_mut(leaves[0].len()).enumerate().for_each(
        |(i, c)| {
            c.copy_from_slice(leaves[i].as_slice());
        }
    );
    */

    let _ = gpu_leaves_buf.copy_from_host(leaves1.as_slice());

    print_time(now, "data copy to gpu");
    let now = Instant::now();

    unsafe {
        fill_digests_buf_linear_gpu_with_gpu_ptr(
            gpu_digests_buf.as_mut_ptr() as *mut c_void,
            gpu_caps_buf.as_mut_ptr() as *mut c_void,
            gpu_leaves_buf.as_ptr() as *mut c_void,
            digests_count,
            caps_count,
            leaves_count,
            leaf_size,
            cap_height,
            H::HASHER_TYPE as u64,
        )
    };
    print_time(now, "kernel");
    let now = Instant::now();

    if digests_buf.len() > 0 {
        let mut host_digests_buf: Vec<F> = vec![F::ZERO; digests_size];
        let _ = gpu_digests_buf.copy_to_host(host_digests_buf.as_mut_slice(), digests_size);
        host_digests_buf
            .par_chunks_exact(4)
            .zip(digests_buf)
            .for_each(|(x, y)| {
                unsafe {
                    let mut parts = U8U64 { f1: [0; 32] };
                    parts.f2[0] = x[0].to_canonical_u64();
                    parts.f2[1] = x[1].to_canonical_u64();
                    parts.f2[2] = x[2].to_canonical_u64();
                    parts.f2[3] = x[3].to_canonical_u64();
                    let (slice, _) = parts.f1.split_at(H::HASH_SIZE);
                    let h: H::Hash = H::Hash::from_bytes(slice);
                    y.write(h);
                };
            });
    }

    if cap_buf.len() > 0 {
        let mut host_caps_buf: Vec<F> = vec![F::ZERO; caps_size];
        let _ = gpu_caps_buf.copy_to_host(host_caps_buf.as_mut_slice(), caps_size);
        host_caps_buf
            .par_chunks_exact(4)
            .zip(cap_buf)
            .for_each(|(x, y)| {
                unsafe {
                    let mut parts = U8U64 { f1: [0; 32] };
                    parts.f2[0] = x[0].to_canonical_u64();
                    parts.f2[1] = x[1].to_canonical_u64();
                    parts.f2[2] = x[2].to_canonical_u64();
                    parts.f2[3] = x[3].to_canonical_u64();
                    let (slice, _) = parts.f1.split_at(H::HASH_SIZE);
                    let h: H::Hash = H::Hash::from_bytes(slice);
                    y.write(h);
                };
            });
    }
    print_time(now, "copy results");
}

#[cfg(feature = "cuda")]
fn fill_digests_buf_gpu_ptr<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves_ptr: *const F,
    leaves_len: usize,
    leaf_len: usize,
    cap_height: usize,
) {
    let digests_count: u64 = digests_buf.len().try_into().unwrap();
    let leaves_count: u64 = leaves_len.try_into().unwrap();
    let caps_count: u64 = cap_buf.len().try_into().unwrap();
    let cap_height: u64 = cap_height.try_into().unwrap();
    let leaf_size: u64 = leaf_len.try_into().unwrap();

    let _lock = gpu_lock.lock().unwrap();

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
            HostOrDeviceSlice::cuda_malloc(0 as i32, digests_size)
                .unwrap();
        let mut gpu_cap_buf: HostOrDeviceSlice<'_, F> =
            HostOrDeviceSlice::cuda_malloc(0 as i32, caps_size)
                .unwrap();

                unsafe{
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
        );
    }
        print_time(now, "fill init");

        let mut host_digests: Vec<F> = vec![F::ZERO; digests_size];
        let mut host_caps: Vec<F> = vec![F::ZERO; caps_size];
        let stream1 = CudaStream::create().unwrap();
        let stream2 = CudaStream::create().unwrap();

        gpu_digests_buf.copy_to_host_async(host_digests.as_mut_slice(), &stream1).expect("copy digests");
        gpu_cap_buf.copy_to_host_async(host_caps.as_mut_slice(), &stream2).expect("copy caps");
        stream1.synchronize().expect("cuda sync");
        stream2.synchronize().expect("cuda sync");
        stream1.destroy().expect("cuda stream destroy");
        stream2.destroy().expect("cuda stream destroy");

        let now = Instant::now();

        if digests_buf.len() > 0 {
            host_digests
                .par_chunks_exact(4)
                .zip(digests_buf)
                .for_each(|(x, y)| {
                    unsafe {
                        let mut parts = U8U64 { f1: [0; 32] };
                        parts.f2[0] = x[0].to_canonical_u64();
                        parts.f2[1] = x[1].to_canonical_u64();
                        parts.f2[2] = x[2].to_canonical_u64();
                        parts.f2[3] = x[3].to_canonical_u64();
                        let (slice, _) = parts.f1.split_at(H::HASH_SIZE);
                        let h: H::Hash = H::Hash::from_bytes(slice);
                        y.write(h);
                    };
                });
        }

        if cap_buf.len() > 0 {
            host_caps
                .par_chunks_exact(4)
                .zip(cap_buf)
                .for_each(|(x, y)| {
                    unsafe {
                        let mut parts = U8U64 { f1: [0; 32] };
                        parts.f2[0] = x[0].to_canonical_u64();
                        parts.f2[1] = x[1].to_canonical_u64();
                        parts.f2[2] = x[2].to_canonical_u64();
                        parts.f2[3] = x[3].to_canonical_u64();
                        let (slice, _) = parts.f1.split_at(H::HASH_SIZE);
                        let h: H::Hash = H::Hash::from_bytes(slice);
                        y.write(h);
                    };
                });
        }
        print_time(now, "copy results");
    }

#[cfg(feature = "cuda")]
fn fill_digests_buf_meta<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
    cap_height: usize,
) {
    use crate::plonk::config::HasherType;

    let leaf_size = leaves[0].len();
    // if the input is small or if it Keccak hashing, just do the hashing on CPU
    if leaf_size <= H::HASH_SIZE / 8 || H::HASHER_TYPE == HasherType::Keccak {
        fill_digests_buf::<F, H>(digests_buf, cap_buf, &leaves[..], cap_height);
    } else {
        fill_digests_buf_gpu_v1::<F, H>(digests_buf, cap_buf, &leaves[..], cap_height);
    }
}

#[cfg(not(feature = "cuda"))]
fn fill_digests_buf_meta<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
    cap_height: usize,
) {
    fill_digests_buf::<F, H>(digests_buf, cap_buf, &leaves[..], cap_height);
}

impl<F: RichField, H: Hasher<F>> MerkleTree<F, H> {
    pub fn new(leaves_2d: Vec<Vec<F>>, cap_height: usize) -> Self {
        let log2_leaves_len = log2_strict(leaves_2d.len());
        assert!(
            cap_height <= log2_leaves_len,
            "cap_height={} should be at most log2(leaves.len())={}",
            cap_height,
            log2_leaves_len
        );

        let leaf_size = leaves_2d[0].len();
        let leaves_len = leaves_2d.len();

        let num_digests = 2 * (leaves_len - (1 << cap_height));
        let mut digests = Vec::with_capacity(num_digests);

        let len_cap = 1 << cap_height;
        let mut cap = Vec::with_capacity(len_cap);

        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
        let now = Instant::now();
        fill_digests_buf_meta::<F, H>(digests_buf, cap_buf, &leaves_2d[..], cap_height);
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
        let leaves_1d = leaves_2d.into_iter().flatten().collect();

        Self {
            leaves: leaves_1d,
            leaf_size,
            digests,
            cap: MerkleCap(cap),
        }
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
    pub fn new_gpu_leaves(leaves_gpu_ptr: HostOrDeviceSlice<'_, F>, leaves_len: usize, leaf_len: usize, cap_height: usize) -> Self {
        let log2_leaves_len = log2_strict(leaves_len);
        assert!(
            cap_height <= log2_leaves_len,
            "cap_height={} should be at most log2(leaves.len())={}",
            cap_height,
            log2_leaves_len
        );

        // copy data from GPU in async mode
        let start = std::time::Instant::now();
        let mut host_leaves: Vec<F> = vec![F::ZERO; leaves_len * leaf_len];
        let stream = CudaStream::create().unwrap();
        leaves_gpu_ptr.copy_to_host_async(
            host_leaves.as_mut_slice(),
            &stream
        ).expect("copy to host error");
        print_time(start, "Copy leaves from GPU");

        let num_digests = 2 * (leaves_len - (1 << cap_height));
        let mut digests = Vec::with_capacity(num_digests);

        let len_cap = 1 << cap_height;
        let mut cap = Vec::with_capacity(len_cap);

        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
        let now = Instant::now();
        fill_digests_buf_gpu_ptr::<F,H>(
            digests_buf,
            cap_buf,
            leaves_gpu_ptr.as_ptr(),
            leaves_len,
            leaf_len,
            cap_height,
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
        let _ = stream.synchronize();
        let _ = stream.destroy();

        Self {
            leaves: host_leaves,
            leaf_size: leaf_len,
            digests,
            cap: MerkleCap(cap),
        }
    }

    pub fn get(&self, i: usize) -> &[F] {
        let (_ , v) = self.leaves.split_at(i * self.leaf_size);
        let (v, _) = v.split_at(self.leaf_size);
        v
    }

    pub fn get_leaves_1D(&self) -> Vec<F> {
        self.leaves.clone()
    }

    pub fn get_leaves_2D(&self) -> Vec<Vec<F>> {
        let v2d : Vec<Vec<F>> = self.leaves.chunks_exact(self.leaf_size).map(
            |leaf| {
                leaf.to_vec()
            }
        ).collect();
        v2d
    }

    pub fn get_leaves_count(&self) -> usize {
        self.leaves.len() / self.leaf_size
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
    use crate::plonk::config::{GenericConfig, KeccakGoldilocksConfig, Poseidon2GoldilocksConfig, PoseidonGoldilocksConfig};

    fn random_data<F: RichField>(n: usize, k: usize) -> Vec<Vec<F>> {
        (0..n).map(|_| F::rand_vec(k)).collect()
    }

    fn verify_all_leaves<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        leaves: Vec<Vec<F>>,
        cap_height: usize,
    ) -> Result<()> {
        let tree = MerkleTree::<F, C::Hasher>::new(leaves.clone(), cap_height);
        for (i, leaf) in leaves.into_iter().enumerate() {
            let proof = tree.prove(i);
            verify_merkle_proof_to_cap(leaf, i, &tree.cap, &proof)?;
        }
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_cap_height_too_big() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let cap_height = log_n + 1; // Should panic if `cap_height > len_n`.

        let leaves = random_data::<F>(1 << log_n, 7);
        let _ = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new(leaves, cap_height);
    }

    #[test]
    fn test_cap_height_eq_log2_len() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, log_n)?;

        Ok(())
    }

    #[test]
    fn test_merkle_trees_poseidon_g64() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        // GPU warmup
        #[cfg(feature = "cuda")]
        let _x: HostOrDeviceSlice<'_, F> = HostOrDeviceSlice::cuda_malloc(0, 64)
            .unwrap();

        let log_n = 12;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, 1)?;

        Ok(())
    }

    #[test]
    fn test_merkle_trees_poseidon2_g64() -> Result<()> {
        const D: usize = 2;
        type C = Poseidon2GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 12;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

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
        let leaves = random_data::<F>(n, 7);

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
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, 1)?;

        Ok(())
    }
}
