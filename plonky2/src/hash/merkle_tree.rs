#[cfg(feature = "cuda")]
use alloc::sync::Arc;
use alloc::vec::Vec;
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
use crate::{
    fill_delete, fill_delete_rounds, fill_digests_buf_in_rounds_in_c_on_gpu,
    fill_digests_buf_in_rounds_in_c_on_gpu_with_gpu_ptr, fill_init, fill_init_rounds, get_cap_ptr,
    get_digests_ptr, get_leaves_ptr,
};

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
    pub leaves: Vec<Vec<F>>,

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
    assert_eq!(leaves.len(), digests_buf.len() / 2 + 1);
    if digests_buf.is_empty() {
        H::hash_or_noop(&leaves[0])
    } else {
        // Layout is: left recursive output || left child digest
        //             || right child digest || right recursive output.
        // Split `digests_buf` into the two recursive outputs (slices) and two child digests
        // (references).
        let (left_digests_buf, right_digests_buf) = digests_buf.split_at_mut(digests_buf.len() / 2);
        let (left_digest_mem, left_digests_buf) = left_digests_buf.split_last_mut().unwrap();
        let (right_digest_mem, right_digests_buf) = right_digests_buf.split_first_mut().unwrap();
        // Split `leaves` between both children.
        let (left_leaves, right_leaves) = leaves.split_at(leaves.len() / 2);

        let (left_digest, right_digest) = plonky2_maybe_rayon::join(
            || fill_subtree::<F, H>(left_digests_buf, left_leaves),
            || fill_subtree::<F, H>(right_digests_buf, right_leaves),
        );

        left_digest_mem.write(left_digest);
        right_digest_mem.write(right_digest);
        H::two_to_one(left_digest, right_digest)
    }
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
    let n_rounds: u64 = log2_strict(leaves.len()).try_into().unwrap();

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

        fill_init_rounds(leaves_count, n_rounds);

        // copy data to C
        let mut pd: *mut u64 = get_digests_ptr();
        let mut pl: *mut u64 = get_leaves_ptr();
        let mut pc: *mut u64 = get_cap_ptr();
        print_time(now, "Fill init");
        let now = Instant::now();

        /*
         * Note: std::ptr::copy(val, pl, 8); does not
         * work in "release" mode: it produces sigsegv. Hence, we replaced it with
         * manual copy.
         */
        for leaf in leaves {
            for elem in leaf {
                let val = &elem.to_canonical_u64();
                *pl = *val;
                pl = pl.add(1);
            }
        }
        print_time(now, "copy to C");
        let now = Instant::now();

        // let now = Instant::now();
        // println!("Digest size {}, Leaves {}, Leaf size {}, Cap H {}", digests_count, leaves_count, leaf_size, cap_height);
        // fill_digests_buf_in_c(digests_count, caps_count, leaves_count, leaf_size, cap_height);
        // fill_digests_buf_in_rounds_in_c(digests_count, caps_count, leaves_count, leaf_size, cap_height);
        // println!("Time to fill digests in C: {} ms", now.elapsed().as_millis());

        fill_digests_buf_in_rounds_in_c_on_gpu(
            digests_count,
            caps_count,
            leaves_count,
            leaf_size,
            cap_height,
        );
        // println!("Time to fill digests in C on GPU: {} ms", now.elapsed().as_millis());
        print_time(now, "kernel");
        let now = Instant::now();

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

        // copy data from C
        /*
         * Note: std::ptr::copy(pd, parts.f2.as_mut_ptr(), H::HASH_SIZE); does not
         * work in "release" mode: it produces sigsegv. Hence, we replaced it with
         * manual copy.
         */
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

        fill_delete_rounds();
        fill_delete();
        print_time(now, "free")
    }
}

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

    let leaves1 = leaves.to_vec().into_iter().flatten().collect::<Vec<F>>();
    let _ = gpu_leaves_buf.copy_from_host(leaves1.as_slice());
    
    // The code below copies in parallel to offsets - however, it is 2X slower than the code above
    /*
    let ls = leaves[0].len();
    leaves.into_par_iter().enumerate().for_each(
      |(i, x)| {
        let _ = gpu_leaves_buf.copy_from_host_offset(x.as_slice(), i * ls, ls);
      }  
    );
    */

    print_time(now, "data copy to gpu");
    let now = Instant::now();

    unsafe {
        fill_digests_buf_in_rounds_in_c_on_gpu_with_gpu_ptr(
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
fn fill_digests_buf_meta<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
    cap_height: usize,
) {
    use crate::plonk::config::HasherType;

    let leaf_size = leaves[0].len();
    // if the input is small, just do the hashing on CPU
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
    pub fn new(leaves: Vec<Vec<F>>, cap_height: usize) -> Self {
        let log2_leaves_len = log2_strict(leaves.len());
        assert!(
            cap_height <= log2_leaves_len,
            "cap_height={} should be at most log2(leaves.len())={}",
            cap_height,
            log2_leaves_len
        );

        let num_digests = 2 * (leaves.len() - (1 << cap_height));
        let mut digests = Vec::with_capacity(num_digests);

        let len_cap = 1 << cap_height;
        let mut cap = Vec::with_capacity(len_cap);

        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
        fill_digests_buf_meta::<F, H>(digests_buf, cap_buf, &leaves[..], cap_height);

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
            leaves,
            digests,
            cap: MerkleCap(cap),
        }
    }

    pub fn get(&self, i: usize) -> &[F] {
        &self.leaves[i]
    }

    /// Create a Merkle proof from a leaf index.
    pub fn prove(&self, leaf_index: usize) -> MerkleProof<F, H> {
        let cap_height = log2_strict(self.cap.len());
        let num_layers = log2_strict(self.leaves.len()) - cap_height;
        debug_assert_eq!(leaf_index >> (cap_height + num_layers), 0);

        let digest_tree = {
            let tree_index = leaf_index >> num_layers;
            let tree_len = self.digests.len() >> cap_height;
            &self.digests[tree_len * tree_index..tree_len * (tree_index + 1)]
        };

        // Mask out high bits to get the index within the sub-tree.
        let mut pair_index = leaf_index & ((1 << num_layers) - 1);
        let siblings = (0..num_layers)
            .map(|i| {
                let parity = pair_index & 1;
                pair_index >>= 1;

                // The layers' data is interleaved as follows:
                // [layer 0, layer 1, layer 0, layer 2, layer 0, layer 1, layer 0, layer 3, ...].
                // Each of the above is a pair of siblings.
                // `pair_index` is the index of the pair within layer `i`.
                // The index of that the pair within `digests` is
                // `pair_index * 2 ** (i + 1) + (2 ** i - 1)`.
                let siblings_index = (pair_index << (i + 1)) + (1 << i) - 1;
                // We have an index for the _pair_, but we want the index of the _sibling_.
                // Double the pair index to get the index of the left sibling. Conditionally add `1`
                // if we are to retrieve the right sibling.
                let sibling_index = 2 * siblings_index + (1 - parity);
                digest_tree[sibling_index]
            })
            .collect();

        MerkleProof { siblings }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use anyhow::Result;

    use super::*;
    use crate::field::extension::Extendable;
    use crate::hash::merkle_proofs::verify_merkle_proof_to_cap;
    use crate::plonk::config::{GenericConfig, KeccakGoldilocksConfig, PoseidonGoldilocksConfig};

    fn random_data<F: RichField>(n: usize, k: usize) -> Vec<Vec<F>> {
        (0..n).map(|_| F::rand_vec(k)).collect()
    }

    const test_leaves: [u64; 28] = [
        12382199520291307008,
        18193113598248284716,
        17339479877015319223,
        10837159358996869336,
        9988531527727040483,
        5682487500867411209,
        13124187887292514366,
        8395359103262935841,
        1377884553022145855,
        2370707998790318766,
        3651132590097252162,
        1141848076261006345,
        12736915248278257710,
        9898074228282442027,
        10465118329878758468,
        5866464242232862106,
        15506463679657361352,
        18404485636523119190,
        15311871720566825080,
        5967980567132965479,
        14180845406393061616,
        15480539652174185186,
        5454640537573844893,
        3664852224809466446,
        5547792914986991141,
        5885254103823722535,
        6014567676786509263,
        11767239063322171808,
    ];

    fn test_data<F: RichField>(n: usize, k: usize) -> Vec<Vec<F>> {
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let mut elem = Vec::with_capacity(k);
            for j in 0..k {
                elem.push(F::from_canonical_u64(test_leaves[i * k + j]));
            }
            data.push(elem);
        }
        data
    }

    fn verify_all_leaves<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        leaves: Vec<Vec<F>>,
        cap_height: usize,
    ) -> Result<()> {
        let now = Instant::now();
        let tree = MerkleTree::<F, C::Hasher>::new(leaves.clone(), cap_height);
        println!(
            "Time to build Merkle tree with {} leaves: {} ms",
            leaves.len(),
            now.elapsed().as_millis()
        );
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
    fn test_merkle_trees_poseidon() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 12;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);
        // let leaves = test_data(n, 7);        
        
        verify_all_leaves::<F, C, D>(leaves, 1)?;       

        Ok(())
    }

    #[test]
    fn test_merkle_trees_keccak() -> Result<()> {
        const D: usize = 2;
        type C = KeccakGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 14;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);
        // let leaves = test_data(n, 7);
        
        verify_all_leaves::<F, C, D>(leaves, 1)?;

        Ok(())
    }
}
