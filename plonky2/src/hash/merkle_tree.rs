use alloc::sync::Arc;
use alloc::vec::Vec;
use num::range;
use once_cell::sync::Lazy;
use core::mem::MaybeUninit;
use core::slice;
use std::sync::Mutex;
use std::time::Instant;

use plonky2_maybe_rayon::*;
use serde::{Deserialize, Serialize};

use crate::{fill_delete, get_digests_ptr, get_leaves_ptr, get_cap_ptr, fill_init,  fill_digests_buf_linear_gpu};
use crate::hash::hash_types::RichField;
use crate::hash::merkle_proofs::MerkleProof;
use crate::plonk::config::{GenericHashOut, Hasher};
use crate::util::log2_strict;

static gpu_lock: Lazy<Arc<Mutex<i32>>> = Lazy::new(|| Arc::new(Mutex::new(0)));

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

    // if one leaf => return it hash
    if leaves.len() == 1 {
        let hash = H::hash_or_noop(&leaves[0]);
        digests_buf[0].write(hash);
        return hash
    }
    // if two leaves => return their concat hash
    if leaves.len() == 2 {
        let hash_left = H::hash_or_noop(&leaves[0]);
        let hash_right = H::hash_or_noop(&leaves[1]);
        digests_buf[0].write(hash_left);
        digests_buf[1].write(hash_right);
        return H::two_to_one(hash_left, hash_right)
    }

    assert_eq!(leaves.len(), digests_buf.len() / 2 + 1);

    // leaves first - we can do all in parallel
    let (_, digests_leaves) = digests_buf.split_at_mut(digests_buf.len() - leaves.len());
    digests_leaves.into_par_iter().zip(leaves).for_each(|(digest, leaf)| {
        digest.write(H::hash_or_noop(leaf));
    });
    
    // internal nodes - we can do in parallel per level    
    let mut last_index = digests_buf.len() - leaves.len();

    log2_strict(leaves.len());
    for level_log in range(1, log2_strict(leaves.len())).rev() {        
        let level_size = 1 << level_log;
        // println!("Size {} Last index {}", level_size, last_index);
        let (_, digests_slice) = digests_buf.split_at_mut(last_index - level_size);
        let (digests_slice, next_digests) = digests_slice.split_at_mut(level_size);

        digests_slice.into_par_iter().zip(last_index - level_size .. last_index).for_each(|(digest, idx)| {
            let left_idx = 2 * (idx + 1) - last_index;
            let right_idx = left_idx + 1;
            
            unsafe {
                let left_digest = next_digests[left_idx].assume_init();
                let right_digest = next_digests[right_idx].assume_init();
                digest.write(H::two_to_one(left_digest, right_digest));
                // println!("Size {} Index {} {:?} {:?}", level_size, idx, left_digest, right_digest);
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
}

#[repr(C)]
union U8U64 {
    f1: [u8; 32],
    f2: [u64; 4],
}

fn fill_digests_buf_gpu<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
    cap_height: usize,
) {
    let digests_count: u64 = digests_buf.len().try_into().unwrap();
    let leaves_count: u64 = leaves.len().try_into().unwrap();
    let caps_count: u64 = cap_buf.len().try_into().unwrap();
    let cap_height: u64  = cap_height.try_into().unwrap();
    let leaf_size: u64 = leaves[0].len().try_into().unwrap();
    let hash_size: u64 = H::HASH_SIZE.try_into().unwrap();

    let _lock = gpu_lock.lock().unwrap();

    unsafe {
        fill_init(digests_count, leaves_count, caps_count, leaf_size, hash_size);        
    
        // copy data to C
        let mut pd : *mut u64 = get_digests_ptr();
        let mut pl : *mut u64 = get_leaves_ptr();
        let mut pc : *mut u64 = get_cap_ptr();
                
        for leaf in leaves {
            for elem in leaf {
                let val = &elem.to_canonical_u64();
                std::ptr::copy(val, pl, 8);
                pl = pl.add(1);                
            }
        }

        let now = Instant::now();
        // println!("Digest size {}, Leaves {}, Leaf size {}, Cap H {}", digests_count, leaves_count, leaf_size, cap_height);
        fill_digests_buf_linear_gpu(digests_count, caps_count, leaves_count, leaf_size, cap_height);
        println!("Time to fill digests in C on GPU: {} ms", now.elapsed().as_millis());

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
        for dg in digests_buf {
            let mut parts = U8U64 {f1: [0; 32]};
            std::ptr::copy(pd, parts.f2.as_mut_ptr(), H::HASH_SIZE);
            let h : H::Hash = H::Hash::from_bytes(&parts.f1);
            dg.write(h);
            pd = pd.add(4);
        }        
        for cp in cap_buf {
            let mut parts = U8U64 {f1: [0; 32]};
            std::ptr::copy(pc, parts.f2.as_mut_ptr(), H::HASH_SIZE);
            let h : H::Hash = H::Hash::from_bytes(&parts.f1);
            cp.write(h);
            pc = pc.add(4);
        }

        fill_delete();
    }
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
        let leaf_size = leaves[0].len();

        let num_digests = 2 * (leaves.len() - (1 << cap_height));
        let mut digests = Vec::with_capacity(num_digests);

        let len_cap = 1 << cap_height;
        let mut cap = Vec::with_capacity(len_cap);

        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
        // TODO ugly way: if it is 25, it is Keccak
        if H::HASH_SIZE == 25 || leaf_size <= H::HASH_SIZE / 8 {
            fill_digests_buf::<F, H>(digests_buf, cap_buf, &leaves[..], cap_height);
        }
        else {
            fill_digests_buf_gpu::<F, H>(digests_buf, cap_buf, &leaves[..], cap_height);
        }

        unsafe {
            // SAFETY: `fill_digests_buf` and `cap` initialized the spare capacity up to
            // `num_digests` and `len_cap`, resp.
            digests.set_len(num_digests);
            cap.set_len(len_cap);
        }

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
        let subtree_digest_size = (1 << (num_layers + 1)) - 2;      // 2 ^ (k+1) - 2
        let subtree_idx = leaf_index / (1 << num_layers);

        let siblings: Vec<<H as Hasher<F>>::Hash> = Vec::with_capacity(num_layers);
        if num_layers == 0 {
            return MerkleProof {siblings}
        }

        // digests index where we start
        let idx = subtree_digest_size - (1 << num_layers) + (leaf_index % (1 << num_layers));               

        let siblings = (0..num_layers)
            .map(|i| {
                // relative index
                let rel_idx = (idx + 2 - (1 << i+1)) / (1 << i);
                // absolute index
                let mut abs_idx = subtree_idx * subtree_digest_size + rel_idx;
                if (rel_idx & 1) == 1 {
                    abs_idx -= 1;
                }
                else {
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
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

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
    fn test_merkle_trees() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, 1)?;

        Ok(())
    }
}
