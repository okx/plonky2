#ifndef __MERKLE_H__
#define __MERKLE_H__

#include <stdint.h>

#define HASH_SIZE   32

#define HASH_SIZE_U64   4

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

enum HashType {
    HashPoseidon = 0,
    HashKeccak = 1,
    HashPoseidonBN128 = 2,
    HashPoseidon2 = 3,
    HashMonolith = 4
};

EXTERNC void fill_digests_buf_linear_cpu_avx512(
    void *digests_buf_ptr,
    void *cap_buf_ptr,
    void *leaves_buf_ptr,
    uint64_t digests_buf_size,
    uint64_t cap_buf_size,
    uint64_t leaves_buf_size,
    uint64_t leaf_size,
    uint64_t cap_height,
    uint64_t hash_type
);

#endif // __MERKEL_H__
