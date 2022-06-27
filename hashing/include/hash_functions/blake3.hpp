#pragma once

#include <internal/config.hpp>
#include <tools/usm_smart_ptr.hpp>


#include <cstddef>
#include <cstdint>

#define BLAKE3_VERSION_STRING "0.3.7"
#define BLAKE3_KEY_LEN 32
#define BLAKE3_OUT_LEN 32
#define BLAKE3_BLOCK_LEN 64
#define BLAKE3_CHUNK_LEN 1024
#define BLAKE3_MAX_DEPTH 54

// This struct is a private implementation detail. It has to be here because
// it's part of blake3_hasher below.
struct blake3_chunk_state {
    uint32_t cv[8];
    uint64_t chunk_counter;
    uint8_t buf[BLAKE3_BLOCK_LEN];
    uint8_t buf_len;
    uint8_t blocks_compressed;
    uint8_t flags;
};

struct blake3_hasher {
    uint32_t key[8];
    blake3_chunk_state chunk;
    uint8_t cv_stack_len;
    // The stack size is MAX_DEPTH + 1 because we do lazy merging. For example,
    // with 7 chunks, we have 3 entries in the stack. Adding an 8th chunk
    // requires a 4th entry, rather than merging everything down to 1, because we
    // don't know whether more input is coming. This is different from how the
    // reference implementation does things.
    uint8_t cv_stack[(BLAKE3_MAX_DEPTH + 1) * BLAKE3_OUT_LEN];
};

//const char *blake3_version();

//void blake3_hasher_init(blake3_hasher *self);

//void blake3_hasher_init_keyed(blake3_hasher *self, const uint8_t key[BLAKE3_KEY_LEN]);

//void blake3_hasher_init_derive_key(blake3_hasher *self, const char *context);

//void blake3_hasher_init_derive_key_raw(blake3_hasher *self, const void *context, size_t context_len);

//void blake3_hasher_update(blake3_hasher *self, const void *input, size_t input_len);

//void blake3_hasher_finalize(const blake3_hasher *self, uint8_t *out, size_t out_len);

//void blake3_hasher_finalize_seek(const blake3_hasher *self, uint64_t seek, uint8_t *out, size_t out_len);


#include <cassert>

#include <cstddef>
#include <cstdint>
#include <cstring>

// internal flags
enum blake3_flags {
    CHUNK_START = 1 << 0,
    CHUNK_END = 1 << 1,
    PARENT = 1 << 2,
    ROOT = 1 << 3,
    KEYED_HASH = 1 << 4,
    DERIVE_KEY_CONTEXT = 1 << 5,
    DERIVE_KEY_MATERIAL = 1 << 6,
};


// There are some places where we want a static size that's equal to the
// MAX_SIMD_DEGREE, but also at least 2.
#define MAX_SIMD_DEGREE_OR_2 (MAX_SIMD_DEGREE > 2 ? MAX_SIMD_DEGREE : 2)


namespace hash::internal {
    using namespace usm_smart_ptr;

    sycl::event
    launch_blake3_kernel(sycl::queue &item, device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch, dword n_outbit, const byte *key, dword keylen);

    int test_blake3();

}