#pragma once

#include <cstdint>
#include <vector>
#include <sycl/sycl.hpp>

/**
 * To update on every abi update so two you won't be able to link the new declarations against an older library.
 */
#define abi_rev v_1

using byte = uint8_t;
using dword = uint32_t;
using qword = uint64_t;

//#define IMPLICIT_MEMORY_COPY 1 // ONLY ON LINUX AND MACOS

namespace hash {
    /**
     * Defines the various types of hashes supported.
     */
    enum class method {
        sha256,
        keccak,
        blake2b,
        sha1,
        sha3,
        md5,
        md2
    };


}