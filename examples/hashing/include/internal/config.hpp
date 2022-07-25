/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's SYCL-For-CUDA-Examples
 *
 *  config.hpp
 *
 *  Description:
 *    Hashing function configuration
 **************************************************************************/
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