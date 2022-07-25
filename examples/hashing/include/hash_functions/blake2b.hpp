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
 *  blake2b.hpp
 *
 *  Description:
 *    Blake2 hash function
 **************************************************************************/
#pragma once

#include <internal/config.hpp>
#include <tools/usm_smart_ptr.hpp>

constexpr dword BLAKE2B_ROUNDS = 12;
constexpr dword BLAKE2B_BLOCK_LENGTH = 128;
constexpr dword BLAKE2B_CHAIN_SIZE = 8;
constexpr dword BLAKE2B_CHAIN_LENGTH = (BLAKE2B_CHAIN_SIZE * sizeof(qword));
constexpr dword BLAKE2B_STATE_SIZE = 16;
constexpr dword BLAKE2B_STATE_LENGTH = (BLAKE2B_STATE_SIZE * sizeof(qword));

struct blake2b_ctx {
    int64_t digestlen{};
    dword keylen{};
    dword pos{};
    qword t0{};
    qword t1{};
    qword f0{};
    byte buff[BLAKE2B_BLOCK_LENGTH] = {0};
    qword chain[BLAKE2B_CHAIN_SIZE] = {0};
    qword state[BLAKE2B_STATE_SIZE] = {0};
};

namespace hash::internal {
    class blake2b_kernel;

    using namespace usm_smart_ptr;

    usm_shared_ptr<blake2b_ctx, alloc::device> get_blake2b_ctx(sycl::queue &q, const byte *key, dword keylen, dword n_outbit);


    sycl::event
    launch_blake2b_kernel(sycl::queue &item, sycl::event e, device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch, dword n_outbit, const byte *key,
                          dword keylen);

    sycl::event
    launch_blake2b_kernel(sycl::queue &item, sycl::event e, device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch, dword n_outbit, const byte *key,
                          dword keylen, device_accessible_ptr<blake2b_ctx>);

}