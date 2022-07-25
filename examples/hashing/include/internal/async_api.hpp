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
 *  async_api.hpp
 *
 *  Description:
 *    Asynchronous API foir hashing functions
 **************************************************************************/
#pragma once

#include <utility>
#include "common.hpp"
#include "handle.hpp"

namespace hash {
    /**
     * Base class for hashing
     * @tparam M
     * @tparam n_outbit
     */
    template<hash::method M, int n_outbit = 0>
    class hasher {
    private:
        runners runners_;
    public:
        explicit hasher(runners v) : runners_(std::move(v)) {}

        handle hash(const byte *indata, dword inlen, byte *outdata, dword n_batch, byte *key, dword keylen) {
            size_t size = runners_.size();
            std::vector<handle_item> handles;
            handles.reserve(size);
            auto items = internal::get_hash_queue_work_item<M, n_outbit>(runners_, indata, inlen, outdata, n_batch);
            for (size_t i = 0; i < size; ++i) {
                handles.emplace_back(internal::hash_with_data_copy<M, n_outbit>(items[i], key, keylen));
            }
            return handle(std::move(handles));
        }

        handle hash(const byte *indata, dword inlen, byte *outdata, dword n_batch) {
            return hash(indata, inlen, outdata, n_batch, nullptr, 0);
        }


    };


    /**
     * Blake 2B
     * @tparam n_outbit
     */
    template<int n_outbit>
    class hasher<method::blake2b, n_outbit> {

    private:
        hash::runners runners_;
        std::vector<usm_shared_ptr < blake2b_ctx, alloc::device>> keyed_ctxts_{};
    public:
        explicit hasher(const hash::runners &v, const byte *key, dword keylen) : runners_(v) {
            size_t size = v.size();
            keyed_ctxts_.reserve(size);

            for (size_t i = 0; i < size; ++i) {
                keyed_ctxts_.emplace_back(internal::get_blake2b_ctx(runners_[i].q, key, keylen, n_outbit));
            }

        }

        handle hash(const byte *indata, dword inlen, byte *outdata, dword n_batch) {
            size_t size = runners_.size();
            std::vector<handle_item> handles;
            handles.reserve(2 * size);
            auto items = internal::get_hash_queue_work_item<method::blake2b, n_outbit>(runners_, indata, inlen, outdata, n_batch);
            for (size_t i = 0; i < size; ++i) {
                handles.emplace_back(internal::hash_with_data_copy<method::blake2b, n_outbit>(items[i], nullptr, 0, keyed_ctxts_[i].get()));
            }
            return handle(std::move(handles));
        }
    };


    using md2 = hasher<hash::method::md2>;
    using md5 = hasher<hash::method::md5>;
    using sha1 = hasher<hash::method::sha1>;
    using sha256 = hasher<hash::method::sha256>;

    template<int n_outbit>
    using keccak = hasher<hash::method::keccak, n_outbit>;

    template<int n_outbit>
    using sha3 = hasher<hash::method::sha3, n_outbit>;

    template<int n_outbit>
    using blake2b = hasher<hash::method::blake2b, n_outbit>;
}
