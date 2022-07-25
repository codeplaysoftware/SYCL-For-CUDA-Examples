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
 *  sha1.hpp
 *
 *  Description:
 *    SHA1 hash function
 **************************************************************************/
#pragma once

#include <internal/config.hpp>
#include <tools/usm_smart_ptr.hpp>


constexpr dword SHA1_BLOCK_SIZE = 20;

namespace hash::internal {
    class sha1_kernel;

    using namespace usm_smart_ptr;

    sycl::event launch_sha1_kernel(sycl::queue &q, sycl::event e, device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch);

}