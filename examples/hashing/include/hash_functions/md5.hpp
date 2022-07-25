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
 *  md5.hpp
 *
 *  Description:
 *    MD5 hash function
 **************************************************************************/
#pragma once

#include <internal/config.hpp>
#include <tools/usm_smart_ptr.hpp>

/****************************** MACROS ******************************/
constexpr dword MD5_BLOCK_SIZE = 16;            // MD5 outputs a 16 byte digest

namespace hash::internal {
    class md5_kernel;

    using namespace usm_smart_ptr;

    sycl::event launch_md5_kernel(sycl::queue &q, sycl::event e, device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch);

}