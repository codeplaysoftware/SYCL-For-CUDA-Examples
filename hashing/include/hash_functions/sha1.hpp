#pragma once

#include <internal/config.hpp>
#include <tools/usm_smart_ptr.hpp>


constexpr dword SHA1_BLOCK_SIZE = 20;

namespace hash::internal {
    class sha1_kernel;

    using namespace usm_smart_ptr;

    sycl::event launch_sha1_kernel(sycl::queue &q, sycl::event e, device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch);

}