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