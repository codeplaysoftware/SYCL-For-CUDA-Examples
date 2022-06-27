#pragma once

#include <internal/config.hpp>
#include <tools/usm_smart_ptr.hpp>

constexpr dword MD2_BLOCK_SIZE = 16;

namespace hash::internal {
    class md2_kernel;

    using namespace usm_smart_ptr;


    sycl::event launch_md2_kernel(sycl::queue &q, sycl::event e, device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch);

}