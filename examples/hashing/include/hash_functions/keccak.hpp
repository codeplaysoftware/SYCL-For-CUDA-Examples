#pragma once

#include <internal/config.hpp>
#include <tools/usm_smart_ptr.hpp>


constexpr dword KECCAK_ROUND = 24;
constexpr dword KECCAK_STATE_SIZE = 25;
constexpr dword KECCAK_Q_SIZE = 192;

namespace hash::internal {

    template<dword n_outbit>
    class keccak_kernel;

    using namespace usm_smart_ptr;


    sycl::event
    launch_keccak_kernel(bool is_sha3, sycl::queue &item, sycl::event e, device_accessible_ptr<byte> indata, device_accessible_ptr<byte> outdata, dword inlen, dword n_batch, dword n_outbit);


}