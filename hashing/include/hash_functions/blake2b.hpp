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