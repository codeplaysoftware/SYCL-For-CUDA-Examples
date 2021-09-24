#pragma once

#include <cstring>
#include <sycl_hash.hpp>
#include <tools/fill_rand.hpp>
#include <tools/chrono.hpp>

#include <iomanip>

using namespace usm_smart_ptr;


template<hash::method M, int ... args>
void compare_two_devices(sycl::queue q1, sycl::queue q2, size_t input_block_size, size_t n_blocs) {
    std::cout << "Comparing " << hash::get_name<M, args...>() << " on: " << q1.get_device().get_info<sycl::info::device::name>() << " and: " << q2.get_device().get_info<sycl::info::device::name>()
              << "   ...";
    size_t out_block_size = hash::get_block_size<M, args...>();
    auto input_data1 = usm_unique_ptr<byte, alloc::shared>(input_block_size * n_blocs, q1);
    auto output_hashes1 = usm_unique_ptr<byte, alloc::shared>(out_block_size * n_blocs, q1);
    auto input_data2 = usm_unique_ptr<byte, alloc::shared>(input_block_size * n_blocs, q2);
    auto output_hashes2 = usm_unique_ptr<byte, alloc::shared>(out_block_size * n_blocs, q2);

    fill_rand<byte>(input_data1.get(), input_data1.alloc_count());
    memcpy(input_data2.raw(), input_data1.raw(), input_data1.alloc_size());


    if constexpr (M == hash::method::blake2b) {
        byte key[64];
        std::memset(key, 1, 64);
        hash::compute<M, args...>(q1, input_data1.get(), input_block_size, output_hashes1.get(), n_blocs, key, 64);
        hash::compute<M, args...>(q2, input_data2.get(), input_block_size, output_hashes2.get(), n_blocs, key, 64);
    } else {
        hash::compute<M, args...>(q1, input_data1.get(), input_block_size, output_hashes1.get(), n_blocs);
        hash::compute<M, args...>(q2, input_data2.get(), input_block_size, output_hashes2.get(), n_blocs);
    }

    auto[idx1, idx2]= std::mismatch(output_hashes1.raw(), output_hashes1.raw() + output_hashes1.alloc_count(), output_hashes2.raw());
    if (idx1 != output_hashes1.raw() + output_hashes1.alloc_count()) {
        std::cout << "mismatch" << std::endl;
    } else {
        std::cout << "pass" << std::endl;
    }
}


template<hash::method M, int ... args>
double benchmark_one_queue(sycl::queue q, size_t input_block_size, size_t n_blocs, size_t n_iters = 1) {
    auto all_input_data = usm_unique_ptr<byte, alloc::device>(input_block_size * n_blocs, q);
    auto all_output_hashes = usm_unique_ptr<byte, alloc::device>(hash::get_block_size<M, args...>() * n_blocs, q);
    if constexpr (M == hash::method::blake2b) {
        byte key[64];
        std::memset(key, 1, 64);
        hash::compute<M, args...>(q, all_input_data.get(), input_block_size, all_output_hashes.get(), n_blocs, key, 64);/* Preheat */
        auto before = std::chrono::steady_clock::now();
        for (size_t i = 0; i < n_iters; ++i) {
#ifdef VERBOSE_HASH_LIB
            std::cerr << i << "  ";
#endif
            hash::compute<M, args...>(q, all_input_data.get(), input_block_size, all_output_hashes.get(), n_blocs, key, 64);
        }
        auto after = std::chrono::steady_clock::now();
        auto time = std::chrono::duration<double, std::milli>(after - before).count();
        return (double) n_iters / time * (double) (input_block_size * n_blocs) / 1e6;
    } else {
        hash::compute<M, args...>(q, all_input_data.get(), input_block_size, all_output_hashes.get(), n_blocs);/* Preheat */
        auto before = std::chrono::steady_clock::now();
        for (size_t i = 0; i < n_iters; ++i) {
#ifdef VERBOSE_HASH_LIB
            std::cerr << i << "  ";
#endif
            hash::compute<M, args...>(q, all_input_data.get(), input_block_size, all_output_hashes.get(), n_blocs);
        }
        auto after = std::chrono::steady_clock::now();
        auto time = std::chrono::duration<double, std::milli>(after - before).count();
        return (double) n_iters / time * (double) (input_block_size * n_blocs) / 1e6;
    }
}


template<hash::method M, int ... args>
void run_benchmark(sycl::queue q, size_t input_block_size, size_t n_blocs, size_t n_iters) {
    std::cout << "Running " << hash::get_name<M, args...>() << " on:" << q.get_device().get_info<sycl::info::device::name>() << ": ";
    auto gflops = benchmark_one_queue<M, args...>(q, input_block_size, n_blocs, n_iters);
    std::cout << "\nGB hashed per sec: " << gflops << "\n\n";
}
