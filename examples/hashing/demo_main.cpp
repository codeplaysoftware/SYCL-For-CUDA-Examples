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
 *  demo_main.cpp
 *
 *  Description:
 *    Main function for hashing demo
 **************************************************************************/
/*Copyright 2021 Codeplay Software Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use these files except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

For your convenience, a copy of the License has been included in this
repository.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#include <sycl_hash.hpp>
#include <tools/sycl_queue_helpers.hpp>
#include "src/benchmarks/misc.hpp"

int main() {
    size_t input_block_size = 512 * 1024; //bytes
    size_t n_blocs = 1024 * 6;
    size_t n_iters = 40;
    auto cpu_q = try_get_queue(sycl::cpu_selector{});
    auto cuda_q = try_get_queue(cuda_selector{});

//    auto ptr = (byte *) malloc(input_block_size * 100 * sizeof(byte));
//    auto out = (byte *) malloc(hash::get_block_size<hash::method::sha256>() * 100 * sizeof(byte));
//    double cpu_speed = benchmark_one_queue<hash::method::sha256>(cpu_q, input_block_size, 80);
//    double gpu_speed = benchmark_one_queue<hash::method::sha256>(cuda_q, input_block_size, n_blocs, 5);
//    hash::sha256 hasher({{cpu_q,  cpu_speed}, {cuda_q, gpu_speed}});
//    auto e = hasher.hash(ptr, input_block_size, out, 100);
//    hash::compute_md2(cpu_q, ptr, input_block_size, out, n_blocs);
//    hash::compute_sha3<512>(cpu_q, ptr, input_block_size, out, n_blocs);


    //GPU
    run_benchmark<hash::method::keccak, 128>(cuda_q, input_block_size, n_blocs, n_iters);
    run_benchmark<hash::method::sha3, 256>(cuda_q, input_block_size, n_blocs, n_iters);
    run_benchmark<hash::method::md5>(cuda_q, input_block_size, n_blocs, n_iters);
    run_benchmark<hash::method::blake2b, 128>(cuda_q, input_block_size, n_blocs, n_iters);
    run_benchmark<hash::method::sha1>(cuda_q, input_block_size, n_blocs, n_iters);
    run_benchmark<hash::method::sha256>(cuda_q, input_block_size, n_blocs, n_iters);
    run_benchmark<hash::method::md2>(cuda_q, input_block_size, n_blocs, n_iters);

    //CPU
    run_benchmark<hash::method::keccak, 128>(cpu_q, input_block_size, n_blocs, n_iters);
    run_benchmark<hash::method::md5>(cpu_q, input_block_size, n_blocs, n_iters);
    run_benchmark<hash::method::blake2b, 128>(cpu_q, input_block_size, n_blocs, n_iters);
    run_benchmark<hash::method::sha1>(cpu_q, input_block_size, n_blocs, n_iters);
    run_benchmark<hash::method::sha256>(cpu_q, input_block_size, n_blocs, n_iters);
    run_benchmark<hash::method::md2>(cpu_q, input_block_size, n_blocs, n_iters);


    // CPU == GPU ??
    compare_two_devices<hash::method::sha256>(cuda_q, cpu_q, 1024, 4096);
    compare_two_devices<hash::method::keccak, 128>(cuda_q, cpu_q, 1024, 4096);
    compare_two_devices<hash::method::md2>(cuda_q, cpu_q, 1024, 4096);
    compare_two_devices<hash::method::md5>(cuda_q, cpu_q, 1024, 4096);
    compare_two_devices<hash::method::sha1>(cuda_q, cpu_q, 1024, 4096);
    compare_two_devices<hash::method::blake2b, 128>(cuda_q, cpu_q, 1024, 4096);
}

