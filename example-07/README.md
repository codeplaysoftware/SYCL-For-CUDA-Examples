# SYCL Hashing Algorithms

This repository contains hashing algorithms implemented using [SYCL](https://www.khronos.org/sycl/) which is a heterogeneous programming model based on standard C++.

The following hashing methods are currently available:

- sha256
- sha1 (unsecure)
- md2 (unsecure)
- md5 (unsecure)
- keccak (128 224 256 288 384 512)
- sha3 (224 256 384 512)
- blake2b

## Benchmarks

Some functions were ported from a CUDA implementation. The SYCL code was tested unchanged across the different implementations and hardware. Here's how they perform (the values are in GB/s):

| Function | Native CUDA | SYCL on CUDA (optimised/original)           | SYCL on ComputeCPP CPU (spir64/spirv64) | SYCL on DPC++ CPU (spir64_x86_64) | SYCL on hipSYCL (omp/cuda) |
| -------- | ----------- | ------------------------------------------- | --------------------------------------- | --------------------------------- | -------------------------- |
| keccak   | 15.7        | 23.0                                        | 4.14 / 4.08                             | 4.98                              | 4.32 / 23.0                |
| md5      | 14.6        | 20.3                                        | 6.26 / 8.70                             | 9.93                              | 9.27 / 20.2                |
| blake2b  | 14.7        | 21.6 / 18.6                                 | 9.46 / 9.46                             | 12.4                              | 7.71 / 17.8                |
| sha1     | 13.1        | 19.34 / 14.9                                | 3.61 / 2.59                             | 3.30                              | 4.39 / 19.2                |
| sha256   | 13.4        | 19.15 / 13.6                                | 2.23 / 1.74                             | 2.91                              | 2.93 / 19.0                |
| md2      | 4.18        | 4.23/ 2.40                                  | 0.22 / 0.25                             | 0.176                             | 0.25 / 2.33                |

### Note
Something broke the spir64 backend of DPC++ and it produces now very slow code

Benchmark configuration:

- block_size: 512 kiB
- n_blocks: 4\*1536
- n_outbit: 128
- GPU: GTX 1660 Ti
- OS: rhel8.4
- CPU: 2x E5-2670 v2

### Remark

These are not the "best" settings as the optimum changes with the algorithm. The benchmarks measure the time to run 40 iterations, without copying the memory between the device and the host. In a real application, you could be memory bound.

## How to build

```bash
git clone https://github.com/Michoumichmich/SYCL-Hashing-Algorithms.git ; cd SYCL-Hashing-Algorithms;
mkdir build; cd build
CXX=<sycl_compiler> cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

This will build the library, and a demo executable. Running it will perform a benchmark on your CPU and CUDA device (if available).

You do not necessarily need to pass the `<sycl_compiler>` to cmake, it depends on the implementation you're using and its toolchain.

## How to use

Let's assume you used this [script](https://github.com/Michoumichmich/oneAPI-setup-script) to setup the toolchain with CUDA support.

Here's a minimal example:

```C++
#include <sycl/sycl.hpp> // SYCL headers
#include "sycl_hash.hpp" // The headers
#include "tools/sycl_queue_helpers.hpp" // To make sycl queue
using namespace hash;

int main(){
    auto cuda_q = try_get_queue(cuda_selector{}); // create a queue on a cuda device and attach an exception handler

    constexpr int hash_size = get_block_size<method::sha256>();
    constexpr int n_blocks = 20; // amount of hash to do in parallel
    constexpr int item_size = 1024;

    byte input[n_blocks * item_size]; // get an array of 20 same-sized data items to hash;
    byte output[n_blocks * hash_size]; // reserve space for the output

    compute<method::sha256>(cuda_q, input, item_size, output, n_blocks); // do the computing
    compute_sha256(cuda_q, input, item_size, output, n_blocks); // identical

    /**
     * For SHA3 one could write:
     * compute_sha3<512>(cuda_q, input, item_size, output, n_blocks);
     */

    return 0;
}
```

And, for clang build with

```
-fsycl -fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda--sm_50 -I<include_dir> <build_dir>/libsycl_hash.a
```

And your hash will run on the GPU.

# Sources

You may find [here](https://github.com/Michoumichmich/cuda-hashing-algos-with-benchmark) the fork of the original CUDA implementations with the benchmarks added.

# Tested implementations

- [Intel's clang](https://github.com/intel/llvm) with OpenCL on CPU (using Intel's driver) and [Codeplay's CUDA backend](https://www.codeplay.com/solutions/oneapi/for-cuda/)
- [hipSYCL](https://github.com/illuhad/hipSYCL) on macOS with the OpenMP backend (set `hipSYCL_DIR` then `cmake .. -DHIPSYCL_TARGETS="..."`)
- [ComputeCPP](https://developer.codeplay.com/products/computecpp/ce/home) you can build with `cmake .. -DComputeCpp_DIR=/path_to_computecpp -DCOMPUTECPP_BITCODE=spir64 -DCMAKE_BUILD_TYPE=Release`, Tested on the host device, `spir64` and `spirv64`. See [ComputeCpp SDK](https://github.com/codeplaysoftware/computecpp-sdk)

# Acknowledgements

This repository contains code written by Matt Zweil & The Mochimo Core Contributor Team. Please see the [files](https://github.com/mochimodev/cuda-hashing-algos) for their respective licences.
