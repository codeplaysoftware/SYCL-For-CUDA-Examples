/**
 * SYCL FOR CUDA : Vector Addition Example
 *
 * Copyright 2020 Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 * @File: vector_addition.cpp
 */

#include <algorithm>
#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

int main(int argc, char *argv[]) {
  constexpr const size_t N = 100000;
  const sycl::range VecSize{N};

  sycl::buffer<double> bufA{VecSize};
  sycl::buffer<double> bufB{VecSize};
  sycl::buffer<double> bufC{VecSize};

  // Initialize input data
  {
    sycl::host_accessor h_a{bufA, sycl::write_only};
    sycl::host_accessor h_b{bufB, sycl::write_only};

    for (int i = 0; i < N; i++) {
      h_a[i] = sin(i) * sin(i);
      h_b[i] = cos(i) * cos(i);
    }
  }

  auto CUDASelector = [](sycl::device const &dev) {
    if (dev.get_platform().get_backend() == sycl::backend::ext_oneapi_cuda) {
      std::cout << " CUDA device found " << std::endl;
      return 1;
    } else {
      return -1;
    }
  };
  sycl::queue myQueue{CUDASelector};

  // Command Group creation
  auto cg = [&](sycl::handler &h) {
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;

    auto a = bufA.get_access<read_t>(h);
    auto b = bufB.get_access<read_t>(h);
    auto c = bufC.get_access<write_t>(h);

    h.parallel_for(VecSize, [=](sycl::id<1> i) { c[i] = a[i] + b[i]; });
  };

  myQueue.submit(cg);

  {
    sycl::host_accessor h_c{bufC, sycl::read_only};

    double sum = 0.0f;
    for (int i = 0; i < N; i++) {
      sum += h_c[i];
    }
    std::cout << "Sum is : " << sum << std::endl;
  }

  return 0;
}
