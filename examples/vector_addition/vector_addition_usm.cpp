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

class CUDASelector : public sycl::device_selector {
public:
  int operator()(const sycl::device &device) const override {
    if(device.get_platform().get_backend() == sycl::backend::ext_oneapi_cuda){
      std::cout << " CUDA device found " << std::endl;
      return 1;
    } else{
      return -1;
    }
  }
};

int main(int argc, char *argv[]) {
  constexpr const size_t n = 100000;

  // Create a sycl queue with our CUDASelector
  sycl::queue myQueue{CUDASelector()};

  // Host input vectors
  double *h_a;
  double *h_b;
  // Host output vector
  double *h_c;

  // Device input vectors
  double *d_a;
  double *d_b;
  // Device output vector
  double *d_c;

  // Size, in bytes, of each vector
  size_t bytes = n * sizeof(double);

  // Allocate memory for each vector on host
  h_a = (double *)malloc(bytes);
  h_b = (double *)malloc(bytes);
  h_c = (double *)malloc(bytes);

  // Allocate memory for each vector on GPU
  d_a = sycl::malloc_device<double>(n, myQueue);
  d_b = sycl::malloc_device<double>(n, myQueue);
  d_c = sycl::malloc_device<double>(n, myQueue);

  // Initialize vectors on host
  for (int i = 0; i < n; i++) {
    h_a[i] = sin(i) * sin(i);
    h_b[i] = cos(i) * cos(i);
  }

  myQueue.memcpy(d_a, h_a, bytes).wait();
  myQueue.memcpy(d_b, h_b, bytes).wait();

  // Command Group creation
  auto cg = [&](sycl::handler &h) {
    h.parallel_for(sycl::range(n),
                   [=](sycl::id<1> i) {
                     d_c[i] = d_a[i] + d_b[i];
                   });
  };

  // Run the kernel defined above
  myQueue.submit(cg).wait();

  // Copy the result back to host
  myQueue.memcpy(h_c, d_c, bytes).wait();

  double sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += h_c[i];
  }
  std::cout << "Sum is : " << sum << std::endl;

  return 0;
}
