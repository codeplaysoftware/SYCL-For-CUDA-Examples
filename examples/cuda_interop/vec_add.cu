/* 
Copyright (c) 2022 Tom Papatheodore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <algorithm>
#include <iostream>
#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>

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

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n) {
  // Get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n) {
    c[id] = a[id] + b[id];
  }
}

int main(int argc, char *argv[]) {
  using namespace sycl;
  // Size of vectors
  int n = 100000;

  device dev{CUDASelector().select_device()};
  context myContext{dev};
  queue myQueue{myContext, dev};

  {
    buffer<double> bA{range<1>(n)};
    buffer<double> bB{range<1>(n)};
    buffer<double> bC{range<1>(n)};

    {
      auto hA = bA.get_access<access::mode::write>();
      auto hB = bB.get_access<access::mode::write>();

      // Initialize vectors on host
      for (int i = 0; i < n; i++) {
        hA[i] = sin(i) * sin(i);
        hB[i] = cos(i) * cos(i);
      }
    }

    // Dispatch a command group with all the dependencies
    myQueue.submit([&](handler& h) {
      auto accA = bA.get_access<access::mode::read>(h);
      auto accB = bB.get_access<access::mode::read>(h);
      auto accC = bC.get_access<access::mode::write>(h);

      h.host_task([=](interop_handle ih) {
        auto dA = reinterpret_cast<double*>(ih.get_native_mem<backend::ext_oneapi_cuda>(accA));
        auto dB = reinterpret_cast<double*>(ih.get_native_mem<backend::ext_oneapi_cuda>(accB));
        auto dC = reinterpret_cast<double*>(ih.get_native_mem<backend::ext_oneapi_cuda>(accC));

        int blockSize, gridSize;
        // Number of threads in each thread block
        blockSize = 1024;
        // Number of thread blocks in grid
        gridSize = static_cast<int>(ceil(static_cast<float>(n) / blockSize));
        // Call the CUDA kernel directly from SYCL
        vecAdd<<<gridSize, blockSize>>>(dA, dB, dC, n);
        // Interop with host_task doesn't add CUDA event to task graph
        // so we must manually sync here.
        cudaDeviceSynchronize();
      });
    });

    {
     auto hC = bC.get_access<access::mode::read>();
     // Sum up vector c and print result divided by n, this should equal 1 within
     // error
     double sum = 0;
     for (int i = 0; i < n; i++) {
        sum += hC[i];
     }
      std::cout << "Final result " << sum / n << std::endl;
    }
  }


  return 0;
}
