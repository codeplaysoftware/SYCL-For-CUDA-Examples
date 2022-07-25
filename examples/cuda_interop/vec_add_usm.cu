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

  // Size, in bytes, of each vector
  size_t bytes = n * sizeof(double);

  device dev{CUDASelector().select_device()};
  context myContext{dev};
  queue myQueue{myContext, dev};

  // Allocate memory for each vector on host
  auto d_A = reinterpret_cast<double*>(malloc_shared(bytes, myQueue));
  auto d_B = reinterpret_cast<double*>(malloc_shared(bytes, myQueue));
  auto d_C = reinterpret_cast<double*>(malloc_shared(bytes, myQueue));

  // Initialize vectors on host
  for (int i = 0; i < n; i++) {
    d_A[i] = sin(i) * sin(i);
    d_B[i] = cos(i) * cos(i);
  }

  myQueue.submit([&](handler& h) {
      h.host_task([=](interop_handle ih) {
        // Number of threads in each thread block
        int blockSize = 1024;

        // Number of thread blocks in grid
        int gridSize = static_cast<int>(ceil(static_cast<float>(n) / blockSize));

        // Execute the kernel
        vecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
        // Interop with host_task doesn't add CUDA event to task graph
        // so we must manually sync here.
        cudaDeviceSynchronize();
        });
  });

  myQueue.wait();

  // Sum up vector c and print result divided by n, this should equal 1 within
  // error
  double sum = 0;
  for (int i = 0; i < n; i++) {
    sum += d_C[i];
  }
  std::cout << "Final result " << sum / n << std::endl;

  free(d_A, myContext);
  free(d_B, myContext);
  free(d_C, myContext);

  return 0;
}
