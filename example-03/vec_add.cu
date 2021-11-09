// Original source reproduced unmodified here from: 
// https://github.com/olcf/vector_addition_tutorials/blob/master/CUDA/vecAdd.cu

#include <algorithm>
#include <iostream>
#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>

class CUDASelector : public sycl::device_selector {
public:
  int operator()(const sycl::device &device) const override {
    if(device.get_platform().get_backend() == sycl::backend::cuda){
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
        auto dA = reinterpret_cast<double*>(ih.get_native_mem<backend::cuda>(accA));
        auto dB = reinterpret_cast<double*>(ih.get_native_mem<backend::cuda>(accB));
        auto dC = reinterpret_cast<double*>(ih.get_native_mem<backend::cuda>(accC));

        int blockSize, gridSize;
        // Number of threads in each thread block
        blockSize = 1024;
        // Number of thread blocks in grid
        gridSize = static_cast<int>(ceil(static_cast<float>(n) / blockSize));
        // Call the CUDA kernel directly from SYCL
        vecAdd<<<gridSize, blockSize>>>(dA, dB, dC, n);
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
