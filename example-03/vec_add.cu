// Original source reproduced unmodified here from: 
// https://github.com/olcf/vector_addition_tutorials/blob/master/CUDA/vecAdd.cu

#include <algorithm>
#include <iostream>
#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>

class CUDASelector : public sycl::device_selector {
public:
  int operator()(const sycl::device &Device) const override {
    using namespace sycl::info;

    const std::string DriverVersion = Device.get_info<device::driver_version>();

    if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos)) {
      std::cout << " CUDA device found " << std::endl;
      return 1;
    };
    return -1;
  }
};

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n) {
  // Get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n)
    c[id] = a[id] + b[id];
}

int main(int argc, char *argv[]) {
  using namespace sycl;
  // Size of vectors
  int n = 100000;

  // Create a SYCL context for interoperability with CUDA Runtime API
  // This is temporary until the property extension is implemented
  const bool UsePrimaryContext = true;
  sycl::device dev{CUDASelector().select_device()};
  sycl::context myContext{dev, {}, UsePrimaryContext};
  sycl::queue myQueue{myContext, dev};

  {
    buffer<double> bA{range<1>(n)};
    buffer<double> bB{range<1>(n)};
    buffer<double> bC{range<1>(n)};

    {
      auto h_a = bA.get_access<access::mode::write>();
      auto h_b = bB.get_access<access::mode::write>();

      // Initialize vectors on host
      for (int i = 0; i < n; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
      }
    }

    // Dispatch a command group with all the dependencies
    myQueue.submit([&](handler& h) {
      auto accA = bA.get_access<access::mode::read>(h);
      auto accB = bB.get_access<access::mode::read>(h);
      auto accC = bC.get_access<access::mode::write>(h);

      h.interop_task([=](interop_handler ih) {
        auto d_a = reinterpret_cast<double*>(ih.get_mem<backend::cuda>(accA));
        auto d_b = reinterpret_cast<double*>(ih.get_mem<backend::cuda>(accB));
        auto d_c = reinterpret_cast<double*>(ih.get_mem<backend::cuda>(accC));

        int blockSize, gridSize;
        // Number of threads in each thread block
        blockSize = 1024;
        // Number of thread blocks in grid
        gridSize = (int)ceil((float)n / blockSize);
        // Call the CUDA kernel directly from SYCL
        vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
      });
    });

    {
     auto h_c = bC.get_access<access::mode::read>();
     // Sum up vector c and print result divided by n, this should equal 1 within
     // error
     double sum = 0;
      for (int i = 0; i < n; i++)
        sum += h_c[i];
      printf("final result: %f\n", sum / n);
    }
  }


  return 0;
}
