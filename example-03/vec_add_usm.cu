// Original source reproduced unmodified here from: 
// https://github.com/olcf/vector_addition_tutorials/blob/master/CUDA/vecAdd.cu

#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>

class CUDASelector : public sycl::device_selector {
public:
  int operator()(const sycl::device &Device) const override {
    using namespace sycl::info;

    const std::string DriverVersion = Device.get_info<device::driver_version>();

    if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos)) {
      std::cout << " CUDA device found \n";
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

  // Size, in bytes, of each vector
  size_t bytes = n * sizeof(double);

  // Create a SYCL context for interoperability with CUDA Runtime API
  // This is temporary until the property extension is implemented
  const bool UsePrimaryContext = true;
  device dev{CUDASelector().select_device()};
  context myContext{dev, {}, UsePrimaryContext};
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
      h.interop_task([=](interop_handler ih) {
        // Number of threads in each thread block
        int blockSize = 1024;

        // Number of thread blocks in grid
        int gridSize = static_cast<int>(ceil(static_cast<float>(n) / blockSize));

        // Execute the kernel
        vecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
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
