#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>
#include <algorithm>
#include <cassert>
#include <cublas_v2.h>
#include <cuda.h>
#include <iostream>
#include <vector>

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

int main() {
  using namespace sycl;

  constexpr size_t ROWS = 6;
  constexpr size_t COLUMNS = 5;
  constexpr float ALPHA = 1.0f;
  constexpr float BETA = 0.0f;

  std::vector<float> hostA(ROWS * COLUMNS);
  std::vector<float> hostB(COLUMNS);
  std::vector<float> hostC(ROWS);

  int index = 11;
  for (size_t i = 0; i < COLUMNS; i++) {
    for (size_t j = 0; j < ROWS; j++) {
      hostA[(i * ROWS) + j] = static_cast<float>(index++);
    }
  }

  std::fill(std::begin(hostB), std::end(hostB), 1.0f);

  // hostA:
  // [11, 17, 23, 29, 35]
  // [12, 18, 24, 30, 36]
  // [13, 19, 25, 31, 37]
  // [14, 20, 26, 32, 38]
  // [15, 21, 27, 33, 39]
  // [16, 22, 28, 34, 40]

  // hostB:
  // [1, 1, 1, 1, 1]

  // hostC:
  // [0, 0, 0, 0, 0, 0]

  queue q{CUDASelector()};

  cublasHandle_t handle;
  cublasCreate(&handle);

  {
    buffer<float, 2> bufferA{hostA.data(), range<2>{ROWS, COLUMNS}};
    buffer<float, 1> bufferB{hostB.data(), range<1>{COLUMNS}};
    buffer<float, 1> bufferC{hostC.data(), range<1>{ROWS}};

    q.submit([&](handler &h) {
      // exercise-01
    });
  }

  assert(hostC[0] == 115); // [11, 17, 23, 29, 35]     [1]
  assert(hostC[1] == 120); // [12, 18, 24, 30, 36]     [1]
  assert(hostC[2] == 125); // [13, 19, 25, 31, 37]  *  [1]
  assert(hostC[3] == 130); // [14, 20, 26, 32, 38]     [1]
  assert(hostC[4] == 135); // [15, 21, 27, 33, 39]     [1]
  assert(hostC[5] == 140); // [16, 22, 28, 34, 40]

  cublasDestroy(handle);

  return EXIT_SUCCESS;
}
