#include <algorithm>
#include <iostream>
#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>

#include <cublas_v2.h>
#include <cuda.h>

#define CHECK_ERROR(FUNC) checkCudaErrorMsg(FUNC, " " #FUNC)

void inline checkCudaErrorMsg(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "ERROR CUBLAS:" << msg << " - " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

void inline checkCudaErrorMsg(cudaError status, const char *msg) {
  if (status != cudaSuccess) {
    std::cout << "ERROR CUDA: " << msg << " - " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

void inline checkCudaErrorMsg(CUresult status, const char *msg) {
  if (status != CUDA_SUCCESS) {
    std::cout << "ERROR CUDA: " << msg << " - " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

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

  constexpr size_t WIDTH = 1024;
  constexpr size_t HEIGHT = 1024;
  constexpr float ALPHA = 1.0f;
  constexpr float BETA = 0.0f;

  std::vector<float> h_A(WIDTH * HEIGHT), h_B(WIDTH * HEIGHT),
      h_C(WIDTH * HEIGHT);

  std::cout << "Size: " << h_C.size() << std::endl;

  // A is an identity matrix
  std::fill(std::begin(h_A), std::end(h_A), 0.0f);
  for (size_t i = 0; i < WIDTH; i++) {
    h_A[i * WIDTH + i] = 1.0f;
  }

  // B is a matrix fill with 1
  std::fill(std::begin(h_B), std::end(h_B), 1.0f);

  sycl::queue q{CUDASelector()};

  // Allocate memory on the device
  float* d_A = sycl::malloc_device<float>(WIDTH*HEIGHT,q);
  float* d_B = sycl::malloc_device<float>(WIDTH*HEIGHT,q);
  float* d_C = sycl::malloc_device<float>(WIDTH*HEIGHT,q);

  // Copy matrices A & B to device from host vectors
  const size_t numBytes = WIDTH * HEIGHT * sizeof(float);
  q.memcpy(d_A, h_A.data(), numBytes).wait();
  q.memcpy(d_B, h_B.data(), numBytes).wait();

  // Create cublas handle
  cublasHandle_t handle;
  CHECK_ERROR(cublasCreate(&handle));

  q.submit([&](handler &h) {

    h.host_task([=](sycl::interop_handle ih) {

      // Set the correct cuda context & stream
      cuCtxSetCurrent(ih.get_native_context<backend::cuda>());
      cublasSetStream(handle, ih.get_native_queue<backend::cuda>());

      // Call generalised matrix-matrix multiply
      CHECK_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, WIDTH, HEIGHT,
                              WIDTH, &ALPHA, d_A, WIDTH, d_B, WIDTH, &BETA,
                              d_C, WIDTH));
    });
  }).wait();

  // Copy the result back to host
  q.memcpy(h_C.data(), d_C, numBytes).wait();

  // C must be all ones
  int i = 0;
  const bool allEqual =
      std::all_of(std::begin(h_C), std::end(h_C), [&i](float num) {
        ++i;
        if (num != 1) {
          std::cout << i << " Not one : " << num << std::endl;
        }
        return num == 1;
      });

  if (!allEqual) {
    std::cout << " Incorrect result " << std::endl;
  } else {
    std::cout << " Correct! " << std::endl;
  }

  CHECK_ERROR(cublasDestroy(handle));

  return allEqual ? EXIT_SUCCESS : EXIT_FAILURE;
}
