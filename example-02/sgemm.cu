#include <algorithm>
#include <iostream>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>

#define CHECK_ERROR(FUNC) checkCudaErrorMsg(FUNC, " " #FUNC)

void inline checkCudaErrorMsg(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << msg << " - " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

void inline checkCudaErrorMsg(cudaError status, const char *msg) {
  if (status != CUDA_SUCCESS) {
    std::cout << msg << " - " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main() {
  constexpr size_t WIDTH = 1024;
  constexpr size_t HEIGHT = 1024;
  constexpr float ALPHA = 1.0f;
  constexpr float BETA = 0.0f;

  std::vector<float> h_A(WIDTH * HEIGHT), h_B(WIDTH * HEIGHT),
      h_C(WIDTH * HEIGHT);

  std::cout << "Size: " << h_C.size() << std::endl;
  float *d_A, *d_B, *d_C;

  // A is an identity matrix
  std::fill(std::begin(h_A), std::end(h_A), 0.0f);
  for (size_t i = 0; i < WIDTH; i++) {
    h_A[i * WIDTH + i] = 1.0f;
  }

  // B is a matrix fill with 1
  std::fill(std::begin(h_B), std::end(h_B), 1.0f);

  const size_t numBytes = WIDTH * HEIGHT * sizeof(float);

  CHECK_ERROR(cudaMalloc((void **)&d_A, numBytes));
  CHECK_ERROR(cudaMalloc((void **)&d_B, numBytes));
  CHECK_ERROR(cudaMalloc((void **)&d_C, numBytes));

  CHECK_ERROR(cudaMemcpy(d_A, h_A.data(), numBytes, cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemcpy(d_B, h_B.data(), numBytes, cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CHECK_ERROR(cublasCreate(&handle));

  // C = A * B
  CHECK_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, WIDTH, HEIGHT,
                          WIDTH, &ALPHA, d_A, WIDTH, d_B, WIDTH, &BETA, d_C,
                          WIDTH));

  CHECK_ERROR(cudaMemcpy(h_C.data(), d_C, numBytes, cudaMemcpyDeviceToHost));

  // C must be all ones
  const bool allEqual = std::all_of(std::begin(h_C), std::end(h_C),
                                    [](float num) { return num == 1; });

  if (!allEqual) {
    std::cout << " Incorrect result " << std::endl;
  } else {
    std::cout << " Correct! " << std::endl;
  }

  CHECK_ERROR(cublasDestroy(handle));
  CHECK_ERROR(cudaFree(d_A));
  CHECK_ERROR(cudaFree(d_B));
  CHECK_ERROR(cudaFree(d_C));

  return allEqual ? EXIT_SUCCESS : EXIT_FAILURE;
}
