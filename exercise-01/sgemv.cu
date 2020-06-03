#include "cublas_v2.h"
#include <cassert>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

int main() {
  constexpr size_t ROWS = 6;
  constexpr size_t COLUMNS = 5;
  constexpr float ALPHA = 1.0f;
  constexpr float BETA = 0.0f;

  cublasHandle_t handle;

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

  float *deviceA = nullptr;
  float *deviceB = nullptr;
  float *deviceC = nullptr;

  cudaMalloc((void **)&deviceA, ROWS * COLUMNS * sizeof(float));
  cudaMalloc((void **)&deviceB, COLUMNS * sizeof(float));
  cudaMalloc((void **)&deviceC, ROWS * sizeof(float));

  cublasCreate(&handle);

  cublasSetMatrix(ROWS, COLUMNS, sizeof(float), hostA.data(), ROWS, deviceA,
                  ROWS);
  cublasSetVector(COLUMNS, sizeof(float), hostB.data(), 1, deviceB, 1);
  cublasSetVector(ROWS, sizeof(float), hostC.data(), 1, deviceC, 1);
  cublasSgemv(handle, CUBLAS_OP_N, ROWS, COLUMNS, &ALPHA, deviceA, ROWS,
              deviceB, 1, &BETA, deviceC, 1);
  cublasGetVector(ROWS, sizeof(float), deviceC, 1, hostC.data(), 1);

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  assert(hostC[0] == 115); // [11, 17, 23, 29, 35]     [1]
  assert(hostC[1] == 120); // [12, 18, 24, 30, 36]     [1]
  assert(hostC[2] == 125); // [13, 19, 25, 31, 37]  *  [1]
  assert(hostC[3] == 130); // [14, 20, 26, 32, 38]     [1]
  assert(hostC[4] == 135); // [15, 21, 27, 33, 39]     [1]
  assert(hostC[5] == 140); // [16, 22, 28, 34, 40]

  cublasDestroy(handle);
}