/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's SYCL-For-CUDA-Examples
 *
 *  sycl_sgemm.cpp
 *
 *  Description:
 *    SGEMM operation in SYCL with buffers
 **************************************************************************/
#include <algorithm>
#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

#include <cuda.h>
#include <cublas_v2.h>

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

/*
 * void inline checkCudaErrorMsg(CUresult status, const char *msg) {
  if (status != CUDA_SUCCESS) {
    std::cout << "ERROR CUDA: " << msg << " - " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}
*/

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

  cublasHandle_t handle;
  CHECK_ERROR(cublasCreate(&handle));

  {
    buffer<float, 2> b_A{h_A.data(), range<2>{WIDTH, HEIGHT}};
    buffer<float, 2> b_B{h_B.data(), range<2>{WIDTH, HEIGHT}};
    buffer<float, 2> b_C{h_C.data(), range<2>{WIDTH, HEIGHT}};

    q.submit([&](handler &h) {
      auto d_A = b_A.get_access<sycl::access::mode::read>(h);
      auto d_B = b_B.get_access<sycl::access::mode::read>(h);
      auto d_C = b_C.get_access<sycl::access::mode::write>(h);

      h.host_task([=](sycl::interop_handle ih) {
        auto cuStream = ih.get_native_queue<backend::ext_oneapi_cuda>();
        cublasSetStream(handle, cuStream);
        auto cuA = reinterpret_cast<float *>(ih.get_native_mem<backend::ext_oneapi_cuda>(d_A));
        auto cuB = reinterpret_cast<float *>(ih.get_native_mem<backend::ext_oneapi_cuda>(d_B));
        auto cuC = reinterpret_cast<float *>(ih.get_native_mem<backend::ext_oneapi_cuda>(d_C));

        CHECK_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, WIDTH, HEIGHT,
                                WIDTH, &ALPHA, cuA, WIDTH, cuB, WIDTH, &BETA,
                                cuC, WIDTH));
        cuStreamSynchronize(cuStream);
      });
    });
  }

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
