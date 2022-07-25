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
 *  distributed-batch-gemm.cpp
 *
 *  Description:
 *    Matrix multiplication distributed across devices using MPI
 **************************************************************************/
#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>
#include <algorithm>
#include <mpi.h>
#include <mpi-ext.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <sycl_blas.h>

#define PRINT_DEBUG_MODE 1

int main(int argc, char **argv) {
  /* Create a SYCL queue with the default device selector */
  sycl::queue q(cl::sycl::gpu_selector{});

  /* -------------------------------------------------------------------------------------------
      Check to see if MPI library is CUDA-aware
  --------------------------------------------------------------------------------------------*/
  printf("Run time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (1 == MPIX_Query_cuda_support()) {
    printf("This MPI library has CUDA-aware support.\n");
  } else {
    printf("This MPI library does not have CUDA-aware support.\n");
  }
#else  /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
  printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */

  /* -------------------------------------------------------------------------------------------
     MPI Initialization
  --------------------------------------------------------------------------------------------*/
  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size != 2) {
    if (rank == 0) {
      printf(
          "This program requires exactly 2 MPI ranks, but you are "
          "attempting to use %d! Exiting...\n",
          size);
    }
    MPI_Finalize();
    exit(0);
  }

  double start_time, stop_time, elapsed_time;
  /* Create a SYCL-BLAS executor and get the policy handler */
  blas::Executor<blas::PolicyHandler<blas::codeplay_policy>> executor(q);
  auto policy_handler = executor.get_policy_handler();

  /* Arguments of the Gemm operation.
   * Note: these matrix dimensions are too small to get a performance gain by
   * using SYCL-BLAS, but they are convenient for this sample */
  const int m = 32;
  const int k = 32;
  const int n = 32;
  const int lda = m;
  const int ldb = k;
  const int ldc = m;
  const float alpha = 1;
  const float beta = 0;
  const float batch = 2;

  /* creating local buffer */
  auto local_a_gpu = blas::make_sycl_iterator_buffer<float>(lda * k);
  auto local_b_gpu = blas::make_sycl_iterator_buffer<float>(ldb * n);
  auto local_c_gpu = blas::make_sycl_iterator_buffer<float>(ldc * n);

  /* Create the global buffer */
  auto global_a_gpu = blas::make_sycl_iterator_buffer<float>(batch * lda * k);
  auto global_b_gpu = blas::make_sycl_iterator_buffer<float>(batch * ldb * n);
  auto global_c_gpu = blas::make_sycl_iterator_buffer<float>(batch * ldc * n);

  if (rank == 0) {
    // Setting buffer value for A and B
    std::vector<float> A = std::vector<float>(batch * lda * k, float(1.0));
    std::vector<float> B = std::vector<float>(batch * ldb * n, float(1.0));
    policy_handler.copy_to_device(A.data(), global_a_gpu, batch * lda * k);
    policy_handler.copy_to_device(B.data(), global_b_gpu, batch * ldb * n);
  }
  /* -------------------------------------------------------------------------------------------
    Create an SYCL interoperability with CUDA to scatter the data each batch A,
  B among the two MPI process
  --------------------------------------------------------------------------------------------*/
  start_time = MPI_Wtime();
  auto ht_a = [&](sycl::handler &h) {
    auto global_a_acc =
        global_a_gpu.get_buffer().template get_access<sycl::access::mode::read>(
            h);
    auto local_a_acc =
        local_a_gpu.get_buffer().template get_access<sycl::access::mode::write>(
            h);
    h.interop_task([=](sycl::interop_handle ih) {
      auto global_a_ptr = reinterpret_cast<float *>(
          ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(global_a_acc));
      auto local_a_ptr = reinterpret_cast<float *>(
          ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(local_a_acc));
      MPI_Scatter(global_a_ptr, lda * k, MPI_FLOAT, local_a_ptr, lda * k,
                  MPI_FLOAT, 0, MPI_COMM_WORLD);
    });
  };
  q.submit(ht_a);

  auto ht_b = [&](sycl::handler &h) {
    auto global_b_acc =
        global_b_gpu.get_buffer().template get_access<sycl::access::mode::read>(
            h);
    auto local_b_acc =
        local_b_gpu.get_buffer().template get_access<sycl::access::mode::write>(
            h);
    h.interop_task([=](sycl::interop_handle ih) {
      auto global_b_ptr = reinterpret_cast<float *>(
          ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(global_b_acc));
      auto local_b_ptr = reinterpret_cast<float *>(
          ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(local_b_acc));
      MPI_Scatter(global_b_ptr, ldb * n, MPI_FLOAT, local_b_ptr, ldb * n,
                  MPI_FLOAT, 0, MPI_COMM_WORLD);
    });
  };
  q.submit(ht_b);
  q.wait_and_throw();

  /* Execute the GEMM operation */
  auto event = blas::_gemm(executor, 'n', 'n', m, n, k, alpha, local_a_gpu, lda,
                           local_b_gpu, ldb, beta, local_c_gpu, ldc);
  policy_handler.wait(event);

  /* -------------------------------------------------------------------------------------------
    Create a SYCL interoperability with CUDA to replace the original input with
  normalized value
  --------------------------------------------------------------------------------------------*/
  auto ht_c = [&](sycl::handler &h) {
    auto global_c_acc = global_c_gpu.get_buffer()
                            .template get_access<sycl::access::mode::write>(h);
    auto local_c_acc =
        local_c_gpu.get_buffer().template get_access<sycl::access::mode::read>(
            h);
    h.interop_task([=](sycl::interop_handle ih) {
      auto local_c_ptr = reinterpret_cast<float *>(
          ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(local_c_acc));
      auto global_c_ptr = reinterpret_cast<float *>(
          ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(global_c_acc));
      MPI_Gather(local_c_ptr, ldc * n, MPI_FLOAT, global_c_ptr, ldc * n,
                 MPI_FLOAT, 0, MPI_COMM_WORLD);
    });
  };

  q.submit(ht_c);
  q.wait_and_throw();
  stop_time = MPI_Wtime();
  elapsed_time = stop_time - start_time;

  /* -------------------------------------------------------------------------------------------
       Print the output
    --------------------------------------------------------------------------------------------*/
  if (rank == 0) {
    std::cout << "elapsed_time" << elapsed_time;
#if defined(PRINT_DEBUG_MODE)
    auto C = global_c_gpu.get_buffer().get_host_access();
    for (int i = 0; i < batch * ldc * n; i++) {
      std::cout << " value at " << i << " : " << C[i] << "\n";
    }
#endif
  }

  MPI_Finalize();
  return 0;
}
