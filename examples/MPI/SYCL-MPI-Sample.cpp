
#include <CL/sycl.hpp>

#include <CL/sycl/backend/cuda.hpp>
#include <algorithm>
#include <mpi.h>
#include <mpi-ext.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  /* -------------------------------------------------------------------------------------------
      SYCL Initialization, which internally sets the CUDA device
  --------------------------------------------------------------------------------------------*/
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
      printf("This program requires exactly 2 MPI ranks, but you are "
             "attempting to use %d! Exiting...\n",
             size);
    }
    MPI_Finalize();
    exit(0);
  }

  /* -------------------------------------------------------------------------------------------
     Setting data size to 1MB
     Allocating 1 MB data on Host
  --------------------------------------------------------------------------------------------*/
  long int N = 1 << 10;
  std::vector<double> A(N, 1.0);
  size_t local_size = N / size;

  /* -------------------------------------------------------------------------------------------
     Create SYCL buffers
  --------------------------------------------------------------------------------------------*/
  sycl::buffer<double> input_buffer(std::begin(A), std::end(A));
  sycl::buffer<double> local_buffer(sycl::range{local_size});
  sycl::buffer<double> out_buffer(sycl::range{1});
  sycl::buffer<double> global_sum(sycl::range{1});

  double start_time, stop_time, elapsed_time;
  start_time = MPI_Wtime();
  /* -------------------------------------------------------------------------------------------
    Create an SYCL interoperability with CUDA to scatter the data among two MPI
  process
  --------------------------------------------------------------------------------------------*/

  auto ht = [&](sycl::handler &h) {
    sycl::accessor input_acc{input_buffer, h, sycl::read_write};
    sycl::accessor local_acc{local_buffer, h, sycl::read_write};
    h.host_task([=](sycl::interop_handle ih) {
      auto cuda_ptr = reinterpret_cast<double *>(
          ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(input_acc));
      auto cuda_local_ptr = reinterpret_cast<double *>(
          ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(local_acc));
      MPI_Scatter(cuda_ptr, local_size, MPI_DOUBLE, cuda_local_ptr, local_size,
                  MPI_DOUBLE, 0, MPI_COMM_WORLD);
    });
  };
  q.submit(ht);
 

  /* -------------------------------------------------------------------------------------------
    Create a SYCL GPU kernel to sale each element of the data based on the MPI
  process ID
  --------------------------------------------------------------------------------------------*/
  auto cg = [&](sycl::handler &h) {
    auto acc = local_buffer.get_access(h);
    auto kern = [=](cl::sycl::id<1> id) { acc[id] *= (rank + 1); };
    h.parallel_for(sycl::range<1>{local_size}, kern);
  };
  q.submit(cg);

  /* -------------------------------------------------------------------------------------------
    Create a SYCL GPU kernel to partially reduce each local data into an scalar
  --------------------------------------------------------------------------------------------*/
  auto cg2 = [&](sycl::handler &h) {
    auto acc = local_buffer.get_access(h);
    h.parallel_for(sycl::nd_range<1>(
                       cl::sycl::range<1>(local_size),
                       cl::sycl::range<1>(std::min(local_size, size_t(256)))),
                   sycl::reduction(out_buffer, h, 1.0, std::plus<double>()),
                   [=](sycl::nd_item<1> idx, auto &reducer) {
                     reducer.combine(acc[idx.get_global_id(0)]);
                   });
  };
  q.submit(cg2);
  /* -------------------------------------------------------------------------------------------
    Create a SYCL interoperability with CUDA to calculate the total sum of the
  reduced scalar created by each MPI process
  --------------------------------------------------------------------------------------------*/
  auto ht2 = [&](sycl::handler &h) {
    sycl::accessor out_acc{out_buffer, h, sycl::read_write};
    sycl::accessor global_sum_acc{global_sum, h, sycl::read_write};
    h.host_task([=](sycl::interop_handle ih) {
      auto cuda_out_ptr = reinterpret_cast<double *>(
          ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(out_acc));
      auto cuda_global_sum_ptr = reinterpret_cast<double *>(
          ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(global_sum_acc));
      MPI_Allreduce(cuda_out_ptr, cuda_global_sum_ptr, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
    });
  };

  q.submit(ht2);

  /* -------------------------------------------------------------------------------------------
    Create a SYCL GPU kernel to normalize local buffer based on the global sum
  result
  --------------------------------------------------------------------------------------------*/
  auto cg3 = [&](sycl::handler &h) {
    auto acc = local_buffer.get_access(h);
    auto global_sum_acc = global_sum.get_access(h);
    auto kern = [=](cl::sycl::id<1> id) { acc[id] /= global_sum_acc[0]; };
    h.parallel_for(sycl::range<1>{local_size}, kern);
  };
  q.submit(cg3);

  /* -------------------------------------------------------------------------------------------
    Create a SYCL interoperability with CUDA to replace the original input with
  normalized value
  --------------------------------------------------------------------------------------------*/
  auto ht3 = [&](sycl::handler &h) {
    sycl::accessor input_acc{input_buffer, h, sycl::read_write};
    sycl::accessor local_acc{local_buffer, h, sycl::read_write};
    h.host_task([=](sycl::interop_handle ih) {
      auto cuda_local_ptr = reinterpret_cast<double *>(
          ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(local_acc));
      auto cuda_input_ptr = reinterpret_cast<double *>(
          ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(input_acc));
      MPI_Gather(cuda_local_ptr, local_size, MPI_DOUBLE, cuda_input_ptr,
                 local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    });
  };

  q.submit(ht3);
  q.wait_and_throw();
  stop_time = MPI_Wtime();
  elapsed_time = stop_time - start_time;

  /* -------------------------------------------------------------------------------------------
     Print the output
  --------------------------------------------------------------------------------------------*/
  if (rank == 0) {
    std::cout << "elapsed_time" << elapsed_time;
#if defined(PRINT_DEBUG_MODE)
    auto p = input_buffer.get_host_access();
    for (int i = 0; i < 1; i++) {
      std::cout << " value at i : " << p[i] << "\n";
    }
#endif
  }
  MPI_Finalize();

  return 0;
}
