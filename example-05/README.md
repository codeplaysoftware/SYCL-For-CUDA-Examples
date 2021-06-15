## Distributed Batch GEMM example

This example shows how to integrate MPI calls within the SYCL DAG using Host Tasks to distribute Batch GEMM accross MPI process.


## Requisites

The Makefile provided assumes the MPICXX compiler points to the DPCPP compiler with CUDA support.
That requires the MPI implementation to be built, or use, the DPCPP compiler.
The MPI implementation needs to have been built with CUDA support (typically called "CUDA-aware" MPI")

The example uses [SYCL-BLAS](https://github.com/codeplaysoftware/sycl-blas) library to call the GEMM routine.
The SYCL-BLAS Library should be [compiled by DPCPP compiler](https://github.com/codeplaysoftware/sycl-blas#compile-with-dpc) to target CUDA backend. The following command line is used to build SYCL-BLAS library:

```bash
cmake -GNinja ../ -DTARGET=NVIDIA_GPU -DSYCL_COMPILER=dpcpp -DBLAS_DATA_TYPES=float -DGEMM_VECTORIZATION_SUPPORT=ON -DBLAS_ENABLE_TESTING=OFF -DENABLE_EXPRESSION_TESTS=OFF -DBLAS_ENABLE_BENCHMARK=OFF -DBLAS_VERIFY_BENCHMARK=OFF -DBLAS_BUILD_SAMPLES=OFF
```

## Compilation

If MPICXX points to DPC++ with CUDA support and its on the path, "make" should build the program.

## Execution

The makefile contains a target to execute the problem in two processes:

```sh
make run
```

The target assumes mpirun is on the PATH
