## MPI + SYCL example

This example shows how to integrate MPI calls within the SYCL DAG using Host Tasks for integration.


## Requisites

The Makefile provided assumes the MPICXX compiler points to the DPCPP compiler with CUDA support.
That requires the MPI implementation to be built, or use, the DPCPP compiler.
The MPI implementation needs to have been built with CUDA support (typically called "CUDA-aware" MPI")

## Compilation

If MPICXX points to DPC++ with CUDA support and its on the path, "make" should build the program.

## Execution

The makefile contains a target to execute the problem in two processes:

```sh
make run
```

The target assumes mpirun is on the PATH


