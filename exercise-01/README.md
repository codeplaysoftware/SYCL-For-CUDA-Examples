Exercise 01: SYCL interop
-------------------------------

In this exercise, you must implement an `interop_task` to let a `SYCL`
application call `cuBLAS` . This application will perform a vector/matrix
multiplication using the `cublasSgemv` routine in `cuBLAS` . A CUDA version of
the application is provided, demonstrating how to call `cublasSgemv` .

Pre-requisites
---------------

These instructions assume that example [docker image](https://hub.docker.com/r/ruyman/dpcpp_cuda_examples/dockerfile) is being used. This image 
simplifies accessing these examples as the environment is set up correctly.
For details on how to get started with the example docker image, refer to the 
root README file.

Requires CMake 3.17 to configure (makes use of FindCUDAToolkit for simplicity)
This exercise must be compiled and executed with the DPC++ compiler.
It is expected that you have read at least example-02 before attempting this exercise.

Building the exercise
=====================

``` sh
$ mkdir build && cd build
$ cmake ../ -DSYCL_ROOT=${SYCL_ROOT_DIR} -DCMAKE_CXX_COMPILER=${SYCL_ROOT_DIR}/bin/clang++
$ make -j 8
```

Exercise
=========

Two source codes are provided. `sgemv.cu` is the original CUDA code calling
CUBLAS library to perform the vector/matrix multiplication. `sycl_sgemv.cpp` is
the unfinished SYCL variant that you must complete. Running the `sycl_sgemv.cpp`
executable at this stage will result in a runtime error.

Both implementations set up the same input data and expect the same output.

Familiarise yourself with the `host_task` by reading through the SYCL source in
example-02.
