Exercise 01: SYCL interop
-------------------------------

In this exercise, you must implement an `interop_task` to let a `SYCL` application call `cuBLAS`.
This application will perform a vector/matrix multiplication using the `cublasSgemv` routine in `cuBLAS`.
A CUDA version of the application is provided, demonstrating how to call `cublasSgemv`.

Requirements
==============

Requires CMake 3.17 to configure (makes use of FindCUDAToolkit for simplicity)
This exercise must be compiled and executed with the DPC++ compiler.
It is expected that you have read at least example-02 before attempting this exercise.


Building the exercise
=====================


Create a build directory and run the following command:

```
CXX=/path/to/dpc++/bin/clang++ cmake build/
```

If NVIDIA CUDA is installed in your system, CMake should be able to generate
the configuration files.

Then run 

```
make
```

to build the exercise

Exercise
=========

Two source codes are provided. `sgemv.cu` is the original CUDA code calling
CUBLAS library to perform the vector/matrix multiplication.
`sycl_sgemv.cpp` is the unfinished SYCL variant that you must complete.
Running the `sycl_sgemv.cpp` executable at this stage will result in a runtime error.

Both implementations set up the same input data and expect the same output.

Familiarise yourself with the `host_task` by reading through the SYCL source in example-02.
