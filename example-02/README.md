SYCL interop with CUDA library
-------------------------------

The example shows how to interop with CUBLAS from a SYCL for CUDA application.
The example uses Codeplay's extension *interop_task* to call the **SGEMM** 
routine in CUBLAS. Parameters are extracted using the interop handler conversion.

Requirements
==============

Requires CMake 3.17 to configure (makes use of FindCUDAToolkit for simplicity)
This example must be compiled and executed with the DPC++ compiler.

Building the example
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

to build the example

Example
=========

Two source codes are provided. `sgemm.cu` is the original CUDA code calling
CUBLAS library to perform the matrix multiplication.
`sycl_sgemm.cpp` is the sycl variant that calls CUBLAS underneath.

Both implementations perform the multiplication of square matrices A and B, 
where A is a matrix full of ones, and B is an identity matrix.
The expected output on C is a matrix full of ones.

