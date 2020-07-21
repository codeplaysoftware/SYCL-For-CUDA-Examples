SYCL interop with CUDA library
-------------------------------

The example shows how to interop with CUBLAS from a SYCL for CUDA application.
The example uses Codeplay's extension *interop_task* to call the **SGEMM** 
routine in CUBLAS. Parameters are extracted using the interop handler conversion.

Pre-requisites
---------------

These instructions assume that example [docker image](https://hub.docker.com/r/ruyman/dpcpp_cuda_examples/dockerfile) is being used. This image 
simplifies accessing these examples as the environment is set up correctly.
For details on how to get started with the example docker image, refer to the 
root README file.

Building the example
=====================

``` sh
$ mkdir build && cd build
$ cmake ../
$ make -j 8
```

Example
=========

Two source codes are provided. `sgemm.cu` is the original CUDA code calling
CUBLAS library to perform the matrix multiplication. `sycl_sgemm.cpp` is the 
SYCL variant that calls CUBLAS underneath.

Both implementations perform the multiplication of square matrices A and B, 
where A is a matrix full of ones, and B is an identity matrix.
The expected output on C is a matrix full of ones.
