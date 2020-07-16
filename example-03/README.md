Example 03: Calling CUDA kernels from SYCL
===============================

In this example, we re-use the trivial SYCL kernel we used on Example 1, 
but instead of writing the SYCL variant, we will keep the original CUDA
kernel, only replacing the CUDA Runtime calls with the SYCL API.

This variant uses buffer and accessor syntax, which is more verbose but allows
the creation of the implicit DAG.
An USM variant is presented for exposition only, support for USM in CUDA is
unstable at the time of writting.

Pre-requisites
---------------

These instructions assume that example [docker image](https://hub.docker.com/r/ruyman/dpcpp_cuda_examples/dockerfile) is being used. This image 
simplifies accessing these examples as the environment is set up correctly.
For details on how to get started with the example docker image, refer to the 
root README file.

The example is built using Makefiles, since there is no support yet on
a release of CMake for changing the CUDA compiler from nvcc.

Building the example
---------------------

``` sh
make  
```

This compiles the SYCL code with the LLVM CUDA support, and generates two 
binaries. NVCC is not used, but the CUDA device libraries need to be available
on `/usr/local/cuda/lib64/` for linking to the device code.

NVCC compiler does not support some of the advanced C++17 syntax used on the
SYCL Runtime headers.

Running the example
--------------------

``` 
./vec_add.exe
```

Calling CUDA kernels from SYCL
-------------------------------

Using Codeplay's `interop_task` extension, the example calls a CUDA kernel from
a SYCL application. Note the example is compiled with the LLVM CUDA compiler, 
not with the SYCL compiler, since there are no SYCL kernels on it. It is only
required to link against the SYCL runtime library to ensure the runtime can use
the application.

At the time of writing, it is not possible to have both CUDA and SYCL kernels
on the same file. It is possible to have different files for CUDA and SYCL 
kernels and call them together from a main application at runtime.

The example uses an extension to the SYCL interface to interact with the CUDA
Runtime API. At the time of writing the extension is not public, so only a
boolean flagis passed to the `sycl::context` creation.
