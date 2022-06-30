SYCL for CUDA examples
==========================

This repository contains examples that demonstrate how to use the CUDA backend
in SYCL.

The examples are built and test in Linux with GCC 7.4, NVCC 10.1 and the
experimental support for CUDA in the DPC++ SYCL implementation.

CUDA is a registered trademark of NVIDIA Corporation
SYCL is a trademark of the Khronos Group Inc.

Prerequisites
-------------

These examples are intended to be used with this [docker image](https://hub.docker.com/r/ruyman/dpcpp_cuda_examples). 
It provides all the examples, libraries and the required environment variables. 

[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) must be installed to run the image.

A useful guide for setting up docker and the NVIDIA Container Toolkit can be found [here](https://www.pugetsystems.com/labs/hpc/Workstation-Setup-for-Docker-with-the-New-NVIDIA-Container-Toolkit-nvidia-docker2-is-deprecated-1568).

Getting Started
-------------

Once docker and the NVIDIA Container Toolkit are installed, we can create a new container and run the examples witin it.

``` sh
$ sudo docker run --gpus all -it ruyman/dpcpp_cuda_examples
```

Once inside the docker image, navigate to `/home/examples/` to find a local clone of this repo. Make sure to pull the latest changes:

``` sh
$ cd /home/examples/SYCL-For-CUDA-Examples
$ git pull
```

Refer to each example and/or exercise for detailed instructions on how  to run it.

Examples
=========

[Vector Addition](examples/vector_addition)
--------------------------------------------

This trivial example can be used to compare a simple vector addition in CUDA to
an equivalent implementation in SYCL for CUDA. The aim of the example is also 
to highlight how to build an application with SYCL for CUDA using DPC++ support, 
for which an example CMakefile is provided.

[CUDA Interop](examples/cuda_interop)
--------------------------------------------

In this example, we re-use the trivial SYCL kernel we used in Vector Addition, but instead of writing the SYCL variant, we will keep the original CUDA kernel, only replacing the CUDA Runtime calls with the SYCL API.

[Fortran Interface](examples/fortran_interface)
--------------------------------------------

This demonstrates an example of how to call a SYCL function from a CUDA fortran code.

[MPI](examples/MPI)
--------------------------------------------

This example shows how to integrate MPI calls within the SYCL DAG using Host Tasks for integration.


[SGEMM Interop](examples/sgemm_interop)
--------------------------

This demonstrates using SYCL's `host_task` for CUDA interoperability, calling CUBLAS's SGEMM routine for matrix multiplication.

[Distributed (MPI) GEMM](examples/distrib_batch_gemm)
--------------------------------------------

This example combines the MPI and SGEMM Interop examples to distribute a matrix multiplication problem between MPI ranks.

[Hashing Algorithms](examples/hashing)
--------------------------------------------

This example is slightly different - it benchmarks a series of hashing algorithms.