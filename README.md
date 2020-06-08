SYCL for CUDA examples
==========================

This repository contains examples that demonstrate how to use the CUDA backend
in SYCL.

The examples are built and test in Linux with GCC 7.4, NVCC 10.1 and the
experimental support for CUDA in the DPC++ SYCL implementation.

CUDA is a registered trademark of NVIDIA Corporation
SYCL is a trademark of the Khronos Group Inc

Docker Image
-------------

There is a docker image available with all the examples and the required
environment set up, see https://hub.docker.com/r/ruyman/dpcpp_cuda_examples.

If you have nvidia-docker, you can simply pull the image and run it to build
the examples:

```sh
$ sudo docker run --gpus all -it ruyman/dpcpp_cuda_examples
```

Once inside the docker image, navigate to /home/examples/ to find a clone 
of this repo. Make sure to pull the latest changes:

```sh
$ cd /home/examples/SYCL-For-CUDA-Examples
$ git pull
```

Refer to each example and/or exercise for detailed instructions on how 
to run it.
