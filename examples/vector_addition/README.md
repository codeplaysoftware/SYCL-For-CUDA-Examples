Vector addition
===============================

This trivial example can be used to compare a simple vector addition in CUDA to
an equivalent implementation in SYCL for CUDA. The aim of the example is also 
to highlight how to build an application with SYCL for CUDA using DPC++ support, 
for which an example CMakefile is provided. For detailed documentation on how to
migrate from CUDA to SYCL, see [SYCL For CUDA Developers](https://developer.codeplay.com/products/computecpp/ce/guides/sycl-for-cuda-developers).

Note currently the CUDA backend does not support the [USM](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/USM.adoc) extension, so we use
`sycl::buffer` and `sycl::accessors` instead.

Pre-requisites
---------------

These instructions assume that example [docker image](https://hub.docker.com/r/ruyman/dpcpp_cuda_examples/dockerfile) is being used. This image 
simplifies accessing these examples as the environment is set up correctly.
For details on how to get started with the example docker image, refer to the 
root README file.

Building the example
---------------------

``` sh
$ mkdir build && cd build
$ cmake ../ -DSYCL_ROOT=${SYCL_ROOT_DIR} -DCMAKE_CXX_COMPILER=${SYCL_ROOT_DIR}/bin/clang++
$ make -j 8
```

This should produce two binaries, `vector_addition` and `sycl_vector_addition` .
The former is the unmodified CUDA source and the second is the SYCL for CUDA
version.

Running the example
--------------------

``` 
$ ./sycl_vector_addition
$ ./vector_addition
```

CMake Build script
------------------------

The provided CMake build script uses the native CUDA support to build the
CUDA application. It also serves as a check that all CUDA requirements
on the system are available (such as an installation of CUDA on the system).

Two flags are required: `-DSYCL_ROOT` , which must point to the place where the
DPC++ compiler is installed, and `-DCMAKE_CXX_COMPILER` , which must point to
the Clang compiler provided by DPC++. 

The CMake target `sycl_vector_addition` will build the SYCL version of
the application.

Note the variable `SYCL_FLAGS` is used to store the Clang flags that enable
the compilation of a SYCL application ( `-fsycl` ) but also the flag that specify
which targets are built ( `-fsycl-targets` ). In this case, we will build the example 
for both NVPTX and SPIR64. This means the kernel for the vector addition will be 
compiled for both backends, and runtime selection to the right queue will 
decide which variant to use.

Note the project is built with C++17 support, which enables the usage of
[deduction guides](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/deduction_guides/SYCL_INTEL_deduction_guides.asciidoc) to reduce the number of template parameters used.

SYCL Vector Addition code
--------------------------

The vector addition example uses a simple approach to implement with a plain
kernel that performs the add. Vectors are stored directly in buffers. Data is
initialized on the host using host accessors. This approach avoids creating
unnecessary storage on the host, and facilitates the SYCL runtime to use
optimized memory paths.

The SYCL queue created later on uses a custom `CUDASelector` to select a CUDA
device, or bail out if its not there. The CUDA selector uses the
`info::device::driver_version` to identify the device exported by the CUDA
backend. If the NVIDIA OpenCL implementation is available on the system, it
will be reported as another SYCL device. The driver version is the best way to
differentiate between the two.

The command group is created as a lambda expression that takes the 
`sycl::handler` parameter. Accessors are obtained from buffers using the
`get_access` method. Finally the `parallel_for` with the SYCL kernel is invoked
as usual.

The command group is subm$ itted to a queue which will convert all the operations
into CUDA commands that will be executed once the host accessor is encountered
later on.

The host accessor will trigger a copy of the data back to the host, and then
the values are reduced into a single sum element.
