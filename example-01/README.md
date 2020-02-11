Example 01: Vector addition 
===============================

This trivial example can be used to compare a simple vector addition in 
CUDA to an equivalent implementation in SYCL for CUDA.
The aim of the example is also to highlight how to build an application
with SYCL for CUDA using DPC++ support, for which an example CMakefile is
provided.a
For detailed documentation on how to migrate from CUDA to SYCL, see 
[SYCL For CUDA Developers](https://developer.codeplay.com/products/computecpp/ce/guides/sycl-for-cuda-developers).

Note currently the CUDA backend doesnt support 
[USM](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/USM.adoc) 
extension, so we use `sycl::buffer` and `sycl::accessors` instead.

Pre-requisites
---------------

#. An installation of DPC++ with CUDA support, see [Get Started Guide](https://github.com/codeplaysoftware/sycl-for-cuda/blob/cuda/sycl/doc/GetStartedWithSYCLCompiler.md) for details on how to build it.
##. The branch used for this example is cuda-dev. Note this branch is in active
development and your mileage may vary.

Building the example
---------------------

1. mkdir build; cd build
2. cmake ../ -DSYCL_ROOT=/path/to/dpc++/install \
    -DCMAKE_CXX_COMPILER=/path/to/dpc++/install/bin/clang++
3. make -j 8

This should produce two binaries, `vector_addition` and `sycl_vector_addition`.
The former is the unmodified CUDA source and the second is the SYCL for CUDA
version.

Running the example
--------------------

The path to `libsycl.so` and the PI plugins must be in `LD_LIBRARY_PATH`.
A simple way of running the app is as follows:

```
$ LD_LIBRARY_PATH=$HOME/open-source/sycl4cuda/lib  ./sycl_vector_addition
```

Note the `SYCL_BE` env variable is not required, since we use a custom
device selector.

CMake Build script
------------------------

The provided CMake build script uses the native CUDA support to build the
CUDA application. It also serves as a check that all CUDA requirements
on the system are available (such as an installation of CUDA on the system).

Two flags are required: -DSYCL\_ROOT, which must point to the place where the
DPC++ compiler is installed, and -DCMAKE_CXX_COMPILER , which must point to
the Clang compiler provided by DPC++. 

The CMake target `sycl_vector_addition` will build the SYCL version of
the application.
Note the variable `SYCL_FLAGS` is used to store the Clang flags that enable
the compilation of a SYCL application (`-fsycl`) but also the flag that specify
which targets are built (`-fsycl-targets`).
In this case, we will build the example for both NVPTX and SPIR64. 
This means the kernel for the vector addition will be compiled for both
backends, and runtime selection to the right queue will decide which variant
to use.

Note the project is built with C++17 support, which enables the usage of
[deduction guides](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/deduction_guides/SYCL_INTEL_deduction_guides.asciidoc) to reduce the number of template parameters used.

SYCL Vector Addition code
--------------------------

The vector addition example uses a simple approach to implement with a plain
kernel that performs the add. Vectors are stored directly in buffers.
Data is initialized on the host using host accessors. 
This approach avoids creating unnecesary storage on the host, and facilitates
the SYCL runtime to use optimized memory paths.

The SYCL queue created later on uses a custom `CUDASelector` to select
a CUDA device, or bail out if its not there. 
The CUDA selector uses the `info::device::driver_version` to identify the 
device exported by the CUDA backend.
This is important, 'cos if the NVIDIA OpenCL implementation is available on the
system, it will be reported as another SYCL device, so checking the driver 
version is the best way to differentiate between the two.

The command group is created as a lambda expression that takes the
`sycl::handler` parameter. Accessors are obtained from buffers using the
`get_access` method.
Finally the `parallel_for` with the SYCL kernel is invoked as usual.

The command group is submitted to a queue which will convert all the 
operations into CUDA commands that will be executed once the host accessor
is encountered later on.

The host accessor will trigger a copy of the data back to the host, an
then the values are reduced into a single sum element.

