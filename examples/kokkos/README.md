Simple Test Case for Kokkos
----

This is a simple standalone test case taken from the Kokkos repository & packaged up here for use with SYCL.
It's doing a vector-matrix-vector product. It's an identity matrix with two vectors of 1s, so the expected answer
is just equal to the problem size.

Building the test case
-----

test_case.cpp contains a simple kernel which has been copied straight from the Kokkos Tutorials (Exercises/02/Solution).

Build it with build.sh, after setting the environment variable:
```
Kokkos_ROOT="[your/kokkos/installation]/lib/cmake/Kokkos"
```

Running the test case
----

Just launch it! There are optional flags:

-N : number of rows
-M : number of columns
-S : total size
-nrepeat : how many times to repeat the test (default 100)

Obviously, not all of N, M & S should be set. The test case will sanity check your args anyway.

Building Kokkos
------

In case you don't have an existing Kokkos build, there are some build scripts in `./kokkos_build_scripts`.
There are scripts for building Kokkos with SYCL, or CUDA (nvcc or clang).

Set the following environment variables:
```
KOKKOS_INSTALL_DIR=[/your/install/dir]
KOKKOS_SOURCE_DIR=[/your/source/dir]
HWLOC_DIR=[/your/hwloc/dir]
```

HWLOC
------

The [Portable Hardware Locality](https://www.open-mpi.org/projects/hwloc/) (hwloc) package is an optional dependency which enables Kokkos to query the hardware topology of the system on which it is running. If you do not have a HWLOC installation, this option can be removed & Kokkos will be built without HWLOC support.

SYCL backend
-------------

Kokkos should work with any SYCL backend, though the focus of this examples repo is SYCL-For-CUDA.
Previous work at Codeplay has involved running Kokkos with SYCL on Nvidia hardware with Ampere architecture, hence the flag:
```
      -DKokkos_ARCH_AMPERE80=ON \
```
This flag is not strictly necessary, but it enables Ahead of Time (AoT) compilation, which can give a significant performance gain when building large projects built on Kokkos.
You should modify the cmake command for your GPU arch.



