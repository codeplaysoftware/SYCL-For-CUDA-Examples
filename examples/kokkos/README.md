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

Building kokkos
------

In case you don't have an existing Kokkos build, there are some build scripts in `./kokkos_build_scripts`.
There are scripts for building Kokkos with SYCL, or CUDA (nvcc or clang).

Set the following environment variables:
```
KOKKOS_INSTALL_DIR=[/your/install/dir]
KOKKOS_SOURCE_DIR=[/your/source/dir]
HWLOC_DIR=[/your/hwloc/dir]
```
**Note**: if you do not have a HWLOC installation, this option can be removed & Kokkos will be built without HWLOC support.

You should modify the cmake command for your GPU arch. The current value:
```
      -DKokkos_ARCH_AMPERE80=ON \
```
represents CUDA's Ampere architecture (SM_80).



