#!/bin/bash
# Install Kokkos w/ sycl-cuda support

set -x #echo on

# Set:
# KOKKOS_INSTALL_DIR=[/your/install/dir]
# KOKKOS_SOURCE_DIR=[/your/source/dir]
# HWLOC_DIR=[/your/hwloc/dir]

# Configure & build kokkos
mkdir kokkos-build
cd kokkos-build

CXXFLAGS="-Xsycl-target-frontend -O3 -fgpu-inline-threshold=100000 -Wno-unknown-cuda-version -Wno-deprecated-declarations -Wno-linker-warnings -ffast-math" \
LDFLAGS="-Xsycl-target-frontend -O3" \
cmake $KOKKOS_SOURCE_DIR -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_INSTALL_PREFIX=$KOKKOS_INSTALL_DIR \
      -DKokkos_CXX_STANDARD=17 \
      -DKokkos_ENABLE_SYCL=ON \
      -DKokkos_ARCH_HSW=ON \
      -DKokkos_ARCH_AMPERE80=ON \
      -DKokkos_ENABLE_HWLOC=ON \
      -DKokkos_ENABLE_UNSUPPORTED_ARCHS=ON \
      -DKokkos_ENABLE_TESTS=OFF \
      -DKokkos_HWLOC_DIR=$HWLOC_DIR

ninja install

cd ..
