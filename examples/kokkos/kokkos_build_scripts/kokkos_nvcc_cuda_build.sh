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

cmake $KOKKOS_SOURCE_DIR -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_CUDA_COMPILER=nvcc \
      -DCMAKE_INSTALL_PREFIX=$KOKKOS_INSTALL_DIR \
      -DKokkos_CXX_STANDARD=17 \
      -DKokkos_ENABLE_SYCL=OFF \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_HSW=ON \
      -DKokkos_ARCH_AMPERE80=ON \
      -DKokkos_ENABLE_HWLOC=ON \
      -DKokkos_ENABLE_UNSUPPORTED_ARCHS=ON \
      -DKokkos_ENABLE_TESTS=OFF \
      -DKokkos_HWLOC_DIR=$HWLOC_DIR

ninja install

cd ..
