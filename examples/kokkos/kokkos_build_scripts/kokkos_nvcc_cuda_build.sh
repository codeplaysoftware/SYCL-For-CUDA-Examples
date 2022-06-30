#!/bin/bash
# Install Kokkos w/ sycl-cuda support

set -x #echo on

module load ompi

KOKKOS_ROOT=$HOME/Kokkos
KOKKOS_SOURCE_DIR=$KOKKOS_ROOT/kokkos

# Configure & build kokkos
mkdir kokkos-build
cd kokkos-build

# CXXFLAGS="-Xsycl-target-frontend -O3" \
# LDFLAGS="-Xsycl-target-frontend -O3" \
cmake $KOKKOS_SOURCE_DIR -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_CUDA_COMPILER=nvcc \
      -DCMAKE_INSTALL_PREFIX=$KOKKOS_ROOT/kokkos_nvcc_cuda_install \
      -DKokkos_CXX_STANDARD=17 \
      -DKokkos_ENABLE_SYCL=OFF \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_HSW=ON \
      -DKokkos_ARCH_AMPERE80=ON \
      -DKokkos_ENABLE_HWLOC=ON \
      -DKokkos_ENABLE_UNSUPPORTED_ARCHS=ON \
      -DKokkos_ENABLE_TESTS=OFF \
      -DKokkos_HWLOC_DIR=$HOME/soft/ompi

ninja install

cd ..
