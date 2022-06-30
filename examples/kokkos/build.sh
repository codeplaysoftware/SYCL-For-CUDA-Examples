#!/bin/bash

rm -rf build
mkdir build
cd build

# Set the environment variable Kokkos_ROOT="[your/kokkos/installation]/lib/cmake/Kokkos"
CXXFLAGS="-Xsycl-target-frontend -O3" \
LDFLAGS="-Xsycl-target-frontend -O3" \
cmake .. -G Ninja \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_COMPILER=clang

ninja

cd ..
