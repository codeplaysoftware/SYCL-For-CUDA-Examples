cmake_minimum_required(VERSION 3.9)
project(Sycl_Tests LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

set(WARNING_FLAGS "-Wall -Wextra -Wshadow -Wdouble-promotion -Wshadow -Wuninitialized -Wmissing-declarations -Woverloaded-virtual")
set(DISABLED_WARNINGS "-Wno-c++20-extensions -Wno-unknown-cuda-version -Wno-unused -Wno-unused-parameter")

set(OPT_FLAGS "-march=native -mtune=native -Ofast -fomit-frame-pointer")

SET(CMAKE_CXX_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda -sycl-std=2020 -fsycl-unnamed-lambda")
SET(CMAKE_EXE_LINKER_FLAGS "-lonemkl_blas_cublas -lonemkl")

set(CMAKE_CXX_FLAGS_RELEASE "${OPT_FLAGS} ${WARNING_FLAGS} ${DISABLED_WARNINGS}")
set(CMAKE_CXX_FLAGS_DEBUG " ${WARNING_FLAGS}  ${DISABLED_WARNINGS} -g3 -Og")

include_directories(include/)
include_directories($ENV{DPCPP_HOME}/deploy/include)
link_directories($ENV{DPCPP_HOME}/deploy/lib)

add_executable(mkl_kernel mkl_matmult.cpp)
add_executable(mkl_kernel_usm mkl_matmult_usm.cpp)

