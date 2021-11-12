cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
# Don't complain about empty CMAKE_CUDA_ARCHITECTURES
cmake_policy(SET CMP0104 OLD)

project(sycl_cuda_interop LANGUAGES CXX CUDA)

find_package(CUDAToolkit)

# SYCL installation
if (NOT SYCL_ROOT) 
  message(FATAL_ERROR "No SYCL installation detected")
endif(NOT SYCL_ROOT)

set(SYCL_INCLUDE_DIR "${SYCL_ROOT}/lib/clang/14.0.0/include/")
set(SYCL_LIB "${SYCL_ROOT}/lib/libsycl.so")
set(SYCL_FLAGS "-fsycl" 
      "-fsycl-targets=nvptx64-nvidia-cuda"
      "-fsycl-unnamed-lambda"
      "-Wno-linker-warnings")

# Build the CUDA code
add_executable(cuda_sgemv sgemv.cu)
target_compile_features(cuda_sgemv PUBLIC cxx_std_11)
set_target_properties(cuda_sgemv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cuda_sgemv PROPERTY BUILD_RPATH "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")
target_link_libraries(cuda_sgemv CUDA::toolkit CUDA::cublas)

add_executable (sycl_sgemv sycl_sgemv.cpp)
target_compile_features(sycl_sgemv PUBLIC cxx_std_17)
target_compile_options(sycl_sgemv PUBLIC ${SYCL_FLAGS})
target_compile_definitions(sycl_sgemv PUBLIC CUDA_NO_HALF)
target_link_libraries(sycl_sgemv PUBLIC ${SYCL_FLAGS})
target_include_directories(sycl_sgemv PUBLIC ${SYCL_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
target_link_libraries(sycl_sgemv PUBLIC CUDA::toolkit CUDA::cublas)
