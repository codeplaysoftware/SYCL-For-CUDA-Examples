cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(sycl_cuda_interop LANGUAGES CXX CUDA)

find_package(CUDAToolkit)

# SYCL installation
if (NOT SYCL_ROOT) 
  message(FATAL_ERROR "No SYCL installation detected")
endif(NOT SYCL_ROOT)

set(SYCL_INCLUDE_DIR "${SYCL_ROOT}/lib/clang/11.0.0/include/")
set(SYCL_LIB "${SYCL_ROOT}/lib/libsycl.so")
set(SYCL_FLAGS "-fsycl" 
      "-fsycl-targets=nvptx64-nvidia-cuda-sycldevice"
      "-fsycl-unnamed-lambda")


# Build the CUDA code
add_executable(sgemm_cuda sgemm.cu)
target_compile_features(sgemm_cuda PUBLIC cxx_std_11)
set_target_properties(sgemm_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET sgemm_cuda PROPERTY BUILD_RPATH "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")
target_link_libraries(sgemm_cuda CUDA::toolkit CUDA::cublas)

# Build the SYCL code
add_executable (sycl_sgemm sycl_sgemm.cpp)
target_compile_features(sycl_sgemm PUBLIC cxx_std_17)
target_compile_options(sycl_sgemm PUBLIC ${SYCL_FLAGS})
target_compile_definitions(sycl_sgemm PUBLIC CUDA_NO_HALF)
target_link_libraries(sycl_sgemm PUBLIC ${SYCL_FLAGS})
target_include_directories(sycl_sgemm PUBLIC ${SYCL_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
target_link_libraries(sycl_sgemm PUBLIC CUDA::toolkit  CUDA::cublas)
