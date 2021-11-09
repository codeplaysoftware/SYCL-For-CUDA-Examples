cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
# Don't complain about empty CMAKE_CUDA_ARCHITECTURES
cmake_policy(SET CMP0104 OLD)

project(cmake_and_cuda LANGUAGES CXX CUDA)

include(CTest)

# SYCL installation
if (NOT SYCL_ROOT) 
  message(FATAL_ERROR "No SYCL installation detected")
endif(NOT SYCL_ROOT)

set(SYCL_INCLUDE_DIR "${SYCL_ROOT}/lib/clang/14.0.0/include/")
set(SYCL_LIB "${SYCL_ROOT}/lib/libsycl.so")
set(SYCL_FLAGS "-fsycl" 
      "-fsycl-targets=nvptx64-nvidia-cuda-sycldevice" 
      "-fsycl-unnamed-lambda")

# Build the CUDA code
add_executable(vector_addition vector_addition.cu)
target_compile_features(vector_addition PUBLIC cxx_std_11)
set_target_properties(vector_addition PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET vector_addition PROPERTY BUILD_RPATH "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")

# Build the SYCL code
add_executable (sycl_vector_addition vector_addition.cpp)
target_compile_features(sycl_vector_addition PUBLIC cxx_std_17)
target_compile_options(sycl_vector_addition PUBLIC ${SYCL_FLAGS})
target_link_libraries(sycl_vector_addition PUBLIC ${SYCL_FLAGS})
target_include_directories(sycl_vector_addition PUBLIC ${SYCL_INCLUDE_DIR})
target_link_libraries(sycl_vector_addition PUBLIC ${SYCL_LIB})


# Build the SYCL (USM) code
add_executable (sycl_vector_addition_usm vector_addition_usm.cpp)
target_compile_features(sycl_vector_addition_usm PUBLIC cxx_std_17)
target_compile_options(sycl_vector_addition_usm PUBLIC ${SYCL_FLAGS})
target_link_libraries(sycl_vector_addition_usm PUBLIC ${SYCL_FLAGS})
target_include_directories(sycl_vector_addition_usm PUBLIC ${SYCL_INCLUDE_DIR})
target_link_libraries(sycl_vector_addition_usm PUBLIC ${SYCL_LIB})

