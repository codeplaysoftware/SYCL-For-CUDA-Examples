cmake_minimum_required(VERSION 3.9)
project(SYCL_HASH LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

option(VERBOSE_LIB "Adds various prints in the code" OFF)
if (VERBOSE_LIB)
    #message(WARNING "Verbose mode on. Did you forget it?")
    add_compile_definitions(VERBOSE_HASH_LIB)
endif ()

# If you're using the DPCPP compiler, these flags will be used. Set here the devies you want to target
set(DPCPP_FLAGS -fsycl -fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda -Xcuda-ptxas -v -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_50 -Wno-linker-warnings)
include(cmake/FindSYCL.cmake)

# Default C++ Flags for warnings and optimisation
set(WARNING_FLAGS "-Wall -Wextra -Wshadow -Wdouble-promotion -fno-common -Winit-self -Wuninitialized -Wmissing-declarations -Woverloaded-virtual")
set(EXTRA_W_FLAGS "-pedantic -Wall -Wextra -Wcast-align -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-declarations -Wmissing-include-dirs  -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5") #-Wnoexcept -Wold-style-cast -Wstrict-null-sentinel -switch-default -Wlogical-op
set(DISABLED_WARNINGS "-Wno-c++20-extensions -Wno-inline-namespace-reopened-noninline -Wno-undef -Wno-unused -Wno-unused-command-line-argument")
set(OPT_FLAGS "-march=native -mtune=native -Ofast -fomit-frame-pointer")

# Adding the flags to the targets
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS} ${DISABLED_WARNINGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} ${EXTRA_W_FLAGS} ${DISABLED_WARNINGS} -g -Og")

include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/include)


set(sycl_hash_all_kernels
        src/hash_functions/sha256.cpp
        src/hash_functions/blake2b.cpp
        src/hash_functions/sha1.cpp
        src/hash_functions/md5.cpp
        src/hash_functions/keccak.cpp
        src/hash_functions/md2.cpp
        src/tools/queue_tester.cpp
        )

set(sycl_hash_all_sources
        src/benchmarks/misc.hpp
        include/sycl_hash.hpp
        include/internal/config.hpp
        include/internal/handle.hpp
        include/internal/common.hpp
        include/internal/determine_kernel_config.hpp
        include/internal/sync_api.hpp
        include/internal/async_api.hpp
        include/hash_functions/sha256.hpp
        include/hash_functions/blake2b.hpp
        include/hash_functions/sha1.hpp
        include/hash_functions/md5.hpp
        include/hash_functions/keccak.hpp
        include/hash_functions/md2.hpp
        include/tools/chrono.hpp
        include/tools/fill_rand.hpp
        include/tools/sycl_queue_helpers.hpp
        include/tools/usm_smart_ptr.hpp
        )

add_library(sycl_hash SHARED ${sycl_hash_all_sources} ${sycl_hash_all_kernels})
add_sycl_to_target(TARGET sycl_hash SOURCES ${sycl_hash_all_kernels})

add_executable(demo demo_main.cpp src/benchmarks/misc.hpp)
target_link_libraries(demo PUBLIC sycl_hash)
add_sycl_to_target(TARGET demo SOURCES demo_main.cpp)

include(tests/CMakeLists.txt)
