set(A_SYCL_FOUND false)

find_package(hipSYCL CONFIG)

#[     {          "name": "My Compiler Kit",          "compilers":          {             "C": "/home/michel/sycl_workspace/deploy/bin/clang-13",             "CXX": "/home/michel/sycl_workspace/deploy/bin/clang++" },"environmentVariables":{"LD_PRELOAD":"/opt/intel/opencl/libOpenCL.so.1"}     } ]


if (hipSYCL_FOUND)
    set(A_SYCL_FOUND true)
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif ()

    cmake_policy(SET CMP0005 NEW)
    add_definitions(-DHIPSYCL_DEBUG_LEVEL=0)

    if (NOT HIPSYCL_DEBUG_LEVEL)
        if (CMAKE_BUILD_TYPE MATCHES "Debug")
            set(HIPSYCL_DEBUG_LEVEL 3 CACHE STRING
                    "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
                    FORCE)
        else ()
            set(HIPSYCL_DEBUG_LEVEL 2 CACHE STRING
                    "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
                    FORCE)
        endif ()
    endif ()
endif ()


if (ComputeCpp_DIR)
    include(cmake/Modules/FindComputeCpp.cmake)

    if (ComputeCpp_ROOT_DIR)
        set(A_SYCL_FOUND true)
    endif ()
    add_compile_definitions(USING_COMPUTECPP)
    message(STATUS " Using ComputeCpp CMake")
    message(STATUS " Path to ComputeCpp implementation: ${COMPUTECPP_PACKAGE_ROOT_DIR} ")
    #set(CMAKE_CXX_STANDARD 11)
    include(FindOpenCL)
endif ()


if (TRISYCL_INCLUDE_DIR AND NOT A_SYCL_FOUND)
    set(A_SYCL_FOUND true)
    message(STATUS " Using triSYCL CMake")
    include(FindTriSYCL)
endif ()

# We expect the DPCPP compiler to have used
if (NOT A_SYCL_FOUND)
    function(add_sycl_to_target arg1 arg2)
        target_compile_options(${arg2} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS} -sycl-std=2020 -std=c++20 -fsycl-unnamed-lambda>)
        target_link_options(${arg2} PRIVATE ${DPCPP_FLAGS} -sycl-std=2020 -std=c++20 -fsycl-unnamed-lambda)
    endfunction()

endif ()
