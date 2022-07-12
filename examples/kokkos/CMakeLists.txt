cmake_minimum_required (VERSION 3.10)
cmake_policy(SET CMP0074 NEW)
project (Kokkos_Test_Case)

set(Kokkos_DIR "$ENV{Kokkos_ROOT}" CACHE STRING "Kokkos root directory")
find_package(Kokkos REQUIRED)

add_executable(test_case test_case.cpp)
target_link_libraries(test_case Kokkos::kokkos)


