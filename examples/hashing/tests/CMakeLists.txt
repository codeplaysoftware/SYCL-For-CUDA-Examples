include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


enable_testing()

add_executable(
        test_all_hashes
        tests/tests.cpp
)

add_sycl_to_target(TARGET test_all_hashes SOURCES tests/tests.cpp)

target_link_libraries(test_all_hashes PUBLIC gtest_main sycl_hash)

include(GoogleTest)
gtest_discover_tests(test_all_hashes)