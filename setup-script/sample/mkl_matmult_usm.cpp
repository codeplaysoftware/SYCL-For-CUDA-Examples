#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <chrono.hpp>
#include <common.hpp>
#include <usm_smart_ptr.hpp>

using namespace usm_smart_ptr;

int main(int argc, char *argv[]) {
    using T = float;
    size_t n_laps = 30;
    size_t mat_size = 16384; // Bound by your GPU's memory.

    if (argc > 1) {
        mat_size = std::stoul(argv[1], nullptr, 10);
    }
    T alpha = 1, beta = 0; // gemm parameters

    sycl::queue my_queue = try_get_queue(cuda_selector{});

    std::cout << "Initalizing the matrices..." << std::endl;
    long n = mat_size, m = mat_size, k = mat_size, ldA = mat_size, ldB = mat_size, ldC = mat_size;
    // Initializing USM shared memory in an std::unique_ptr for auto mem management
    auto A = make_unique_ptr<T, alloc::shared>(mat_size * mat_size, my_queue);
    auto B = make_unique_ptr<T, alloc::shared>(mat_size * mat_size, my_queue);
    auto C = make_unique_ptr<T, alloc::device>(mat_size * mat_size, my_queue);
    fill_rand(A.get(), A.count());
    fill_rand(B.get(), B.count());

    std::cout << "Running on:" << my_queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    Chrono c("computing + error handling");

    try {
        sycl::event e;
        for (size_t i = 0; i < n_laps; i++) {
            std::cout << i << '/' << n_laps << '\n';
            using oneapi::mkl::transpose;
            using oneapi::mkl::blas::column_major::gemm;
            // C <- alpha*OP(A)*OP(B) + beta*C
            e = gemm(my_queue, transpose::nontrans, transpose::nontrans, m, n, k, alpha, A.get(), ldA, B.get(), ldB, beta, C.get(), ldC, {e});
        }
        e.wait_and_throw();
    }
    catch (sycl::exception const &e) {
        std::cout << "Caught synchronous SYCL exception during GEMM: " << e.what() << std::endl;
    }
    catch (std::exception const &e) {
        std::cout << "Caught synchronous STL exception during GEMM: " << e.what() << std::endl;
    }

    uint64_t operations_performed = n_laps * mat_size * mat_size * (2 * mat_size - 1);
    std::cout << "Gflops : " << operations_performed / 1000000000 / c.stop() << std::endl;

    return 0;
}