#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <chrono.hpp>
#include <common.hpp>

int main(int argc, char *argv[]) {
    using T = float;
    size_t n_laps = 30;
    size_t mat_size = 16384;
    if (argc > 1) {
        mat_size = std::stoul(argv[1], nullptr, 10);
    }
    T alpha = 1, beta = 0; // gemm parameters

    sycl::queue my_queue = try_get_queue(cuda_selector{});

    std::cout << "Initalizing the matrices..." << std::endl;
    size_t n = mat_size, m = mat_size, k = mat_size, ldA = mat_size, ldB = mat_size, ldC = mat_size;
    std::vector<T> A(mat_size * mat_size);
    std::vector<T> B(mat_size * mat_size);
    std::vector<T> C(mat_size * mat_size);
    fill_rand(A);
    fill_rand(B);

    // create sycl buffers of matrix data for offloading between device and host
    sycl::buffer<T, 1> A_buffer(A.data(), A.size());
    sycl::buffer<T, 1> B_buffer(B.data(), B.size());
    sycl::buffer<T, 1> C_buffer(C.data(), C.size());

    std::cout << "Running on:" << my_queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    Chrono c("computing + error handling");
    for (size_t i = 0; i < n_laps; i++) {
        std::cout << i << '/' << n_laps << '\n';
        // add oneapi::mkl::blas::gemm to execution queue and catch any synchronous exceptions
        try {
            using oneapi::mkl::transpose;
            using oneapi::mkl::blas::column_major::gemm; // row_major not implemented on cublas
            // C <- alpha*OP(A)*OP(B) + beta*C
            gemm(my_queue, transpose::nontrans, transpose::nontrans, m, n, k, alpha, A_buffer, ldA, B_buffer, ldB, beta,
                 C_buffer, ldC);
        }
        catch (sycl::exception const &e) {
            std::cout << "Caught synchronous SYCL exception during GEMM: " << e.what() << std::endl;
        }
        catch (std::exception const &e) {
            std::cout << "Caught synchronous STL exception during GEMM: " << e.what() << std::endl;
        }
        // ensure any asynchronous exceptions caught are handled before proceeding
        my_queue.wait_and_throw();
    }
    uint64_t operations_performed = n_laps * mat_size * mat_size * (2 * mat_size - 1);
    std::cout << "Gflops : " << operations_performed / 1000000000 / c.stop() << std::endl;

    return 0;
}