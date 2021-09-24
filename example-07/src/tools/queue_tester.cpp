#include <sycl/sycl.hpp>
#include <tools/sycl_queue_helpers.hpp>

void queue_tester(sycl::queue &q) {
    q.submit([](sycl::handler &cgh) {
        cgh.single_task<class queue_kernel_tester>([]() {});
    }).wait_and_throw();
}