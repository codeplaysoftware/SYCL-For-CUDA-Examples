#pragma once

#include <sycl/sycl.hpp>
#include <random>
#include <algorithm>
#include <type_traits>
#include <usm_smart_ptr.hpp>

using namespace usm_smart_ptr;

class cuda_selector : public sycl::device_selector {
public:
    int operator()(const sycl::device &device) const override {
        return device.get_platform().get_backend() == sycl::backend::cuda;
        //return device.is_gpu() && (device.get_info<sycl::info::device::driver_version>().find("CUDA") != std::string::npos);
    }
};

/**
 * Tries to get a CUDA device else returns the host device
 */
sycl::queue try_get_queue(const sycl::device_selector &selector) {
    sycl::device dev;
    try {
        dev = sycl::device(selector);
    }
    catch (...) {
        dev = sycl::device(sycl::host_selector());
        std::cout << "Warning: GPU device not found! Fall back on: " << dev.get_info<sycl::info::device::name>()
                  << std::endl;
    }
    auto exception_handler = [](const sycl::exception_list &exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e) {
                std::cout << "Caught asynchronous SYCL exception: " << e.what() << std::endl;
            }
            catch (std::exception const &e) {
                std::cout << "Caught asynchronous STL exception: " << e.what() << std::endl;
            }
        }
    };

    return sycl::queue(dev, exception_handler);
}


/**
 * Fills a container/array with random numbers from positions first to last
 */
template<typename T, class ForwardIt>
void do_fill_rand_on_host(ForwardIt first, ForwardIt last) {
    static std::random_device dev;
    static std::mt19937 engine(dev());
    auto generator = [&]() {
        if constexpr (std::is_integral<T>::value) {
            static std::uniform_int_distribution<T> distribution;
            return distribution(engine);
        } else if constexpr (std::is_floating_point<T>::value) {
            static std::uniform_real_distribution<T> distribution;
            return distribution(engine);
        } else if constexpr (std::is_same_v<T, sycl::half>) {
            static std::uniform_real_distribution<float> distribution;
            return distribution(engine);
        }
    };
    std::generate(first, last, generator);
}


/**
 * This function accepts only memory that is accessible from the CPU
 * To achive this it uses fantom types that wraps the pointer.
 * This could be done by calling the runtime to check where is the
 * usm memory allocated, but here we can avoid doing that.
 */
template<typename T, sycl::usm::alloc location>
typename std::enable_if<location == sycl::usm::alloc::host || location == sycl::usm::alloc::shared, void>::type
fill_rand(const usm_ptr<T, location> &v, size_t count) {
    do_fill_rand_on_host<T>(+v, v + count);
}

/**
 * This function would only accept device allocated memory
 */
/*template<typename T, sycl::usm::alloc location>
typename std::enable_if<location == sycl::usm::alloc::device, void>::type
fill_rand(const usm_ptr<T, location> &v, size_t count) {
    do_fill_rand_on_device<T>(+v, v + count);
}*/

template<typename T>
void fill_rand(std::vector<T> &v) {
    do_fill_rand_on_host<T>(v.begin(), v.end());
}
