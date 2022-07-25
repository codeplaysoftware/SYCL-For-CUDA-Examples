/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's SYCL-For-CUDA-Examples
 *
 *  sycl_queue_helpers.hpp
 *
 *  Description:
 *    Helper functions relating to SYCL queues
 **************************************************************************/
#pragma once

#include <sycl/sycl.hpp>
#include <iostream>
#include "../internal/common.hpp"

#ifdef USING_COMPUTECPP
class queue_kernel_tester;
namespace cl::sycl::usm{
    using cl::sycl::experimental::usm::alloc;
}
#endif

/**
 * Selects a CUDA device (but returns sometimes an invalid one)
 */
class cuda_selector : public sycl::device_selector {
public:
    int operator()(const sycl::device &device) const override {
#if defined(SYCL_IMPLEMENTATION_ONEAPI) || defined(SYCL_IMPLEMENTATION_INTEL)
        return device.get_platform().get_backend() == sycl::backend::ext_oneapi_cuda && device.get_info<sycl::info::device::is_available>() ? 1 : -1;
#else
        return device.is_gpu() && (device.get_info<sycl::info::device::name>().find("NVIDIA") != std::string::npos) ? 1 : -1;
#endif
    }
};


void queue_tester(sycl::queue &q);


/**
 * Tries to get a queue from a selector else returns the host device
 * @tparam strict if true will check whether the queue can run a trivial task which implied
 * that the translation unit needs to be compiler with support for the device you're selecting.
 */
template<bool strict = true, typename T>
inline sycl::queue try_get_queue(const T &selector) {
    auto exception_handler = [](const sycl::exception_list &exceptions) {
        for (std::exception_ptr const &e: exceptions) {
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

    sycl::device dev;
    sycl::queue q;
    try {
        dev = sycl::device(selector);
        q = sycl::queue(dev, exception_handler);

        try {
            if constexpr (strict) {
                if (dev.is_cpu() || dev.is_gpu()) { //Only CPU and GPU not host, dsp, fpga, ?...
                    queue_tester(q);
                }
            }
        } catch (...) {
            std::cerr << "Warning: " << dev.get_info<sycl::info::device::name>() << " found but not working! Fall back on: ";
            dev = sycl::device(sycl::host_selector());
            q = sycl::queue(dev, exception_handler);
            std::cerr << dev.get_info<sycl::info::device::name>() << '\n';
            return q;
        }
    }
    catch (...) {

        dev = sycl::device(sycl::host_selector());
        q = sycl::queue(dev, exception_handler);
        std::cerr << "Warning: Expected device not found! Fall back on: " << dev.get_info<sycl::info::device::name>() << '\n';
    }
    return q;
}

#if defined(__linux__) || defined(__APPLE__) || defined(__LINUX__)

#include <sys/mman.h>
#include <unistd.h>

/**
 * Checks whether a pointer was allocated on the host device as the pointer query is not reliable on DPC++ on the host.
 * @see http://si-head.nl/articles/msync
 * @return Wether the memory was allocated on the host OS.
 */
template<typename T>
inline bool valid_pointer(T *p) {
    // Get page size and calculate page mask
    auto pagesz = (size_t) sysconf(_SC_PAGESIZE);
    size_t pagemask = ~(pagesz - 1);
    // Calculate base address
    void *base = (void *) (((size_t) p) & pagemask);
    return msync(base, sizeof(T), MS_ASYNC) == 0;
}

#else
template<typename T>
inline bool valid_pointer(T *p) {
    return false;
}
#endif


template<typename T, bool debug = false>
inline bool is_ptr_usable([[maybe_unused]] const T *ptr, [[maybe_unused]] const sycl::queue &q) {
    if (q.get_device().is_host()) {
        return valid_pointer(ptr);
    }

    try {
        sycl::get_pointer_device(ptr, q.get_context());
        sycl::usm::alloc alloc_type = sycl::get_pointer_type(ptr, q.get_context());
        if constexpr(debug) {
            std::cerr << "Allocated on:" << q.get_device().get_info<sycl::info::device::name>() << " USM type: ";
            switch (alloc_type) {
                case sycl::usm::alloc::host:
                    std::cerr << "alloc::host" << '\n';
                    break;
                case sycl::usm::alloc::device:
                    std::cerr << "alloc::device" << '\n';
                    break;
                case sycl::usm::alloc::shared:
                    std::cerr << "alloc::shared" << '\n';
                    break;
                case sycl::usm::alloc::unknown:
                    std::cerr << "alloc::unknown" << '\n';
                    break;
            }
        }
        return alloc_type == sycl::usm::alloc::shared // Shared memory is ok
               || alloc_type == sycl::usm::alloc::device // Device memory is ok
               || (alloc_type == sycl::usm::alloc::host && q.get_device().is_cpu()) // We discard host allocated memory because of poor performance unless on the CPU
                ;
    } catch (...) {
        if constexpr (debug) {
            std::cerr << "Not allocated on:" << q.get_device().get_info<sycl::info::device::name>() << '\n';
        }
        return false;
    }

}


/**
 * Usefull for memory bound computation.
 * Returns CPU devices that represents different numa nodes.
 * @return
 */
/* inline hash::runners get_cpu_runners_numa() {
    try {
        sycl::device d{sycl::cpu_selector{}};
        auto numa_nodes = d.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);
        hash::runners runners_;
        std::transform(numa_nodes.begin(), numa_nodes.end(), runners_.begin(), [](auto &dev) -> hash::runner { return {try_get_queue(dev), 1}; });
        return runners_;
    }
    catch (...) {
        return {{sycl::queue{sycl::host_selector{}}, 1}};
    }
} */
