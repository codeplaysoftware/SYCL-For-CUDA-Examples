#pragma once

#include <sycl/sycl.hpp>
#include <cstddef>
#include <utility>

namespace hash::internal {


    struct kernel_config {
        size_t wg_size;
        size_t block;
    };


    inline kernel_config get_kernel_sizes(const sycl::queue &q, size_t job_size) {
        kernel_config config{.wg_size= 1, .block= job_size};
        if (q.get_device().is_gpu()) {
            /**
             * If the device is a GPU we will try to have as many threads in each work group as possible.
             * We need to bound the value of `max_work_group_size` as it can be ANY 64-bit integer
             */
            config.wg_size = std::min(std::max(1ul, 2 * q.get_device().get_info<sycl::info::device::max_work_group_size>()), job_size);
            config.wg_size = std::min(config.wg_size, 64ul); //TODO Find a better alternative than a hardcoded 64 ?
            config.block = (job_size / config.wg_size) + (job_size % config.wg_size != 0);
        } else {
            /**
             * We need that case because on a CPU, one work group runs on one thread, and threads are expensive to launch
             * We'll multiply the thread count by a factor in order to allow the scheduler to better balance the work load.
             */
            config.block = std::min((size_t) std::max(1u, 2 * q.get_device().get_info<sycl::info::device::max_compute_units>()), job_size);
            config.wg_size = job_size / config.block + (job_size % config.block != 0);

            /* We check that the work groups are not too big */
            size_t max_wg_size = std::min(std::max(1ul, q.get_device().get_info<sycl::info::device::max_work_group_size>()), job_size);
            if (config.wg_size > max_wg_size) {
                config.wg_size = max_wg_size;
                config.block = (job_size / config.wg_size) + (job_size % config.wg_size != 0);
            }

        }
        assert(config.block * config.wg_size >= job_size);
        return config;
    }

}