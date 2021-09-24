#pragma once

#include <sycl_hash.hpp>
#include <iomanip>
#include <iostream>
#include <cstring>
#include <tools/sycl_queue_helpers.hpp>

template<bool strict = true>
static inline sycl::queue try_get_queue_with_device(const sycl::device &in_dev) {
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

    sycl::device dev;
    sycl::queue q;
    try {
        dev = in_dev;
        q = sycl::queue(dev, exception_handler);
        if constexpr (strict) {
            if (dev.is_cpu() || dev.is_gpu()) { //Only CPU and GPU not host, dsp, fpga, ?...
                queue_tester(q);
            }
        }
    }
    catch (...) {
        dev = sycl::device(sycl::host_selector());
        q = sycl::queue(dev, exception_handler);
        std::cout << "Warning: Expected device not found! Fall back on: " << dev.get_info<sycl::info::device::name>() << std::endl;
    }
    return q;
}


void print_hex(byte *ptr, dword len) {
    for (size_t i = 0; i < len; ++i) // only the first block
        std::cout << std::hex << std::setfill('0') << std::setw(2) << (int) (ptr[i]) << " ";
    std::cout << std::dec << std::endl << std::endl;
}

void duplicate(byte *in, byte *out, dword item_len, dword count) {
    for (size_t i = 0; i < count; ++i) {
        std::memcpy(out + item_len * i, in, item_len);
    }
}



std::vector<sycl::queue> get_all_queues_once() {
    std::vector<sycl::device> devices1 = sycl::device::get_devices();
    std::vector<sycl::queue> queues1;
    std::for_each(devices1.begin(), devices1.end(), [&](auto &d) { queues1.emplace_back(try_get_queue_with_device(d)); });
    return queues1;
}


std::vector<sycl::queue> get_all_queues(){
   static std::vector<sycl::queue> queues = get_all_queues_once();
   return queues;
}



template<typename Func>
void for_all_workers(Func f) {
    static auto  queues = get_all_queues();
    {
        for (const auto &q: queues) {
            std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
            f(hash::runners(1, hash::runner{q, 1}));
        }
    }
}


template<typename Func>
void for_all_workers_pairs(Func f) {
    auto queues = get_all_queues();
    for (const auto &q1: queues) {
        for (const auto &q2: queues) {
            std::cout << "Running on: " << q1.get_device().get_info<sycl::info::device::name>() << " and: " << q2.get_device().get_info<sycl::info::device::name>() << std::endl;
            f({{q1, 1},
               {q2, 1}});
        }

    }

}
