#pragma once

#include <utility>
#include <iostream>
#include "config.hpp"
#include "../tools/usm_smart_ptr.hpp"


namespace hash {
    using namespace usm_smart_ptr;
    struct handle_item {
        usm_unique_ptr<byte, alloc::device> input_dev_data_;
        usm_unique_ptr<byte, alloc::device> output_dev_data_;
        sycl::event dev_e_;
    };

    /**
     * Holds unique pointers to the memory used by the different queues.
     * This object is thus not copyable.
     */
    class handle {
    private:
        std::vector<handle_item> items_{};
    public:
        /**
         * Move constructor.
         */
        explicit handle(std::vector<handle_item> &&input) noexcept:
                items_(std::move(input)) {
        }

        handle() = default;

        /**
         * Rule of five, we need to redefine it.
         */
        handle &operator=(handle &&other) noexcept {
            std::swap(items_, other.items_);
            return *this;
        }

        /**
         * Waits on all the events, then clears the vector
         * which results in freeing the USM allocated memory
         */
        void wait() {
            for (auto &worker: items_) {
                worker.dev_e_.wait();
            }
            items_.clear();
        }

        /**
         * Waits and throws on all the events, then clears the queue
         * which results in freeing the USM allocated memory
         */
        void wait_and_throw() {
            for (auto &worker: items_) {
                worker.dev_e_.wait_and_throw();
            }
            items_.clear();
        }


        /**
         * No copy constructor.
         */
        handle(const handle &) = delete;

        /**
         * No assignement operator.
         */
        handle &operator=(const handle) = delete;

        /**
         * We need to join all the SYCL kernels/events before freeing the memory they use.
         */
        ~handle() noexcept {
            if (!items_.empty()) {
                std::cerr << "Destroying handled that still holds data. Did you forget to call .wait()?\n";
                for (auto &e: items_) {
                    try {
                        e.dev_e_.wait_and_throw();
                    }
                    catch (sycl::exception const &e) {
                        std::cerr << "Caught asynchronous SYCL exception at handle destruction: " << e.what() << std::endl;
                    }
                    catch (std::exception const &e) {
                        std::cerr << "Caught asynchronous STL exception at handle destruction: " << e.what() << std::endl;
                    }
                }
                items_.clear();
            }
        }
    };
}