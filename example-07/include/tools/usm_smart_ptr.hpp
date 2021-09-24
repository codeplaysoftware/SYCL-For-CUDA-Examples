#pragma once

#include <sycl/sycl.hpp>
#include "../internal/config.hpp"
#include <memory>
#include <utility>
#include <type_traits>

#ifdef USING_COMPUTECPP
namespace cl::sycl::usm {
    using cl::sycl::experimental::usm::alloc;
}
#endif


#include "missing_implementations.hpp"


namespace usm_smart_ptr {
    using namespace sycl::usm;

    template<class T, sycl::usm::alloc Tag>
    struct usm_ptr {
    private:
        T *val_;
    public:
        explicit usm_ptr(T *t) : val_(t) {}

        /**
         * Explicit conversion needed if the memory is not shared
         * @return
         */
#ifdef IMPLICIT_MEMORY_COPY

        operator T *() const noexcept { return val_; }

#else

        explicit operator T *() const noexcept { return val_; }

#endif
        //explicit(Tag != sycl::usm::alloc::shared) operator T *() const noexcept { return val_; }

    };


    template<typename T>
    struct device_accessible_ptr {
    private:
        const T *val_;
    public:
        explicit device_accessible_ptr(T *p) : val_(p) {};

        explicit device_accessible_ptr(const T *p) : val_(p) {};

        device_accessible_ptr(usm_ptr<T, alloc::shared> p) : val_((T *) p) {};

        device_accessible_ptr(usm_ptr<T, alloc::device> p) : val_((T *) p) {};

        operator T *() const noexcept { return (T *) val_; }


    };

    template<typename T>
    struct host_accessible_ptr {
    private:
        T *val_;
    public:
        host_accessible_ptr(usm_ptr<T, alloc::shared> p) : val_((T *) p) {};

        host_accessible_ptr(usm_ptr<T, alloc::host> p) : val_((T *) p) {};

        operator T *() const noexcept { return val_; }


    };


/**
 * SYCL USM Deleter. The std::unique_ptr deleter takes only the pointer
 * to delete as an argument so that's the only work-around.
 */
    template<typename T>
    struct usm_deleter {
        sycl::queue q_;

        explicit usm_deleter(const sycl::queue &q) : q_(q) {}

        void operator()(T *ptr) const noexcept {
            if (ptr)
                sycl::free(ptr, q_);
        }
    };

/**
 * Wrapper for a std::unique_ptr that calls the SYCL deleter (sycl::free).
 * Also holds the number of elements allocated.
 */
    template<typename T, sycl::usm::alloc location>
    class usm_unique_ptr : public std::unique_ptr<T, usm_deleter<T>> {
    private:
        size_t count_;
    public:
        usm_unique_ptr(size_t count, sycl::queue q)
                : std::unique_ptr<T, usm_deleter<T>>(sycl::malloc<T>(count, q, location), usm_deleter<T>{q}) { count_ = count; }

        explicit usm_unique_ptr(sycl::queue q) :
                usm_unique_ptr(1, q) { count_ = 1; }


        [[nodiscard]] inline size_t alloc_size() const noexcept { return count_ * sizeof(T); }

        [[nodiscard]] inline size_t alloc_count() const noexcept { return count_; }

        [[nodiscard]] inline usm_ptr<T, location> get() const noexcept {
            return usm_ptr<T, location>(std::unique_ptr<T, usm_deleter<T>>::get());
        }

        [[nodiscard]] inline T *raw() const noexcept {
            return std::unique_ptr<T, usm_deleter<T>>::get();
        }

    };


/**
 * Same interface as usm_unique_ptr
 * @tparam T
 */
    template<typename T, sycl::usm::alloc location>
    class usm_shared_ptr : public std::shared_ptr<T> {
    private:
        size_t count_;

    public:
        usm_shared_ptr(size_t count, sycl::queue q) : std::shared_ptr<T>(sycl::malloc<T>(count, q, location), usm_deleter<T>{q}) { count_ = count; }

        explicit usm_shared_ptr(sycl::queue q) :
                usm_shared_ptr(1, q) { count_ = 1; }

        [[nodiscard]] inline size_t alloc_size() const noexcept { return count_ * sizeof(T); }

        [[nodiscard]] inline size_t alloc_count() const noexcept { return count_; }

        [[nodiscard]] inline usm_ptr<T, location> get() const noexcept {
            return usm_ptr<T, location>(std::shared_ptr<T>::get());
        }

        [[nodiscard]] inline T *raw() const noexcept {
            return std::shared_ptr<T>::get();
        }

    };

}