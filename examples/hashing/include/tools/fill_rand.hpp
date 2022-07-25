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
 *  fill_rand.hpp
 *
 *  Description:
 *    Random generation for containers
 **************************************************************************/
#pragma once

#include <sycl/sycl.hpp>
#include <random>
#include <algorithm>
#include <type_traits>
#include "usm_smart_ptr.hpp"

using namespace usm_smart_ptr;

/**
 * Fills a container/array with random numbers from positions first to last
 */
template<typename T, class ForwardIt>
static inline void do_fill_rand_on_host(ForwardIt first, ForwardIt last) {
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
template<typename T>
static inline void fill_rand(host_accessible_ptr<T> v, size_t count) {
    do_fill_rand_on_host<T>((T *) v, (T *) v + count);
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
static inline void fill_rand(std::vector<T> &v) {
    do_fill_rand_on_host<T>(v.begin(), v.end());
}
