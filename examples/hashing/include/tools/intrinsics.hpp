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
 *  intrinsics.hpp
 *
 *  Description:
 *    Intrinsic operations for hashing functions
 **************************************************************************/
/**
    Copyright 2021 Codeplay Software Ltd.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use these files except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    For your convenience, a copy of the License has been included in this
    repository.

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
 */

#pragma once

#include <sycl/sycl.hpp>

namespace sbb {

    template<typename T>
    static inline std::enable_if_t<std::is_same_v<T, std::byte> || std::is_same_v<T, unsigned char>, uint32_t>
    upsample(const T hi_hi, const T hi, const T lo, const T lo_lo) {
        uint16_t hi_upsampled = (uint16_t(hi_hi) << 8) + uint16_t(hi);
        uint16_t lo_upsampled = (uint16_t(lo) << 8) + uint16_t(lo_lo);
        return (uint32_t(hi_upsampled) << 16) + uint32_t(lo_upsampled);
    }


    static inline sycl::event memcpy_with_dependency(sycl::queue &q, void *dest, const void *src, size_t numBytes, sycl::event depEvent) {
        return q.submit([=](sycl::handler &cgh) {
            cgh.depends_on(depEvent);
            cgh.memcpy(dest, src, numBytes);
        });
    }

    static inline sycl::event memcpy_with_dependency(sycl::queue &q, void *dest, const void *src, size_t numBytes, const std::vector<sycl::event> &depEvent) {
        return q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depEvent);
            cgh.memcpy(dest, src, numBytes);
        });
    }


    template<typename T>
    static inline constexpr uint8_t get_byte(const T &word, const uint &idx) {
        static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);
        return (word >> (8 * idx)) & 0xFF;
    }

    template<typename T>
    static inline constexpr T set_byte(const T &word, const uint8_t &byte_in, const uint &idx) {
        static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);
        T select_mask = ~(T(0xFF) << (idx * 8));
        T new_val = (T(byte_in) & 0xFF) << (idx * 8);
        return (word & select_mask) + new_val;
    }
}



