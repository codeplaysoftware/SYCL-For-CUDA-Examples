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
 *  vadd_sycl.cpp
 *
 *  Description:
 *    Vector addition in SYCL
 **************************************************************************/
/* This example is a very small one designed to show how compact SYCL code
 * can be. That said, it includes no error checking and is rather terse. */
#include <CL/sycl.hpp>

#include <array>
#include <iostream>

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;

/* This is the class used to name the kernel for the runtime.
 * This must be done when the kernel is expressed as a lambda. */
template <typename T>
class SimpleVadd;

template <typename T, size_t N>
void simple_vadd_sycl(const std::array<T, N>& VA, const std::array<T, N>& VB,
                 std::array<T, N>& VC) {
  cl::sycl::queue deviceQueue;
  cl::sycl::range<1> numOfItems{N};
  cl::sycl::buffer<T, 1> bufferA(VA.data(), numOfItems);
  cl::sycl::buffer<T, 1> bufferB(VB.data(), numOfItems);
  cl::sycl::buffer<T, 1> bufferC(VC.data(), numOfItems);

  deviceQueue.submit([&](cl::sycl::handler& cgh) {
    auto accessorA = bufferA.template get_access<sycl_read>(cgh);
    auto accessorB = bufferB.template get_access<sycl_read>(cgh);
    auto accessorC = bufferC.template get_access<sycl_write>(cgh);

    auto kern = [=](cl::sycl::id<1> wiID) {
      accessorC[wiID] = accessorA[wiID] + accessorB[wiID];
    };
    cgh.parallel_for<class SimpleVadd<T>>(numOfItems, kern);
  });
}

template void simple_vadd_sycl<float, 4>(const std::array<float, 4>& VA, const std::array<float, 4>& VB,
                 std::array<float, 4>& VC);
template void simple_vadd_sycl<int, 4>(const std::array<int, 4>& VA, const std::array<int, 4>& VB,
                 std::array<int, 4>& VC);
                 