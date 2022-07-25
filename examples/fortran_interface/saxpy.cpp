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
 *  saxpy.cpp
 *
 *  Description:
 *    SAXPY in SYCL
 **************************************************************************/
#include <iostream>
#include <CL/sycl.hpp>

extern "C" {
   void saxpy_sycl_cuda_wrapper (float* x, float* y, float a, int N);
};


void saxpy_sycl_cuda_wrapper (float* x, float* y, float a, int N) {
   sycl::context c{sycl::property::context::cuda::use_primary_context()};
   sycl::queue q{c, c.get_devices()[0], sycl::property::queue::cuda::use_default_stream()};
   {
      sycl::buffer bX {x, sycl::range<1>(N)};
      sycl::buffer bY {y, sycl::range<1>(N)};

      q.submit([&](sycl::handler& h) {
         auto aX = bX.get_access<sycl::access::mode::read_write>(h);
         auto aY = bY.get_access<sycl::access::mode::read_write>(h);
         h.parallel_for<class saxpy_kernel>(sycl::range<1>(N), [=](sycl::id<1> id) {
            if (id[0] < N) 
               aY[id] = aX[id] * a + aY[id];
         });
      });

      q.wait_and_throw();
   }
   return;
}
