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
 *  main.cpp
 *
 *  Description:
 *    Demonstrates simple vector addition
 **************************************************************************/
#include <array>
#include <iostream>

template <typename T, size_t N>
void simple_vadd_sycl(const std::array<T, N>& VA, const std::array<T, N>& VB,
                 std::array<T, N>& VC);

template <typename T, size_t N>
void simple_vadd_cuda(const std::array<T, N>& VA, const std::array<T, N>& VB,
                 std::array<T, N>& VC);

int main() {
  const size_t array_size = 4;
  std::array<int, array_size> A = {{1, 2, 3, 4}},
                                           B = {{1, 2, 3, 4}}, C;
  std::array<float, array_size> D = {{1.f, 2.f, 3.f, 4.f}},
                                             E = {{1.f, 2.f, 3.f, 4.f}}, F;
  simple_vadd_sycl(A, B, C);
  simple_vadd_cuda(D, E, F);
  for (unsigned int i = 0; i < array_size; i++) {
    if (C[i] != A[i] + B[i]) {
      std::cout << "The results are incorrect (element " << i << " is " << C[i]
                << "!\n";
      return 1;
    }
    if (F[i] != D[i] + E[i]) {
      std::cout << "The results are incorrect (element " << i << " is " << F[i]
                << "!\n";
      return 1;
    }
  }
  std::cout << "The results are correct!\n";
  return 0;
}
