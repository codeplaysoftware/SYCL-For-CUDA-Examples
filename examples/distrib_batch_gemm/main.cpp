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
