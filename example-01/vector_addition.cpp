
#include <algorithm>
#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

class CUDASelector : public cl::sycl::device_selector {
public:
  int operator()(const cl::sycl::device &Device) const override {
    using namespace cl::sycl::info;

    const std::string DriverVersion = Device.get_info<device::driver_version>();

    if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos)) {
      std::cout << " CUDA device found " << std::endl;
      return 1;
    };
    return -1;
  }
};

class vec_add;
int main(int argc, char *argv[]) {
  constexpr const size_t N = 100000;
  const sycl::range VecSize{N};

  sycl::buffer<double> bufA{VecSize};
  sycl::buffer<double> bufB{VecSize};
  sycl::buffer<double> bufC{VecSize};

  // Initialize input data
  {
    const auto dwrite_t = sycl::access::mode::discard_write;

    auto h_a = bufA.get_access<dwrite_t>();
    auto h_b = bufB.get_access<dwrite_t>();
    for (int i = 0; i < N; i++) {
      h_a[i] = sin(i) * sin(i);
      h_b[i] = cos(i) * cos(i);
    }
  }

  sycl::queue myQueue{CUDASelector()};

  // Command Group creation
  auto cg = [&](sycl::handler &h) {
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;

    auto a = bufA.get_access<read_t>(h);
    auto b = bufB.get_access<read_t>(h);
    auto c = bufC.get_access<write_t>(h);

    h.parallel_for<vec_add>(VecSize,
                            [=](sycl::id<1> i) { c[i] = a[i] + b[i]; });
  };

  myQueue.submit(cg);

  {
    const auto write_t = sycl::access::mode::read;
    auto h_c = bufC.get_access<write_t>();
    double sum = 0.0f;
    for (int i = 0; i < N; i++) {
      sum += h_c[i];
    }
    std::cout << "Sum is : " << sum << std::endl;
  }

  return 0;
}
