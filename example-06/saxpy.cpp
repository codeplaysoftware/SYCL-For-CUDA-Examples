#include <iostream>
#include <CL/sycl.hpp>

extern "C" {
	void saxpy_sycl_cuda_wrapper (float* x, float* y, float a, int N);
};


void saxpy_sycl_cuda_wrapper (float* x, float* y, float a, int N) {
	sycl::context c{sycl::property::context::cuda::use_primary_context()};
	sycl::queue q{c, c.get_devices()[0]};
	{
		sycl::buffer bX {x, sycl::range<1>(N)};

		q.submit([&](sycl::handler& h) {
			auto aX = bX.get_access<sycl::access::mode::read_write>(h);
			h.single_task([=]() {
				aX[0] = 3.f;
			});
		});

		q.wait_and_throw();
	}
	return;
}
