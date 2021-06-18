CUDA Frotran and SYCL integration
======================================

This directory shows an example of how to call a SYCL function
from a CUDA fortran code.

The SYCL routine is called using the Fortran ISO bindings like
any other C function.

```fortran
interface saxpy_sycl
  subroutine saxpy_call(x, y, a, N) &
    bind(C,name='saxpy_sycl_cuda_wrapper')
    implicit none
    real :: x(:), y(:)
    real, value :: a
    integer, value :: N
  end subroutine
end interface
```

The SYCL code implemented in the C++ version of the code works as usual with one minor modification:
Uses the CUDA Primary context to enable inter-operating with the CUDA Fortran code, ensuring the same resources are shared.

The following snipped highligts the construction of a SYCL context associated with the Primary context.

```cpp
sycl::context c{sycl::property::context::cuda::use_primary_context()};
sycl::queue q{c, c.get_devices()[0]};
```


