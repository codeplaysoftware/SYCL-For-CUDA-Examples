# oneMKL samples

The code runs a small benchmarks of your blas implementation using multiplication of square matrices. You can pass the size as an argument of the executable.

Two versions are provided, one of which is using the USM inferface.

If the environment is correctly set you should be able to run the sample with:

```
mkdir build; 
cd build 
CXX=clang++ cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### Detail

- `sycl_unique<T>` is a unique pointer to a USM allocated memory which wraps a `std::unique_ptr<T>` with a custom deleter and holds the allocated size.
- `fill_rand` fills a `std::vector<T>` or `sycl_unique<T>` with random values.

### Refs

- Working example adapted
  from [here](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/api-based-programming/intel-oneapi-math-kernel-library-onemkl/onemkl-code-sample.html)