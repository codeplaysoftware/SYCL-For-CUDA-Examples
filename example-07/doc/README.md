# usm_smart_ptr wrappers

We provide in `include/tools/usm_smart_ptr.hpp` overloads to `std::unique_ptr` and `std::shared_ptr` that manages SYCL's USM memory.
```c++
using namespace usm_smart_ptr;
sbb::queue q;
size_t count;
auto usm_unique_ptr = make_unique_ptr<my_type, alloc::shared>(count, q); // could return a pointer to sbb::malloc_shared<my_type>(count, q);
auto usm_shared_ptr = make_shared_ptr<my_type, alloc::shared>(count, q);
```
You can choose between `alloc::device`, `alloc::shared` and `alloc::host`. Calling `.get()` on the pointers will return a decorated pointer to the underlying memory which allow to keep track of where the memory was allocated.
It's just compile-time type safety. These decorated types allow to further construct the types :
* `device_accessible_ptr<byte>`
* `host_accessible_ptr<byte>`

This prevents the following (potentially incorrect) code from compiling:
```c++
// A function
void fill_on_host(host_accessible_ptr<flloat> ptr, size_t size){
    float* raw_ptr = (float*) ptr;
    //do your thing on the host
}

// later
auto device_memory = make_shared_ptr<flloat, alloc::device>(1024, q);

// attempt to fill the memory
fill_on_host(device_memory.get(), 1024); // Won't compile, we're saved!
```




# API

## Computing hashes
The number of inputs and template arguments to the hashing function depends on the type. Any missing or extra argument passed will result in compilation failure. 
```C++
hash::compute<hash::method::blake2b, n_outbit>(queue, input_ptr, input_block_size, output_hashes, n_blocs, key_ptr, key_size);
hash::compute<hash::method::keccak, n_outbit>(queue, input_ptr, input_block_size, output_hashes, n_blocs);
hash::compute<hash::method::sha3, n_outbit>(queue, input_ptr, input_block_size, output_hashes, n_blocs;
hash::compute<hash::method::sha1>(queue, input_ptr, input_block_size, output_hashes, n_blocs,);
hash::compute<hash::method::sha256>(queue, input_ptr, input_block_size, output_hashes, n_blocs);
hash::compute<hash::method::md2>(queue, input_ptr, input_block_size, output_hashes, n_blocs);
hash::compute<hash::method::md5>(queue, input_ptr, input_block_size, output_hashes, n_blocs);
```

We'll consider the `blake2b` method in the rest. For each method we got two overloads :

### 1. Implicit memory copy
```
hash::compute<hash::method::blake2b, n_outbit>(sbb::queue &q, const byte*, dword, byte*, dword, byte *, dword);
```
This is the overload you would call if you got C++ allocated pointers to your memory (array on the stack, malloc, new[], ...).
When calling this function, the memory will be copied behind the scenes to the device as it's the safets behaviour.


### 2. No memory copy
If you wrap your memory pointers in `device_accessible_ptr<byte>` (see `include/tools/usm_smart_ptr.hpp`), then the library will assume these points to a memory that is accessible by the `sbb::device` you build your `sbb::queue` on.
```
hash::compute<hash::method::blake2b, n_outbit>(sbb::queue &q, const device_accessible_ptr<byte>, dword, device_accessible_ptr<byte>, dword, const byte *, dword);
```
Best, this overload will be called if you use the previously described `usm_smart_ptr wrappers` with `alloc::device` or `alloc::shared`. We voluntarily excluded `alloc::host` as the remote memory accesses could potentially cause performanec issues.

## Hash functions querying
The following `constexpr` function returns the length, in bytes of a hash produced by the method.
```c++
hash::get_block_size<hash::method::keccak, 128>();
```
You can also query the name with:
```c++
hash::get_name<hash::method::keccak, 128>()
```
r


# Kernel work group size formula
The nd_range sizes are computed in `include/determine_kernel_config.hpp`. When running on a CPU we'll try to make twice as much work groups as you've got execution threads on your system as, with OpenCL, each work group seems to be executed on one CPU thread.
When running on the GPU we'll try to make work groups that contains 64 work items each. Going above 64 seems to decrease performance. 
This behaviour might not be optimal and should be customised to fit your SYCL implementation. 
