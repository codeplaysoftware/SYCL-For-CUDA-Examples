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
 *  vadd_cuda.cu
 *
 *  Description:
 *    Vector addition in CUDA
 **************************************************************************/
#include <array>

// CUDA kernel. Each thread takes care of one element of c
template<class T>
__global__ void vecAdd(T *a, T *b, T *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}
 
template <typename T, size_t N>
void simple_vadd_cuda(const std::array<T, N>& VA, const std::array<T, N>& VB,
                 std::array<T, N>& VC) {
    // Device input vectors
    T *d_a;
    T *d_b;
    //Device output vector
    T *d_c;
 
    // Size, in bytes, of each vector
    const size_t bytes = N*sizeof(T);
 
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
 
    // Copy host vectors to device
    cudaMemcpy( d_a, VA.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, VB.data(), bytes, cudaMemcpyHostToDevice);
 
    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 1024;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil((float)N/blockSize);
 
    // Execute the kernel
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
 
    // Copy array back to host
    cudaMemcpy( VC.data(), d_c, bytes, cudaMemcpyDeviceToHost );
 
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}


template void simple_vadd_cuda<float, 4>(const std::array<float, 4>& VA, const std::array<float, 4>& VB,
                 std::array<float, 4>& VC);
template void simple_vadd_cuda<int, 4>(const std::array<int, 4>& VA, const std::array<int, 4>& VB,
                 std::array<int, 4>& VC);

