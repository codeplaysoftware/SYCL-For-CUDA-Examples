// Copyright (c) 2022 Tom Papatheodore

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>

// Macro for checking errors in GPU API calls
#define gpuErrorCheck(call)                                                                  \
do{                                                                                          \
    cudaError_t gpuErr = call;                                                               \
    if(cudaSuccess != gpuErr){                                                               \
        printf("GPU Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(gpuErr)); \
        exit(1);                                                                             \
    }                                                                                        \
}while(0)

// Size of array
#define N 1048576

// Kernel
__global__ void vector_addition(double *a, double *b, double *c)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < N) c[id] = a[id] + b[id];
}

// Main program
int main()
{
    // Number of bytes to allocate for N doubles
    size_t bytes = N*sizeof(double);

    // Allocate memory for arrays A, B, and C on host
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);

    // Allocate memory for arrays d_A, d_B, and d_C on device
    double *d_A, *d_B, *d_C;
    gpuErrorCheck( cudaMalloc(&d_A, bytes) );	
    gpuErrorCheck( cudaMalloc(&d_B, bytes) );
    gpuErrorCheck( cudaMalloc(&d_C, bytes) );

    // Fill host arrays A, B, and C
    for(int i=0; i<N; i++)
    {
        A[i] = 1.0;
        B[i] = 2.0;
        C[i] = 0.0;
    }

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrorCheck( cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) );
    gpuErrorCheck( cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice) );

    // Set execution configuration parameters
    //      thr_per_blk: number of GPU threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 128;
    int blk_in_grid = ceil( float(N) / thr_per_blk );

    // Launch kernel
    vector_addition<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C);

    // Check for synchronous errors during kernel launch (e.g. invalid execution configuration paramters)
    gpuErrorCheck( cudaGetLastError() );

    // Check for asynchronous errors during GPU execution (after control is returned to CPU)
    gpuErrorCheck( cudaDeviceSynchronize() );

    // Copy data from device array d_C to host array C
    gpuErrorCheck( cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost) );

    // Verify results
    double tolerance = 1.0e-14;
    for(int i=0; i<N; i++)
    {
        if( fabs(C[i] - 3.0) > tolerance )
        { 
            printf("Error: value of C[%d] = %f instead of 3.0\n", i, C[i]);
            exit(1);
        }
    }	

    // Free CPU memory
    free(A);
    free(B);
    free(C);

    // Free GPU memory
    gpuErrorCheck( cudaFree(d_A) );
    gpuErrorCheck( cudaFree(d_B) );
    gpuErrorCheck( cudaFree(d_C) );

    printf("\n---------------------------\n");
    printf("__SUCCESS__\n");
    printf("---------------------------\n");
    printf("N                 = %d\n", N);
    printf("Threads Per Block = %d\n", thr_per_blk);
    printf("Blocks In Grid    = %d\n", blk_in_grid);
    printf("---------------------------\n\n");

    return 0;
}
