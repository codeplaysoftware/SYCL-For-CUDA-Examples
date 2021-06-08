MPICOMP = mpicxx -I$(HOME)/sycl_workspace/build_dpcpp/install/include/sycl/ -I$(HOME)/sycl-blas/include -I$(HOME)/sycl-blas/external/computecpp-sdk/include/ -L$(HOME)/sycl-blas/build -O3 -fsycl-unnamed-lambda -std=c++17  -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -lsycl_blas

distributed-batch-gemm: distributed-batch-gemm.o
	$(MPICOMP) distributed-batch-gemm.o -o distributed-batch-gemm

distributed-batch-gemm.o: distributed-batch-gemm.cpp
	$(MPICOMP) -c distributed-batch-gemm.cpp

run: distributed-batch-gemm
	LD_LIBRARY_PATH=~/sycl_workspace/build_dpcpp/install/lib:$(HOME)/sycl-blas/build mpirun -np 2 --mca pml ucx -mca btl ^uct -x UCX_NET_DEVICES=mlx5_0:1 ./distributed-batch-gemm

.PHONY: clean

clean:
	rm -f distributed-batch-gemm *.o
