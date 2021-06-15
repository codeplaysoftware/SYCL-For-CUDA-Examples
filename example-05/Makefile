SYCLCXX=clang++
SYCLFLAGS=-O2 -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda
OBJS=main.o vadd_sycl.o vadd_cuda.o
CUFLAGS=--cuda-gpu-arch=sm_80 -std=c++11 


%.o: %.cpp
	${SYCLCXX} ${SYCLFLAGS} -c -o $@ $<

%.o: %.cu
	${SYCLCXX} ${CUFLAGS} -c -o $@ $<

main.exe: ${OBJS}
	${SYCLCXX} ${SYCLFLAGS} ${CUFLAGS} ${OBJS}  -L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread -o $@

clean: 
	rm -f ${OBJS}

MPICOMP = mpicxx -I$(HOME)/sycl_workspace/build_dpcpp/install/include/sycl/ -I$(HOME)/sycl-blas/include -I$(HOME)/sycl-blas/external/computecpp-sdk/include/ -L$(HOME)/sycl-blas/build -O3 -fsycl-unnamed-lambda -std=c++17  -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -lsycl_blas

distributed-batch-gemm: distributed-batch-gemm.o
	$(MPICOMP) distributed-batch-gemm.o -o distributed-batch-gemm

distributed-batch-gemm.o: distributed-batch-gemm.cpp
	$(MPICOMP) -c distributed-batch-gemm.cpp

run: distributed-batch-gemm
	LD_LIBRARY_PATH=~/sycl_workspace/build_dpcpp/install/lib:$(HOME)/sycl-blas/build mpirun -np 2 --mca pml ucx -mca btl ^uct -x UCX_NET_DEVICES=mlx5_0:1 ./distributed-batch-gemm

.PHONY: clean


