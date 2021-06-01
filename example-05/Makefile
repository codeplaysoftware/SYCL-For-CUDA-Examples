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
