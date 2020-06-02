

CUDACXX=${SYCL_ROOT}/bin/clang++

SYCL_INCLUDE=${SYCL_ROOT}/include/sycl/

CUDAFLAGS=--cuda-gpu-arch=sm_30 

CXXFLAGS=-std=c++17 ${CUDAFLAGS} -I${SYCL_INCLUDE} -g

CUDA_ROOT=/usr/local/cuda/

LIBS=-L${SYCL_ROOT}/include/lib -lOpenCL -lsycl -L${CUDA_ROOT}/lib64 -lcudart

default: vec_add.exe usm_vec_add.exe

vec_add.exe: vec_add.cu
	${CUDACXX} ${CXXFLAGS} $< ${LIBS} -o $@

usm_vec_add.exe: vec_add_usm.cu
	${CUDACXX} ${CXXFLAGS} $< ${LIBS} -o $@


clean:
	rm vec_add.exe usm_vec_add.exe
