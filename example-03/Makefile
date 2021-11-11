

CUDACXX=${SYCL_ROOT_DIR}/bin/clang++

CUDA_ROOT_DIR=/usr/local/cuda/

CUDAFLAGS=--cuda-gpu-arch=sm_70

CXXFLAGS=-std=c++17 ${CUDAFLAGS} -I${SYCL_ROOT_DIR}/include/ -I${SYCL_ROOT_DIR}/include/sycl/ -Wno-linker-warnings  -g

LIBS=-L${SYCL_ROOT_DIR}/lib -lOpenCL -lsycl -L${CUDA_ROOT_DIR}/lib64 -lcudart

default: vec_add.exe usm_vec_add.exe

vec_add.exe: vec_add.cu
	${CUDACXX} ${CXXFLAGS} $< ${LIBS} -o $@

usm_vec_add.exe: vec_add_usm.cu
	${CUDACXX} ${CXXFLAGS} $< ${LIBS} -o $@


clean:
	rm vec_add.exe usm_vec_add.exe
