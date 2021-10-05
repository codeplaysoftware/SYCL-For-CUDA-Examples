#!/bin/sh
export DPCPP_HOME=~/sycl_workspace
export CUDA_ROOT=/usr/local/cuda-10.2
export LD_LIBRARY_PATH=$DPCPP_HOME/deploy/lib/:$DPCPP_HOME/deploy/lib64/:$DPCPP_HOME/lapack/install/lib64/:$DPCPP_HOME/OpenCL-ICD-Loader/install/lib64:$CUDA_ROOT/lib:$CUDA_ROOT/lib64:$LD_LIBRARY_PATH
export PATH=$DPCPP_HOME/deploy/bin/:$CUDA_ROOT/bin:$PATH

mkdir -p $DPCPP_HOME
cd $DPCPP_HOME
mkdir -p deploy

#export LD_PRELOAD=/opt/intel/opencl/libOpenCL.so.1

run_test=false
cmake_test="OFF"

if [[ -z "$DPCPP_TESTS" ]]; then
  echo "Not testing"
else
  echo "testing"
  run_test=true
  cmake_test="ON"
fi

export CXXFLAGS="${CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=1"

# OpenCL headers+ICD
cd $DPCPP_HOME
(if cd OpenCL-Headers; then git pull; else git clone https://github.com/KhronosGroup/OpenCL-Headers.git; fi)
(if cd OpenCL-ICD-Loader; then git pull; else git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader.git; fi)
cd OpenCL-ICD-Loader
mkdir -p build
cd build
cmake \
  -DOPENCL_ICD_LOADER_HEADERS_DIR=$DPCPP_HOME/OpenCL-Headers \
  -DCMAKE_INSTALL_PREFIX=$DPCPP_HOME/OpenCL-ICD-Loader/install \
  ..
make install -j $(nproc)

#sycl compiler with cuda
source /opt/intel/opencl/env/compiler_rt_vars.sh
cd $DPCPP_HOME
(if cd llvm; then git pull; else git clone https://github.com/intel/llvm.git -b sycl; fi)
cd llvm
python3 ./buildbot/configure.py \
  --cuda \
  -t release \
  --cmake-opt="-DCMAKE_INSTALL_PREFIX=$DPCPP_HOME/deploy" \
  --cmake-opt="-DCUDA_SDK_ROOT_DIR=$CUDA_ROOT" \
  --cmake-opt="-DLLVM_BINUTILS_INCDIR=/usr/local/include" \
  --cmake-opt="-DLLVM_ENABLE_PROJECTS=clang;sycl;llvm-spirv;libunwind;opencl;libdevice;xpti;xptifw;libclc;lld;lldb;libc;libcxx;libcxxabi;openmp;clang-tools-extra;compiler-rt" \
  --cmake-opt="-DLLVM_BUILD_TESTS=$cmake_test" \
  --cmake-opt="-DCMAKE_CXX_STANDARD=17" \
  --cmake-opt="-DLLVM_ENABLE_LTO=off" \
  --cmake-opt="-DLLVM_ENABLE_LLD=ON" \
  --cmake-opt="-DSYCL_ENABLE_WERROR=OFF" \
  --cmake-opt="-Wno-dev"
cd build
ninja install -j $(nproc)
if $run_test; then
  echo "testing llvm"
  ninja check -j $(nproc)
fi

#Lapack Reference
cd $DPCPP_HOME
(if cd lapack; then git pull; else git clone https://github.com/Reference-LAPACK/lapack.git; fi)
cd lapack/
mkdir -p build
cd build/
cmake \
  -DBUILD_SHARED_LIBS=ON \
  -DCBLAS=ON \
  -DLAPACKE=ON \
  -DBUILD_TESTING=$cmake_test \
  -DCMAKE_INSTALL_PREFIX=$DPCPP_HOME/lapack/install \
  ..
cmake --build . -j $(nproc) --target install
if $run_test; then
  cmake --build . -j $(nproc) --target test
fi

#oneTBB
cd $DPCPP_HOME
(if cd oneTBB; then git pull; else git clone https://github.com/oneapi-src/oneTBB.git; fi)
cd oneTBB
mkdir -p build
cd build
cmake -GNinja \
  -DCMAKE_CXX_COMPILER=$DPCPP_HOME/deploy/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DTBB_STRICT=OFF \
  -DCMAKE_INSTALL_PREFIX=$DPCPP_HOME/deploy/ \
  -DTBB_TEST=$cmake_test \
  ..
ninja install -j $(nproc)
if $run_test; then
  ninja test -j $(nproc)
fi

#oneMKL
cd $DPCPP_HOME
(if cd oneMKL; then git pull; else git clone https://github.com/oneapi-src/oneMKL.git; fi)
cd oneMKL
mkdir -p build
cd build
cmake -GNinja \
  -DCMAKE_CXX_COMPILER=$DPCPP_HOME/deploy/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DTARGET_DOMAINS=blas \
  -DENABLE_MKLGPU_BACKEND=OFF \
  -DENABLE_CURAND_BACKEND=OFF \
  -DENABLE_MKLCPU_BACKEND=OFF \
  -DENABLE_CUBLAS_BACKEND=ON \
  -DENABLE_NETLIB_BACKEND=ON \
  -DREF_BLAS_ROOT=$DPCPP_HOME/lapack/install \
  -DNETLIB_ROOT=$DPCPP_HOME/lapack/install \
  -DOPENCL_INCLUDE_DIR=$DPCPP_HOME/OpenCL-Headers \
  -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_ROOT \
  -DSYCL_LIBRARY=$DPCPP_HOME/deploy/lib/libsycl.so \
  -DCMAKE_INSTALL_PREFIX=$DPCPP_HOME/deploy/ \
  -DBUILD_FUNCTIONAL_TESTS=$cmake_test \
  ..
ninja install -j $(nproc)
if $run_test; then
  ninja test -j $(nproc)
fi

#oneDNN
cd $DPCPP_HOME
(if cd oneDNN; then git pull; else git clone https://github.com/oneapi-src/oneDNN.git; fi)
cd oneDNN
mkdir -p build
cd build
cmake -GNinja \
  -DCMAKE_C_COMPILER=$DPCPP_HOME/deploy/bin/clang \
  -DCMAKE_CXX_COMPILER=$DPCPP_HOME/deploy/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DDNNL_INSTALL_MODE=BUNDLE \
  -DDNNL_CPU_RUNTIME=DPCPP \
  -DDNNL_GPU_RUNTIME=DPCPP \
  -DDNNL_GPU_VENDOR=NVIDIA \
  -DNNL_BUILD_EXAMPLES=OFF \
  -DTBBROOT=$DPCPP_HOME/deploy \
  -DCUDA_SDK_ROOT_DIR=$CUDA_ROOT \
  -DOPENCLROOT=$DPCPP_HOME/OpenCL-ICD-Loader/install \
  -DOpenCL_INCLUDE_DIR=$DPCPP_HOME/OpenCL-Headers \
  -DCUBLAS_LIBRARY=$CUDA_ROOT/lib64/libcublas.so \
  -DCUBLAS_INCLUDE_DIR=$CUDA_ROOT/include \
  -DCMAKE_INSTALL_PREFIX=$DPCPP_HOME/deploy \
  -DDNNL_BUILD_TESTS=$cmake_test \
  ..
ninja install -j $(nproc)
if $run_test; then
  LD_LIBRARY_PATH=$DPCPP_HOME/deploy/lib ninja test -j $(nproc)
fi
