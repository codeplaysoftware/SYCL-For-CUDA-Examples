rm -rf build && mkdir  build && cd build
cmake ../ -DSYCL_ROOT=${SYCL_ROOT_DIR} -DCMAKE_CXX_COMPILER=${SYCL_ROOT_DIR}/bin/clang++ -DCMAKE_EXPORT_COMPILE_COMMANDS=yes
make -j 8
