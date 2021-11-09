CXX=clang++
FORT=nvfortran
FFLAGS=-c++libs -cuda  
CXXFLAGS=-fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-unnamed-lambda
DPCPP_PATH=/home/ruyman/sycl_workspace/build_dpcpp/install

default: final.exe

saxpy_sycl.so: saxpy.cpp
	$(CXX) $(CXXFLAGS) -fPIC --shared saxpy.cpp -o saxpy_sycl.so

saxpy_cuf.o: saxpy.cuf
	$(FORT) $(FFLAGS) -c saxpy.cuf -o saxpy_cuf.o
       
final.exe: saxpy_cuf.o saxpy_sycl.so
	$(FORT) $(FFLAGS) -o final.exe saxpy_cuf.o saxpy_sycl.so -L${DPCPP_PATH}/lib/ -lsycl 

.PHONY: clean

clean: 
	rm -f saxpy_cuf.o saxpy_sycl.so final.exe mathops.mod

