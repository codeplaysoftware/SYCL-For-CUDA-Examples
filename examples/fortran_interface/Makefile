#/***************************************************************************
# *
# *  Copyright (C) Codeplay Software Ltd.
# *
# *  Licensed under the Apache License, Version 2.0 (the "License");
# *  you may not use this file except in compliance with the License.
# *  You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# *  Unless required by applicable law or agreed to in writing, software
# *  distributed under the License is distributed on an "AS IS" BASIS,
# *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# *  See the License for the specific language governing permissions and
# *  limitations under the License.
# *
# *  Codeplay's SYCL-For-CUDA-Examples
# *
# *  Makefile
# *
# *  Description:
# *    Makefile for fortran interface
# **************************************************************************/
CXX=clang++
FORT=nvfortran
FFLAGS=-c++libs -cuda  
CXXFLAGS=-fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-unnamed-lambda -Wno-linker-warnings
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

