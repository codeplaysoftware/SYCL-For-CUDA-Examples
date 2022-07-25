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
# *    Makefile for MPI example
# **************************************************************************/
MPICOMP=mpicxx -I${SYCL_ROOT_DIR}/include/sycl/ -O1 -fsycl-unnamed-lambda -std=c++17  -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wno-linker-warnings

sycl-mpi-sample: SYCL-MPI-Sample.o
	$(MPICOMP) SYCL-MPI-Sample.o -o sycl-mpi-sample

SYCL-MPI-Sample.o: SYCL-MPI-Sample.cpp
	$(MPICOMP) -c SYCL-MPI-Sample.cpp

run: sycl-mpi-sample
	mpirun -np 2 ./sycl-mpi-sample

.PHONY: clean

clean:
	rm -f sycl-mpi-sample *.o
