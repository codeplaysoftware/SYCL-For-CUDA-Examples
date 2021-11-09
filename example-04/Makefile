MPICOMP=mpicxx -I$(HOME)/sycl_workspace/build_dpcpp/install/include/sycl/ -O1 -fsycl-unnamed-lambda -std=c++17  -fsycl -fsycl-targets=nvptx64-nvidia-cuda

sycl-mpi-smaple: SYCL-MPI-Sample.o
	$(MPICOMP) SYCL-MPI-Sample.o -o sycl-mpi-sample

SYCL-MPI-Sample.o: SYCL-MPI-Sample.cpp
	$(MPICOMP) -c SYCL-MPI-Sample.cpp

run: sycl-mpi-sample
	mpirun -np 2 ./sycl-mpi-sample

.PHONY: clean

clean:
	rm -f sycl-mpi-sample *.o
