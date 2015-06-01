CC=g++
CUD=nvcc
CFLAGS=-std=c++11 -fpermissive -pthread -I./lib/usr/include -I/cm/shared/apps/openmpi/gcc/64/1.8.1/include 
LFLAGS=-larmadillo -lcublas -lcudart -lmpi_cxx -lmpi -pthread -L=/cm/shared/apps/openmpi/gcc/64/1.8.1/lib64 -L=./lib/usr/lib64 -L=/cm/shared/apps/cuda65/toolkit/6.5.14/lib64 -Wl,-rpath=/opt/intel/composer_xe_2013_sp1.0.080/mkl/lib/intel64/,-rpath=./lib/usr/lib64,-rpath=/cm/shared/apps/openmpi/gcc/64/1.8.1/lib64
CUDFLAGS=-O3 -c -arch=sm_20 -Xcompiler -Wall,-Winline,-Wextra,-Wno-strict-aliasing

main: main.o two_layer_net.o mnist.o common.o gpu_func.o
	$(CC) $(LFLAGS) main.o two_layer_net.o mnist.o common.o gpu_func.o -o main

main.o: main.cpp utils/test_utils.h
	$(CC) $(CFLAGS) -c main.cpp

two_layer_net.o: utils/two_layer_net.cpp
	$(CC) $(CFLAGS) -c utils/two_layer_net.cpp

mnist.o: utils/mnist.cpp
	$(CC) $(CFLAGS) -c utils/mnist.cpp

common.o: utils/common.cpp
	$(CC) $(CFLAGS) -c utils/common.cpp

gpu_func.o: gpu_func.cu
	$(CUD) $(CUDFLAGS) -c gpu_func.cu

clean:
	rm -rf *.o main
