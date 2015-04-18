CXX=nvcc
CXXFLAGS=-g --compiler-options -Wall -O3 -arch=sm_20
SRC=main_q2.cu
INC=../common

main: $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -I $(INC) -o $@

clean:
	rm -f *.o *~ *~ main
	rm -rf *.dSYM
