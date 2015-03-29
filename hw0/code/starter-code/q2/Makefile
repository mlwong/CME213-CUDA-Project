CXX=g++
CXXFLAGS=-std=c++11 -g -Wall
SRC=main_q2.cpp
HDR= matrix.hpp matrix_lt.hpp

main: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o *~ *~ main
	rm -rf *.dSYM

