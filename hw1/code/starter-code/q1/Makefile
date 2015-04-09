CXX=g++
CXXFLAGS=-std=c++11 -g -Wall -O3 -fopenmp
SRC=main_q1.cpp tests.cpp

main: $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@

clean:
	rm -f *.o *~ *~ main
	rm -rf *.dSYM

