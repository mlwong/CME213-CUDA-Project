CXX=g++
CXXFLAGS=-std=c++11 -g -Wall -O3 -fopenmp
SRC=main_q2.cpp tests.cpp

main: $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@
	$(CXX) $(CXXFLAGS) $(SRC) -D QUESTION6 -o $@_q6

clean:
	rm -f *.o *~ *~ main
	rm -rf *.dSYM

