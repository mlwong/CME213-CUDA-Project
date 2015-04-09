#ifndef TESTS_H
#define TESTS_H

#include <vector>
#include <string> 

typedef unsigned int uint;

std::vector<uint> ReadVectorFromFile(const std::string& filename);

uint StringToUint(const std::string& line); 

#endif
