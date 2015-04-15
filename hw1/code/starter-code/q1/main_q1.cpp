#include <iostream>
#include <random>
#include <vector>

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <sys/time.h>

#include "tests.h"

typedef unsigned int uint; 
const uint kMaxInt = 100;
const uint kSize = 30000000;

std::vector<uint> serialSum(const std::vector<uint> &v) {
  std::vector<uint> sums(2);
  // TODO
  
  sums[0] = 0;
  sums[1] = 0;
  
  for (uint i = 0; i< v.size(); i++)
  {
    if (v[i] % 2 == 0)
    {
      sums[0] += v[i];
    }
    else
    {
      sums[1] += v[i];
    }
  }
  
  return sums;
}

std::vector<uint> parallelSum(const std::vector<uint> &v) {
  std::vector<uint> sums(2);
  // TODO
  
  uint sum_even = 0;
  uint sum_odd  = 0;
  
#pragma omp parallel for reduction(+:sum_odd) reduction(+:sum_even)
  for (uint i = 0; i< v.size(); i++)
  {
    if (v[i] % 2 == 0)
    {
      sum_even += v[i];
    }
    else
    {
      sum_odd += v[i];
    }
  }

  sums[0] = sum_even;
  sums[1] = sum_odd;
  
  return sums;
}

std::vector<uint> initializeRandomly(const uint size, const uint max_int) { 
  std::vector<uint> res(size);
  std::default_random_engine generator;
  std::uniform_int_distribution<uint> distribution(0, max_int);
  for (uint i = 0; i < size; ++i)
    res[i] = distribution(generator);
  return res;
}

int main(){
   
  // You can uncomment the line below to make your own simple tests
  // std::vector<uint> v = ReadVectorFromFile("vec");
  std::vector<uint> v = initializeRandomly(kSize, kMaxInt);

  std::cout << "Parallel" << std::endl;
  struct timeval start, end;
  gettimeofday(&start, NULL);
  std::vector<uint> sums = parallelSum(v);
  std::cout << "Sum Even: " << sums[0] << std::endl;
  std::cout << "Sum Odd: " << sums[1] << std::endl;
  gettimeofday(&end, NULL);
  double delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec -
                  start.tv_usec) / 1.e6;
  std::cout << "Time: " << delta << std::endl;

  std::cout << "Serial" << std::endl;
  gettimeofday(&start, NULL);
  std::vector<uint> sumsSer = serialSum(v);
  std::cout << "Sum Even: " << sumsSer[0] << std::endl;
  std::cout << "Sum Odd: " << sumsSer[1] << std::endl;
  gettimeofday(&end, NULL);
  delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec -
           start.tv_usec) / 1.e6;
  std::cout << "Time: " << delta << std::endl;

  return 0;
}

