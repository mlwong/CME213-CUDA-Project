#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

struct event_pair
{
  cudaEvent_t start;
  cudaEvent_t end;
};

inline void check_launch(const char * kernel_name)
{
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    std::cerr << "error in " << kernel_name << " kernel" << std::endl;
    std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

inline void start_timer(event_pair * p)
{
  cudaEventCreate(&p->start);
  cudaEventCreate(&p->end);
  cudaEventRecord(p->start, 0);
}


inline double stop_timer(event_pair * p)
{
  cudaEventRecord(p->end, 0);
  cudaEventSynchronize(p->end);
  
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, p->start, p->end);
  cudaEventDestroy(p->start);
  cudaEventDestroy(p->end);
  return elapsed_time;
}

int useless_gpu_add_one (int t);

/*
 * Algorithm 1 to use one thread to calculate each element in matrix D
 * natively
 */
void gpu_GEMM_1(const double alpha,
                const double beta,
                const double* const mat_A,
                const double* const mat_B,
                const double* const mat_C,
                double* mat_D,
			    int m,
			    int n,
			    int l);

/*
 * Algorithm 2 to use a blocking algorithm and shared memory
 */
void gpu_GEMM_2(const double alpha,
                const double beta,
                const double* const mat_A,
                const double* const mat_B,
                const double* const mat_C,
                double* mat_D,
                int m,
                int n,
                int l);

#endif