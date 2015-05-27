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
 * Algorithm 1 of general matrix-matrix multiplication (GEMM)
 * if boolean transpose is false,
 * GEMM operation is expressed as D = alpha*A*B + beta*C
 * otherwise, it is expressed as D = alpha*A^T*B + beta*C
 * One thread is used to calculate one element in matrix D
 * natively
 * if transpose is false
 *  m: number of rows of A / number of rows of C/D
 *  n: number of columns of A / number of rows of B
 *  l: number of columns of B / number of columns of C/D
 * if transpose is true
 *  m: number of rows of A^T / number of rows of C/D
 *  n: number of columns of A^T / number of rows of B
 *  l: number of columns of B / number of columns of C/D
 */
void gpu_GEMM_1 (const double alpha,
                 const double beta,
                 const double* const mat_A,
                 const double* const mat_B,
                 const double* const mat_C,
                 double* mat_D,
			     const int m,
			     const int n,
			     const int l,
                 const bool transpose);

/*
 * Algorithm 2 of general matrix-matrix multiplication (GEMM)
 * if boolean transpose is false,
 * GEMM operation is expressed as D = alpha*A*B + beta*C
 * otherwise, it is expressed as D = alpha*A^T*B + beta*C
 * Blocking algorithm and shared memory is used in this algorithm
 * if transpose is false
 *  m: number of rows of A / number of rows of C/D
 *  n: number of columns of A / number of rows of B
 *  l: number of columns of B / number of columns of C/D
 * if transpose is true
 *  m: number of rows of A^T / number of rows of C/D
 *  n: number of columns of A^T / number of rows of B
 *  l: number of columns of B / number of columns of C/D
 */
void gpu_GEMM_2 (const double alpha,
                 const double beta,
                 const double* const mat_A,
                 const double* const mat_B,
                 const double* const mat_C,
                 double* mat_D,
                 const int m,
                 const int n,
                 const int l,
                 const bool transpose);

/*
 * Applies the sigmoid function to each element of the matrix
 * and returns a new matrix by GPU
 *  m: number of rows of the matrix
 *  n: number of columns of the matrix
 */
void cuda_sigmoid (const double* const mat_1,
                   double* mat_2,
                   const int m,
                   const int n);
/*
 * Applies the softmax to each rowvec of the matrix by GPU
 *  m: number of rows of the matrix
 *  n: number of columns of the matrix
 */
void cuda_softmax (const double* const mat_1,
                   double* mat_2,
                   const int m,
                   const int n);

#endif