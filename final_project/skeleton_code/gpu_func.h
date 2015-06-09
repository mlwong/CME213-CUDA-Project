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
 * Algorithm 0 of general matrix-matrix multiplication (GEMM)
 * GEMM operation is expressed as D = alpha*op(A)*op(B) + beta*C
 * One thread is used to calculate one element in matrix D natively
 * 1D blocks are used
 * natively
 * 
 * Parameters:
 *  m:              Number of rows of op(A) / number of rows of C/D
 *  n:              Number of columns of op(A) / number of rows of op(B)
 *  l:              Number of columns of op(B) / number of columns of C/D
 *  transpose_A:    Whether A should be transposed
 *                  If transpose_A is false, op(A) = A
 *                  Otherwise, op(A) = A^T
 *  transpose_B:    Whether B should be transposed
 *                  If transpose_B is false, op(B) = B
 *                  Otherwise, op(B) = B^T
 */
void gpu_GEMM_0 (const double alpha,
                 const double beta,
                 const double* const mat_A,
                 const double* const mat_B,
                 const double* const mat_C,
                 double* mat_D,
			     const int m,
			     const int n,
			     const int l,
				 const bool transpose_A,
				 const bool transpose_B);

/*
 * Algorithm 1 of general matrix-matrix multiplication (GEMM)
 * GEMM operation is expressed as D = alpha*op(A)*op(B) + beta*C
 * One thread is used to calculate one element in matrix D
 * natively. 2D blocks are used
 * 
 * Parameters:
 *  m:              Number of rows of op(A) / number of rows of C/D
 *  n:              Number of columns of op(A) / number of rows of op(B)
 *  l:              Number of columns of op(B) / number of columns of C/D
 *  transpose_A:    Whether A should be transposed
 *                  If transpose_A is false, op(A) = A
 *                  Otherwise, op(A) = A^T
 *  transpose_B:    Whether B should be transposed
 *                  If transpose_B is false, op(B) = B
 *                  Otherwise, op(B) = B^T
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
                 const bool transpose_A,
                 const bool transpose_B);

/*
 * Algorithm 2 of general matrix-matrix multiplication (GEMM)
 * GEMM operation is expressed as D = alpha*op(A)*op(B) + beta*C
 * Blocking algorithm and shared memory is used in this algorithm
 * 
 * Parameters:
 *  m:              Number of rows of op(A) / number of rows of C/D
 *  n:              Number of columns of op(A) / number of rows of op(B)
 *  l:              Number of columns of op(B) / number of columns of C/D
 *  transpose_A:    Whether A should be transposed
 *                  If transpose_A is false, op(A) = A
 *                  Otherwise, op(A) = A^T
 *  transpose_B:    Whether B should be transposed
 *                  If transpose_B is false, op(B) = B
 *                  Otherwise, op(B) = B^T
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
                 const bool transpose_A,
                 const bool transpose_B);

/*
 * Algorithm 3 of general matrix-matrix multiplication (GEMM)
 * GEMM operation is expressed as D = alpha*op(A)*op(B) + beta*C
 * A better blocking algorithm and shared memory is used in this algorithm
 * 
 * Parameters:
 *  m:              Number of rows of op(A) / number of rows of C/D
 *  n:              Number of columns of op(A) / number of rows of op(B)
 *  l:              Number of columns of op(B) / number of columns of C/D
 *  transpose_A:    Whether A should be transposed
 *                  If transpose_A is false, op(A) = A
 *                  Otherwise, op(A) = A^T
 *  transpose_B:    Whether B should be transposed
 *                  If transpose_B is false, op(B) = B
 *                  Otherwise, op(B) = B^T
 */
void gpu_GEMM_3 (const double alpha,
                 const double beta,
                 const double* const mat_A,
                 const double* const mat_B,
                 const double* const mat_C,
                 double* mat_D,
			     const int m,
			     const int n,
			     const int l,
				 const bool transpose_A,
				 const bool transpose_B);

/*
 * Applies the sigmoid function to each element of the matrix
 * and returns a new matrix by GPU
 *  m: number of rows of the matrix
 *  n: number of columns of the matrix
 */
void gpu_sigmoid (const double* const mat_1,
                  double* mat_2,
                  const int m,
                  const int n);
/*
 * Applies the softmax function to each row vector of the matrix
 * and returns a new matrix by GPU
 *  m: number of rows of the matrix
 *  n: number of columns of the matrix
 */
void gpu_softmax (const double* const mat_1,
                  double* mat_2,
                  const int m,
                  const int n);

/*
 * Sum elements of matrix in each column
 *  m: number of rows of the matrix
 *  n: number of columns of the matrix
 */
void gpu_sum_col (const double* const mat,
                  double* row_vec,
                  const int m,
                  const int n);

/*
 * Elementwise multiplication used to compute dW1
 * and return a new matrix by GPU
 *  mat_da1: input matrix da1
 *  mat_a1:  input matrix a1
 *  mat_dz1: output matrix dz1
 *  m:     number of rows of the matrices
 *  n:     number of columns of the matrices
 */
void gpu_elementwise_mult (const double* const mat_da1,
						   const double* const mat_a1,
						   double* mat_dz1,
						   const int m,
						   const int n);

/*
 * Elementwise substraction between two matrices by GPU
 * C = A - alpha * B
 * A new matrix is returned
 *  mat_A: input matrix A
 *  mat_B: input matrix B
 *  mat_C: output matrix C
 *  m:     number of rows of the matrices
 *  n:     number of columns of the matrices
 */
void gpu_elementwise_subtract (const double alpha,
							   const double* const mat_A,
							   const double* const mat_B,
							   double* mat_C,
							   const int m,
							   const int n);

/*
 * Compute the diff matrix
 * and return a new matrix by GPU
 *  mat_yc:   input matrix yc
 *  mat_y:    input matrix y
 *  mat_diff: output matrix diff
 *  m:     number of rows of the matrices
 *  n:     number of columns of the matrices
 */
void gpu_compute_diff (const double* const mat_yc,
					   const double* const mat_y,
					   double* mat_diff,
					   const int m,
					   const int n);

/*
 * Transpose a matrix and return a new matrix by GPU
 *  mat_1: input matrix 
 *  mat_2: output matrix
 *  m:     number of rows of the input matrix
 *  n:     number of columns of the input matrix
 */
void gpu_transpose (const double* const mat_1,
					double* mat_2,
					const int m,
					const int n);

/*
 * Do the feedforward in GPU entirely
 */
void gpu_accel_feedforward (const double* const mat_X, int X_n_rows, int X_n_cols,
                            const double* const mat_W1, int W1_n_rows, int W1_n_cols,
                            const double* const mat_b1, int b1_n_rows, int b1_n_cols,
                            double* mat_z1, int z1_n_rows, int z1_n_cols,
                            double* mat_a1, int a1_n_rows, int a1_n_cols,
                            const double* const mat_W2, int W2_n_rows, int W2_n_cols,
                            const double* const mat_b2, int b2_n_rows, int b2_n_cols,
                            double* mat_z2, int z2_n_rows, int z2_n_cols,
                            double* mat_a2, int a2_n_rows, int a2_n_cols);

/*
 * Do the backpropagation in GPU entirely
 */
void gpu_accel_backprop (const double reg,
						 const double* const mat_diff, const int diff_n_rows, const int diff_n_cols,
                         const double* const mat_X, const int X_n_rows, const int X_n_cols,
                         const double* const mat_a1, const int a1_n_rows, const int a1_n_cols,
                         const double* const mat_W1, const int W1_n_rows, const int W1_n_cols,
                         const double* const mat_W2, const int W2_n_rows, const int W2_n_cols,
                         double* mat_dW1, const int dW1_n_rows, const int dW1_n_cols,
                         double* mat_dW2, const int dW2_n_rows, const int dW2_n_cols,
                         double* mat_db1, const int db1_n_cols,
                         double* mat_db2, const int db2_n_cols);

/*
 * Do the feedforward and backpropagation in GPU entirely
 * Since this function combines both the feedforward and
 * backpropagation algorithm, some communication cost over
 * the the PCI express is saved such as the cost of transfering
 * the data of sub-matrix X, matrices W1 and W2
 * the second GEMM algorithm is used
 */
void gpu_accel_feedforward_backprop_1 (const double reg,
                                       const double* const mat_X, int X_n_rows, int X_n_cols,
                                       const double* const mat_y, int y_n_rows, int y_n_cols,
                                       const double* const mat_W1, int W1_n_rows, int W1_n_cols,
                                       const double* const mat_b1, int b1_n_rows, int b1_n_cols,
                                       double* mat_z1, int z1_n_rows, int z1_n_cols,
                                       double* mat_a1, int a1_n_rows, int a1_n_cols,
                                       const double* const mat_W2, int W2_n_rows, int W2_n_cols,
                                       const double* const mat_b2, int b2_n_rows, int b2_n_cols,
                                       double* mat_z2, int z2_n_rows, int z2_n_cols,
                                       double* mat_a2, int a2_n_rows, int a2_n_cols,
                                       double* mat_dW1, const int dW1_n_rows, const int dW1_n_cols,
                                       double* mat_dW2, const int dW2_n_rows, const int dW2_n_cols,
                                       double* mat_db1, const int db1_n_cols,
                                       double* mat_db2, const int db2_n_cols);

/*
 * Do the feedforward and backpropagation in GPU entirely
 * Compared to gpu_accel_feedforward_backprop_1, this function
 * further minimizes the communication cost. Redundant communication
 * such as transferring back data of z1, a1, z2 from GPU
 * Also, the third GEMM algorithm, which is faster, is used
 */
void gpu_accel_feedforward_backprop_2 (const double reg,
                                       double* mat_X, int X_n_rows, int X_n_cols,
                                       double* mat_y, int y_n_rows, int y_n_cols,
                                       double* mat_W1, int W1_n_rows, int W1_n_cols,
                                       double* mat_b1, int b1_n_rows, int b1_n_cols,
                                       double* mat_W2, int W2_n_rows, int W2_n_cols,
                                       double* mat_b2, int b2_n_rows, int b2_n_cols,
                                       double* mat_a2, int a2_n_rows, int a2_n_cols,
                                       double* mat_dW1, const int dW1_n_rows, const int dW1_n_cols,
                                       double* mat_dW2, const int dW2_n_rows, const int dW2_n_cols,
                                       double* mat_db1, const int db1_n_cols,
                                       double* mat_db2, const int db2_n_cols);

#endif
