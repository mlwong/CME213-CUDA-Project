#include "gpu_func.h"

#include <cmath>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>

// Block size used in transpose
#define BLOCK_SIZE_x_transpose 64
#define BLOCK_SIZE_y_transpose 8

// Block size used in algorithm 0 of GEMM
#define BLOCK_SIZE_0 256

// Block size used in algorithm 1 of GEMM
#define BLOCK_SIZE_x_1 32
#define BLOCK_SIZE_y_1 16

// Block size used in algorithm 2 of GEMM
#define BLOCK_SIZE_2 32

// Block size used in algorithm 3 of GEMM
#define BLOCK_SIZE_x_3 16
#define BLOCK_SIZE_y_3 4

// Block size used in sigmoid function
#define BLOCK_SIZE_x_SIGMOID 64
#define BLOCK_SIZE_y_SIGMOID 8

// Block size used in softmax function
#define BLOCK_SIZE_x_SOFTMAX 64
#define BLOCK_SIZE_y_SOFTMAX 8

// Block size used in reduction
#define BLOCK_SIZE_REDUCTION 32

// Block size used in elementwise multiplication
#define BLOCK_SIZE_x_elementwise_mult 64
#define BLOCK_SIZE_y_elementwise_mult 8

// Block size used in elementwise substraction
#define BLOCK_SIZE_x_elementwise_subtract 64
#define BLOCK_SIZE_y_elementwise_subtract 8

// Block size used in difference function
#define BLOCK_SIZE_x_DIFF 64
#define BLOCK_SIZE_y_DIFF 8

__global__
void device_add_one (int* d_result, int t)
{
	*d_result = t + 1;
}

int useless_gpu_add_one (int t)
{
	int result;
	int *d_result;

	checkCudaErrors (cudaMalloc((void **)&d_result, 1 * sizeof (int)));

	event_pair timer;
	start_timer (&timer);
	device_add_one<<<1,1>>>(d_result, t);
	check_launch ("device_add_one");
	double time = stop_timer (&timer);

	std::cout << "device_add_one took: " << time << " seconds" << std::endl;

	checkCudaErrors (cudaMemcpy(&result, d_result, 1 * sizeof (int), cudaMemcpyDeviceToHost));
	return result;
}

// Kernel to compute the very naive GEMM algorithm
__global__
void device_GEMM_0 (const double alpha,
				    const double beta,
				    const double* const d_mat_A,
				    const double* const d_mat_B,
				    const double* const d_mat_C,
				    double* d_mat_D,
				    const int m,
				    const int n,
				    const int l,
					const bool transpose_A,
					const bool transpose_B)
{
	// Global thread index in the grid
	const int tid = threadIdx.x + blockDim.x*blockIdx.x;
	
	// Compute the column and row of the element that thread is
	// computing
	int col = tid%l;
	int row = tid/l;

	// If the thread is not inside the matrix D, return
	if (row >= m || col >= l)
	{
		return;
	}
	
	// Compute the index in matrix D
	int idx = col*m + row;
	
	// sum is used to store the element of op(A)*op(B)
	// that is computed by the thread
	double sum = 0.0;
	
	// Do the multiplication and summation
	if (transpose_A == false)
	{
		if (transpose_B == false)
		{
			for (int k = 0; k < n; k++)
			{
				int idx_A = k*m + row;
				int idx_B = col*n + k;
				
				sum += d_mat_A[idx_A]*d_mat_B[idx_B];
			}
		}
		else
		{
			for (int k = 0; k < n; k++)
			{
				int idx_A = k*m + row;
				int idx_B = k*l + col;
				
				sum += d_mat_A[idx_A]*d_mat_B[idx_B];
			}
		}
	}
	else
	{
		if (transpose_B == false)
		{
			for (int k = 0; k < n; k++)
			{
				int idx_A = row*n + k;
				int idx_B = col*n + k;
				
				sum += d_mat_A[idx_A]*d_mat_B[idx_B];
			}
		}
		else
		{
			for (int k = 0; k < n; k++)
			{
				int idx_A = row*n + k;
				int idx_B = k*l + col;
				
				sum += d_mat_A[idx_A]*d_mat_B[idx_B];
			}
		}
	}
	
	// Each thread writes one element of matrix D
	if (beta == 0.0)
	{
		d_mat_D[idx] = alpha*sum;
	}
	else
	{
		d_mat_D[idx] = alpha*sum + beta*d_mat_C[idx];
	}
}

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
				 const bool transpose_B)
{
	double *d_mat_A;
	double *d_mat_B;
	double *d_mat_C;
	double *d_mat_D;
	
	// Allocate the device memory
	checkCudaErrors(cudaMalloc(&d_mat_A, m*n*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_B, n*l*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_C, m*l*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_D, m*l*sizeof(double)));
	
	// Copy data from the host memory to the device memory
	checkCudaErrors(cudaMemcpy(d_mat_A, mat_A, m*n*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_B, mat_B, n*l*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_C, mat_C, m*l*sizeof(double), cudaMemcpyHostToDevice));

    const int num_blocks = (m*l + BLOCK_SIZE_0 - 1)/BLOCK_SIZE_0;

	// Launch the kernel
	device_GEMM_0 <<<num_blocks, BLOCK_SIZE_0>>> (alpha,
											      beta,
											      d_mat_A,
											      d_mat_B,
											      d_mat_C,
											      d_mat_D,
											      m,
											      n,
											      l,
											      transpose_A,
											      transpose_B);
	
	// Copy data from the device memory to the host memory
	checkCudaErrors(cudaMemcpy(mat_D, d_mat_D, m*l*sizeof(double), cudaMemcpyDeviceToHost));
	
	// Free the device memory
	cudaFree(d_mat_A);
	cudaFree(d_mat_B);
	cudaFree(d_mat_C);
	cudaFree(d_mat_D);
}

// Kernel to compute the first GEMM algorithm
__global__
void device_GEMM_1 (const double alpha,
				    const double beta,
				    const double* const d_mat_A,
				    const double* const d_mat_B,
				    const double* const d_mat_C,
				    double* d_mat_D,
				    const int m,
				    const int n,
				    const int l,
					const bool transpose_A,
					const bool transpose_B)
{
	// Global thread index in the grid
	const int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
	const int tid_y = threadIdx.y + blockDim.y*blockIdx.y;
	
	// If the thread is not inside the matrix D, return
	if (tid_x >= m || tid_y >= l)
	{
		return;
	}
	
	// Compute the index in matrix D
	int idx = tid_y*m + tid_x;
	
	// sum is used to store the element of op(A)*op(B)
	// that is computed by the thread
	double sum = 0.0;
	
	// Do the multiplication and summation
	if (transpose_A == false)
	{
		if (transpose_B == false)
		{
			for (int k = 0; k < n; k++)
			{
				int idx_A = k*m + tid_x;
				int idx_B = tid_y*n + k;
				
				sum += d_mat_A[idx_A]*d_mat_B[idx_B];
			}
		}
		else
		{
			for (int k = 0; k < n; k++)
			{
				int idx_A = k*m + tid_x;
				int idx_B = k*l + tid_y;
				
				sum += d_mat_A[idx_A]*d_mat_B[idx_B];
			}
		}
	}
	else
	{
		if (transpose_B == false)
		{
			for (int k = 0; k < n; k++)
			{
				int idx_A = tid_x*n + k;
				int idx_B = tid_y*n + k;
				
				sum += d_mat_A[idx_A]*d_mat_B[idx_B];
			}
		}
		else
		{
			for (int k = 0; k < n; k++)
			{
				int idx_A = tid_x*n + k;
				int idx_B = k*l + tid_y;
				
				sum += d_mat_A[idx_A]*d_mat_B[idx_B];
			}
		}
	}
	
	// Each thread writes one element of matrix D
	if (beta == 0.0)
	{
		d_mat_D[idx] = alpha*sum;
	}
	else
	{
		d_mat_D[idx] = alpha*sum + beta*d_mat_C[idx];
	}
}

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
				 const bool transpose_B)
{
	double *d_mat_A;
	double *d_mat_B;
	double *d_mat_C;
	double *d_mat_D;
	
	// Allocate the device memory
	checkCudaErrors(cudaMalloc(&d_mat_A, m*n*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_B, n*l*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_C, m*l*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_D, m*l*sizeof(double)));
	
	// Copy data from the host memory to the device memory
	checkCudaErrors(cudaMemcpy(d_mat_A, mat_A, m*n*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_B, mat_B, n*l*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_C, mat_C, m*l*sizeof(double), cudaMemcpyHostToDevice));
	
	dim3 n_threads(0, 0);
	dim3 n_blocks(0, 0);
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_1;
	n_threads.y = BLOCK_SIZE_y_1;
	
	// Compute the grid size
	n_blocks.x = (m + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (l + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel
	device_GEMM_1 <<<n_blocks, n_threads>>> (alpha,
											 beta,
											 d_mat_A,
											 d_mat_B,
											 d_mat_C,
											 d_mat_D,
											 m,
											 n,
											 l,
											 transpose_A,
											 transpose_B);
	
	// Copy data from the device memory to the host memory
	checkCudaErrors(cudaMemcpy(mat_D, d_mat_D, m*l*sizeof(double), cudaMemcpyDeviceToHost));
	
	// Free the device memory
	cudaFree(d_mat_A);
	cudaFree(d_mat_B);
	cudaFree(d_mat_C);
	cudaFree(d_mat_D);
}

// Kernel to compute the second GEMM algorithm
template <int block_size>
__global__
void device_GEMM_2 (const double alpha,
				    const double beta,
				    const double* const d_mat_A,
				    const double* const d_mat_B,
				    const double* const d_mat_C,
				    double* d_mat_D,
				    const int m,
				    const int n,
				    const int l,
					const bool transpose_A,
					const bool transpose_B)
{
	// Global thread index in the grid
	const int g_tid_x = threadIdx.x + blockDim.x*blockIdx.x;
	const int g_tid_y = threadIdx.y + blockDim.y*blockIdx.y;
	
	// Local thread index
	const int tid_x = threadIdx.x;
	const int tid_y = threadIdx.y;
	
	// Block index
	const int bid_x = blockIdx.x;
	const int bid_y = blockIdx.y;
	
	// Declaration of the shared memory used to store the
	// sub-matrix of A/A^T
	__shared__ double mat_A_shared[block_size+1][block_size+1];
	
	// Declaration of the shared memory used to store the
	// sub-matrix of B/B^T
	__shared__ double mat_B_shared[block_size+1][block_size+1];
	
	if (transpose_A == false)
	{
		if (transpose_B == false)
		{
			// Index of the first sub-matrix of A processed by the block
			int mat_A_begin = block_size*bid_x;
			
			// Step size used to iterate through the sub-matrices of A
			int mat_A_step  = block_size*m;
			
			// Index of the first sub-matrix of B processed by the block
			int mat_B_begin = n*block_size*bid_y;
			
			// Index of the last sub-matrix of B processed by the blcok
			int mat_B_end   = mat_B_begin + n - 1;
			
			// Step size used to iterate through the sub-matrices of B
			int mat_B_step  = block_size;
			
			// sum is used to store the element of the block sub-matrix
			// that is computed by the thread
			double sum = 0.0;
			
			// Counter to record the current positions of column position
			// of sub-matrix in matrix A and row position of sub-matrix
			// in matrix B
			int idx_A_col = 0;
			int idx_B_row = 0; 
			
			// Loop over all the sub-matrices of A and B
			// required to compute the block sub-matrix
			for (int idx_A = mat_A_begin, idx_B = mat_B_begin;
				 idx_B <= mat_B_end;
				 idx_A += mat_A_step, idx_B += mat_B_step)
			{
				// Load the matrices from device memory to shared memory;
				// Each thread loads one element of each sub-matrix
				
				if (g_tid_x < m && tid_y + idx_A_col < n)
					mat_A_shared[tid_x][tid_y] = d_mat_A[idx_A + m*tid_y + tid_x];
				if (tid_x + idx_B_row < n && g_tid_y < l)
					mat_B_shared[tid_x][tid_y] = d_mat_B[idx_B + n*tid_y + tid_x];
				
				// Synchronize to make sure the matrices are loaded
				__syncthreads();
				
				if (g_tid_x < m && g_tid_y < l)
				{
					int k_bound = min(block_size, n - idx_A_col);
					
					for (int k = 0; k < k_bound; k++)
					{
						sum += mat_A_shared[tid_x][k]*mat_B_shared[k][tid_y];
					}
				}
				
				idx_A_col += block_size;
				idx_B_row += block_size;
				
				// Synchronize to make sure that the preceding computation
				// is done before loading two new sub-matrices of A and B
				// in the next iteration
				__syncthreads();
			}
			
			// Write the block sub-matrix to device memory
			// each thread writes one element
			if (g_tid_x < m && g_tid_y < l)
			{
				int idx_D = m*block_size*bid_y + block_size*bid_x + tid_x + m*tid_y;
				
				if (beta == 0.0)
				{
					d_mat_D[idx_D] = alpha*sum;
				}
				else
				{
					d_mat_D[idx_D] = alpha*sum + beta*d_mat_C[idx_D];
				}
			}
		}
		else
		{
			// Index of the first sub-matrix of A processed by the block
			int mat_A_begin = block_size*bid_x;
			
			// Step size used to iterate through the sub-matrices of A
			int mat_A_step  = block_size*m;
			
			// Index of the first sub-matrix of B^T processed by the block
			int mat_B_t_begin = block_size*bid_y;
			
			// Index of the last sub-matrix of B^T processed by the blcok
			int mat_B_t_end   = mat_B_t_begin + (n - 1)*l;
			
			// Step size used to iterate through the sub-matrices of B^T
			int mat_B_t_step  = block_size*l;
			
			// sum is used to store the element of the block sub-matrix
			// that is computed by the thread
			double sum = 0.0;
			
			// Counter to record the current positions of column position
			// of sub-matrix in matrix A and row position of sub-matrix
			// in matrix B^T
			int idx_A_col = 0;
			int idx_B_t_row = 0; 
			
			// Loop over all the sub-matrices of A and B^T
			// required to compute the block sub-matrix
			for (int idx_A = mat_A_begin, idx_B_t = mat_B_t_begin;
				 idx_B_t <= mat_B_t_end;
				 idx_A += mat_A_step, idx_B_t += mat_B_t_step)
			{				
				// Load the matrices from device memory to shared memory;
				// Each thread loads one element of each sub-matrix
				
				if (g_tid_x < m && tid_y + idx_A_col < n)
					mat_A_shared[tid_x][tid_y] = d_mat_A[idx_A + m*tid_y + tid_x];
				if (tid_x + idx_B_t_row < n && g_tid_y < l)
					mat_B_shared[tid_x][tid_y] = d_mat_B[idx_B_t + l*tid_x + tid_y];
				
				// Synchronize to make sure the matrices are loaded
				__syncthreads();
				
				if (g_tid_x < m && g_tid_y < l)
				{
					int k_bound = min(block_size, n - idx_A_col);
					
					for (int k = 0; k < k_bound; k++)
					{
						sum += mat_A_shared[tid_x][k]*mat_B_shared[k][tid_y];
					}
				}
				
				idx_A_col += block_size;
				idx_B_t_row += block_size;
				
				// Synchronize to make sure that the preceding computation
				// is done before loading two new sub-matrices of A and B^T
				// in the next iteration
				__syncthreads();
			}
			
			// Write the block sub-matrix to device memory
			// each thread writes one element
			if (g_tid_x < m && g_tid_y < l)
			{
				int idx_D = m*block_size*bid_y + block_size*bid_x + tid_x + m*tid_y;
				
				if (beta == 0.0)
				{
					d_mat_D[idx_D] = alpha*sum;
				}
				else
				{
					d_mat_D[idx_D] = alpha*sum + beta*d_mat_C[idx_D];
				}
			}
		}
	}
	else
	{
		if (transpose_B == false)
		{
			// Index of the first sub-matrix of A^T processed by the block
			int mat_A_t_begin = n*block_size*bid_x;
			
			// Step size used to iterate through the sub-matrices of A^T
			int mat_A_t_step  = block_size;
			
			// Index of the first sub-matrix of B processed by the block
			int mat_B_begin = n*block_size*bid_y;
			
			// Index of the last sub-matrix of B processed by the blcok
			int mat_B_end   = mat_B_begin + n - 1;
			
			// Step size used to iterate through the sub-matrices of B
			int mat_B_step  = block_size;
			
			// sum is used to store the element of the block sub-matrix
			// that is computed by the thread
			double sum = 0.0;
			
			// Counter to record the current positions of column position
			// of sub-matrix in matrix A^T and row position of sub-matrix
			// in matrix B
			int idx_A_t_col = 0;
			int idx_B_row = 0; 
			
			// Loop over all the sub-matrices of A^T and B
			// required to compute the block sub-matrix
			for (int idx_A_t = mat_A_t_begin, idx_B = mat_B_begin;
				 idx_B <= mat_B_end;
				 idx_A_t += mat_A_t_step, idx_B += mat_B_step)
			{
				// Load the matrices from device memory to shared memory;
				// Each thread loads one element of each sub-matrix
				
				if (g_tid_x < m && tid_y + idx_A_t_col < n)
					mat_A_shared[tid_x][tid_y] = d_mat_A[idx_A_t + n*tid_x + tid_y];
				if (tid_x + idx_B_row < n && g_tid_y < l)
					mat_B_shared[tid_x][tid_y] = d_mat_B[idx_B + n*tid_y + tid_x];
				
				// Synchronize to make sure the matrices are loaded
				__syncthreads();
				
				if (g_tid_x < m && g_tid_y < l)
				{
					int k_bound = min(block_size, n - idx_A_t_col);
					
					for (int k = 0; k < k_bound; k++)
					{
						sum += mat_A_shared[tid_x][k]*mat_B_shared[k][tid_y];
					}
				}
				
				idx_A_t_col += block_size;
				idx_B_row += block_size;
				
				// Synchronize to make sure that the preceding computation
				// is done before loading two new sub-matrices of A^T and B
				// in the next iteration
				__syncthreads();
			}
			
			// Write the block sub-matrix to device memory
			// each thread writes one element
			if (g_tid_x < m && g_tid_y < l)
			{
				int idx_D = m*block_size*bid_y + block_size*bid_x + tid_x + m*tid_y;
				
				if (beta == 0.0)
				{
					d_mat_D[idx_D] = alpha*sum;
				}
				else
				{
					d_mat_D[idx_D] = alpha*sum + beta*d_mat_C[idx_D];
				}
			}
		}
		else
		{
			// Index of the first sub-matrix of A^T processed by the block
			int mat_A_t_begin = n*block_size*bid_x;
			
			// Step size used to iterate through the sub-matrices of A^T
			int mat_A_t_step  = block_size;
			
			// Index of the first sub-matrix of B^T processed by the block
			int mat_B_t_begin = block_size*bid_y;
			
			// Index of the last sub-matrix of B^T processed by the blcok
			int mat_B_t_end   = mat_B_t_begin + (n - 1)*l;
			
			// Step size used to iterate through the sub-matrices of B^T
			int mat_B_t_step  = block_size*l;
			
			// sum is used to store the element of the block sub-matrix
			// that is computed by the thread
			double sum = 0.0;
			
			// Counter to record the current positions of column position
			// of sub-matrix in matrix A^T and row position of sub-matrix
			// in matrix B^T
			int idx_A_t_col = 0;
			int idx_B_t_row = 0; 
			
			// Loop over all the sub-matrices of A^T and B^T
			// required to compute the block sub-matrix
			for (int idx_A_t = mat_A_t_begin, idx_B_t = mat_B_t_begin;
				 idx_B_t <= mat_B_t_end;
				 idx_A_t += mat_A_t_step, idx_B_t += mat_B_t_step)
			{				
				// Load the matrices from device memory to shared memory;
				// Each thread loads one element of each sub-matrix
				
				if (g_tid_x < m && tid_y + idx_A_t_col < n)
					mat_A_shared[tid_x][tid_y] = d_mat_A[idx_A_t + n*tid_x + tid_y];
				if (tid_x + idx_B_t_row < n && g_tid_y < l)
					mat_B_shared[tid_x][tid_y] = d_mat_B[idx_B_t + l*tid_x + tid_y];
				
				// Synchronize to make sure the matrices are loaded
				__syncthreads();
				
				if (g_tid_x < m && g_tid_y < l)
				{
					int k_bound = min(block_size, n - idx_A_t_col);
					
					for (int k = 0; k < k_bound; k++)
					{
						sum += mat_A_shared[tid_x][k]*mat_B_shared[k][tid_y];
					}
				}
				
				idx_A_t_col += block_size;
				idx_B_t_row += block_size;
				
				// Synchronize to make sure that the preceding computation
				// is done before loading two new sub-matrices of A^T and B^T
				// in the next iteration
				__syncthreads();
			}
			
			// Write the block sub-matrix to device memory
			// each thread writes one element
			if (g_tid_x < m && g_tid_y < l)
			{
				int idx_D = m*block_size*bid_y + block_size*bid_x + tid_x + m*tid_y;
				
				if (beta == 0.0)
				{
					d_mat_D[idx_D] = alpha*sum;
				}
				else
				{
					d_mat_D[idx_D] = alpha*sum + beta*d_mat_C[idx_D];
				}
			}
		}
	}
}

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
				 const bool transpose_B)
{
	double *d_mat_A;
	double *d_mat_B;
	double *d_mat_C;
	double *d_mat_D;
	
	// Allocate the device memory
	checkCudaErrors(cudaMalloc(&d_mat_A, m*n*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_B, n*l*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_C, m*l*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_D, m*l*sizeof(double)));
	
	// Copy data from the host memory to the device memory
	checkCudaErrors(cudaMemcpy(d_mat_A, mat_A, m*n*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_B, mat_B, n*l*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_C, mat_C, m*l*sizeof(double), cudaMemcpyHostToDevice));
	
	dim3 n_threads(0, 0);
	dim3 n_blocks(0, 0);
	
	// Set the size of the sub-block
	n_threads.x = BLOCK_SIZE_2;
	n_threads.y = BLOCK_SIZE_2;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (m + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (l + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel
	device_GEMM_2 <BLOCK_SIZE_2> <<<n_blocks, n_threads>>> (alpha,
														    beta,
														    d_mat_A,
														    d_mat_B,
														    d_mat_C,
														    d_mat_D,
														    m,
														    n,
														    l,
														    transpose_A,
														    transpose_B);
	
	// Copy data from the device memory to the host memory
	checkCudaErrors(cudaMemcpy(mat_D, d_mat_D, m*l*sizeof(double), cudaMemcpyDeviceToHost));
	
	// Free the device memory
	cudaFree(d_mat_A);
	cudaFree(d_mat_B);
	cudaFree(d_mat_C);
	cudaFree(d_mat_D);
}

// Kernel to compute the third GEMM algorithm
template<int block_size_x, int block_size_y>
__global__
void device_GEMM_3 (const double alpha,
				    const double beta,
				    const double* const d_mat_A,
				    const double* const d_mat_B,
				    const double* const d_mat_C,
				    double* d_mat_D,
				    const int m,
				    const int n,
				    const int l,
					const bool transpose_A,
					const bool transpose_B)
{
	// Local thread index
	const int tid_x = threadIdx.x;
	const int tid_y = threadIdx.y;
	
	// Block index
	const int bid_x = blockIdx.x;
	const int bid_y = blockIdx.y;
	
	// Declare the shared memory to store sub-block of matrix B
	__shared__ double mat_B_shared[block_size_x][block_size_x+1];
	
	// Declare a local array in register to store the sums in the
	// elements the thread is responsible for
	double c[block_size_x];
	
#pragma unroll
	for (int i = 0; i< block_size_x; i++)
	{
		c[i] = 0.0;
	}
	
	if (transpose_A == false && transpose_B == false)
	{
		int idx_A = bid_x*block_size_x*block_size_y + tid_x + tid_y*block_size_x;
		int idx_B = tid_x + (bid_y*block_size_x + tid_y)*n;
		
		int idx_B_last = idx_B + n;
		int col_A = 0;
		
		do
		{
#pragma unroll
			for(int i = 0; i < block_size_x; i += block_size_y)
				mat_B_shared[tid_x][tid_y + i] = d_mat_B[idx_B + i*n];
			
			idx_B += block_size_x;
			
			__syncthreads();
			
			int i_bound = min(block_size_x, n - col_A);
			for (int i = 0; i < i_bound; i++, idx_A+=m)
			{
#pragma unroll
				for (int j = 0; j < block_size_x; j++)
				{
					c[j] += d_mat_A[idx_A]*mat_B_shared[i][j];
				}
			}
			
			col_A += block_size_x;
			
			__syncthreads();
		
		}while (idx_B < idx_B_last);
	}
	else if (transpose_A == false && transpose_B == true)
	{
		int idx_A = bid_x*block_size_x*block_size_y + tid_x + tid_y*block_size_x;
		int idx_B = tid_x*l + (bid_y*block_size_x + tid_y);
		
		int idx_B_last = idx_B + n*l;
		int col_A = 0;
		
		do
		{
#pragma unroll
			for(int i = 0; i < block_size_x; i += block_size_y)
				mat_B_shared[tid_x][tid_y + i] = d_mat_B[idx_B + i];
			
			idx_B += block_size_x*l;
			
			__syncthreads();
			
			int i_bound = min(block_size_x, n - col_A);
			for (int i = 0; i < i_bound; i++, idx_A+=m)
			{
#pragma unroll
				for (int j = 0; j < block_size_x; j++)
				{
					c[j] += d_mat_A[idx_A]*mat_B_shared[i][j];
				}
			}
			
			col_A += block_size_x;
			
			__syncthreads();
		
		}while (idx_B < idx_B_last);
	}
	else if (transpose_A == true && transpose_B == false)
	{
		int idx_A = (bid_x*block_size_x*block_size_y + tid_x + tid_y*block_size_x)*n;
		int idx_B = tid_x + (bid_y*block_size_x + tid_y)*n;

		int idx_B_last = idx_B + n;
		int col_A = 0;
		
		do
		{
#pragma unroll
			for(int i = 0; i < block_size_x; i += block_size_y)
				mat_B_shared[tid_x][tid_y + i] = d_mat_B[idx_B + i*n];
			
			idx_B += block_size_x;
			
			__syncthreads();
			
			int i_bound = min(block_size_x, n - col_A);
			for (int i = 0; i < i_bound; i++, idx_A++)
			{
#pragma unroll
				for (int j = 0; j < block_size_x; j++)
				{
					c[j] += d_mat_A[idx_A]*mat_B_shared[i][j];
				}
			}
			
			col_A += block_size_x;
			
			__syncthreads();
		
		}while (idx_B < idx_B_last);
	}
	else
	{
		int idx_A = (bid_x*block_size_x*block_size_y + tid_x + tid_y*block_size_x)*n;
		int idx_B = tid_x*l + (bid_y*block_size_x + tid_y);
		
		int idx_B_last = idx_B + n*l;
		int col_A = 0;
		do
		{
#pragma unroll
			for(int i = 0; i < block_size_x; i += block_size_y)
				mat_B_shared[tid_x][tid_y + i] = d_mat_B[idx_B + i];
			
			idx_B += block_size_x*l;
			
			__syncthreads();
			
			int i_bound = min(block_size_x, n - col_A);
			for (int i = 0; i < i_bound; i++, idx_A++)
			{
#pragma unroll
				for (int j = 0; j < block_size_x; j++)
				{
					c[j] += d_mat_A[idx_A]*mat_B_shared[i][j];
				}
			}
			
			col_A += block_size_x;
			
			__syncthreads();
		
		}while (idx_B < idx_B_last);
	}
	
	if (bid_x*block_size_x*block_size_y + tid_x + tid_y*block_size_x < m)
	{
		int idx_D = bid_x*block_size_x*block_size_y + (tid_x + tid_y*block_size_x) + bid_y*block_size_x*m;
		if (beta == 0.0)
		{
			int i_bound = min(block_size_x, l - bid_y*block_size_x);
			for (int i = 0; i < i_bound; i++, idx_D += m)
			{
				d_mat_D[idx_D] = alpha*c[i];
			}
		
		}
		else
		{
			int i_bound = min(block_size_x, l - bid_y*block_size_x);
			int idx_C = bid_x*block_size_x*block_size_y + (tid_x + tid_y*block_size_x) + bid_y*block_size_x*m;
			for (int i = 0; i < i_bound; i++, idx_C += m, idx_D += m)
			{
				d_mat_D[idx_D] = alpha*c[i] + beta*d_mat_C[idx_C];
			}
		}
	}
}

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
				 const bool transpose_B)
{
	double *d_mat_A;
	double *d_mat_B;
	double *d_mat_C;
	double *d_mat_D;
	
	// Allocate the device memory
	checkCudaErrors(cudaMalloc(&d_mat_A, m*n*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_B, n*l*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_C, m*l*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_D, m*l*sizeof(double)));
	
	// Copy data from the host memory to the device memory
	checkCudaErrors(cudaMemcpy(d_mat_A, mat_A, m*n*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_B, mat_B, n*l*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_C, mat_C, m*l*sizeof(double), cudaMemcpyHostToDevice));
	
	dim3 n_threads(0, 0);
	dim3 n_blocks(0, 0);
	
	// Set the size of each block
	n_threads.x = BLOCK_SIZE_x_3;
	n_threads.y = BLOCK_SIZE_y_3;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (m + n_threads.x*n_threads.y - 1)/(n_threads.x*n_threads.y);
	n_blocks.y = (l + n_threads.x - 1)/n_threads.x;
		
	// Launch the kernel
	device_GEMM_3 <BLOCK_SIZE_x_3, BLOCK_SIZE_y_3> <<<n_blocks, n_threads>>> (alpha,
																		      beta,
																		      d_mat_A,
																		      d_mat_B,
																		      d_mat_C,
																		      d_mat_D,
																		      m,
																		      n,
																		      l,
																		      transpose_A,
																		      transpose_B);
	
	// Copy data from the device memory to the host memory
	checkCudaErrors(cudaMemcpy(mat_D, d_mat_D, m*l*sizeof(double), cudaMemcpyDeviceToHost));
	
	// Free the device memory
	cudaFree(d_mat_A);
	cudaFree(d_mat_B);
	cudaFree(d_mat_C);
	cudaFree(d_mat_D);
}

// Kernel to compute the sigmoid function
__global__
void device_sigmoid (const double* const d_mat_1,
				     double* d_mat_2,
				     const int m,
				     const int n)
{
	// Global thread index in the grid
	const int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
	const int tid_y = threadIdx.y + blockDim.y*blockIdx.y;
	
	// If the thread is not inside the matrix D, return
	if (tid_x >= m || tid_y >= n)
	{
		return;
	}
	
	// Compute the index in the matrix
	int idx = tid_y*m + tid_x;
	
	// Apply the sigmoid function
	d_mat_2[idx] = 1.0 / (1.0 + exp(-d_mat_1[idx]));
}

/*
 * Applies the sigmoid function to each element of the matrix
 * and returns a new matrix by GPU
 *  m: number of rows of the matrix
 *  n: number of columns of the matrix
 */
void gpu_sigmoid (const double* const mat_1,
                  double* mat_2,
                  const int m,
                  const int n)
{
	double *d_mat_1;
	double *d_mat_2;
	
	// Allocate the device memory
	checkCudaErrors(cudaMalloc(&d_mat_1, m*n*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_2, m*n*sizeof(double)));
	
	// Copy data from the host memory to the device memory
	checkCudaErrors(cudaMemcpy(d_mat_1, mat_1, m*n*sizeof(double), cudaMemcpyHostToDevice));
	
	dim3 n_threads(0, 0);
	dim3 n_blocks(0, 0);
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_SIGMOID;
	n_threads.y = BLOCK_SIZE_y_SIGMOID;
	
	// Compute the grid size
	n_blocks.x = (m + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (n + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to apply the sigmoid function
	device_sigmoid <<<n_blocks, n_threads>>> (d_mat_1, d_mat_2, m, n);
	
	// Copy data from the device memory to the host memory
	checkCudaErrors(cudaMemcpy(mat_2, d_mat_2, m*n*sizeof(double), cudaMemcpyDeviceToHost));
	
	// Free the device memory
	cudaFree(d_mat_1);
	cudaFree(d_mat_2);
}

// Kernel to compute the exponential function
__global__
void device_exponent (const double* const d_mat_1,
				      double* d_mat_2,
				      const int m,
				      const int n)
{
	// Global thread index in the grid
	const int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
	const int tid_y = threadIdx.y + blockDim.y*blockIdx.y;
	
	// If the thread is not inside the matrix D, return
	if (tid_x >= m || tid_y >= n)
	{
		return;
	}
	
	// Compute the index in the matrix
	int idx = tid_y*m + tid_x;
	
	// Apply the exponential function
	d_mat_2[idx] = exp(d_mat_1[idx]);
}

__device__
int nextPowerOf2(const int x)
{
	return (1 << (32 - __clz(x - 1)));
}

// Kernel to do reduction along rows
template<int block_size>
__global__
void device_sum_row_block(const double* const d_mat,
						  double* d_col_vec,
						  const int m,
						  const int n)
{	
	// Local thead index and block index
	const int tid_y = threadIdx.y;
	const int bid_x = blockIdx.x;
	
	double my_sum = 0.0;
	for (int i = tid_y; i < n; i += block_size)
	{
		// Compute the index in the matrix
		int idx = i*m + (threadIdx.x + blockDim.x*blockIdx.x);
		my_sum += d_mat[idx];
	}
	
	__shared__ double smem[block_size];
	
	smem[tid_y] = my_sum;
	
	__syncthreads();
	
	//use this for non-power of 2 block_sizes
	for (int shift = nextPowerOf2(block_size) / 2;
		 shift > 0;
		 shift >>= 1)
	{
		if (tid_y < shift && tid_y + shift < block_size)
		{
			smem[tid_y] += smem[tid_y + shift];
		}
		__syncthreads();
	}
	
	if (tid_y == 0)
		d_col_vec[bid_x] = smem[tid_y];
}

// Kernel to normalize the outpus after applying exponential function
__global__
void device_normalize (double* d_mat,
					   const double* d_col_vec,
					   const int m,
					   const int n)
{
	// Global thread index in the grid
	const int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
	const int tid_y = threadIdx.y + blockDim.y*blockIdx.y;
	
	// If the thread is not inside the matrix D, return
	if (tid_x >= m || tid_y >= n)
	{
		return;
	}
	
	// Compute the index in the matrix
	int idx = tid_y*m + tid_x;
	
	// Apply the exponential function
	d_mat[idx] = d_mat[idx]/d_col_vec[tid_x];
}

/*
 * Applies the softmax to each rowvec of the matrix by GPU
 *  m: number of rows of the matrix
 *  n: number of columns of the matrix
 */
void gpu_softmax (const double* const mat_1,
                  double* mat_2,
                  const int m,
                  const int n)
{
	double *d_mat_1;
	double *d_mat_2;
	double *d_col_vec;

	// Allocate the device memory
	checkCudaErrors(cudaMalloc(&d_mat_1, m*n*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_2, m*n*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_col_vec, m*sizeof(double)));
	
	// Copy data from the host memory to the device memory
	checkCudaErrors(cudaMemcpy(d_mat_1, mat_1, m*n*sizeof(double), cudaMemcpyHostToDevice));
	
	dim3 n_threads(0, 0);
	dim3 n_blocks(0, 0);
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_SOFTMAX;
	n_threads.y = BLOCK_SIZE_y_SOFTMAX;
	
	// Compute the grid size
	n_blocks.x = (m + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (n + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to compute the elementwise exponentinal function
	device_exponent <<<n_blocks, n_threads>>> (d_mat_1, d_mat_2, m, n);
	
	// Compute the block dimension
	n_threads.x = 1;
	n_threads.y = BLOCK_SIZE_REDUCTION;
	
	// Compute the grid size
	n_blocks.x = m;
	n_blocks.y = 1;
	
	device_sum_row_block <BLOCK_SIZE_REDUCTION> <<<n_blocks, n_threads>>> (d_mat_2, d_col_vec, m, n);
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_SOFTMAX;
	n_threads.y = BLOCK_SIZE_y_SOFTMAX;
	
	// Compute the grid size
	n_blocks.x = (m + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (n + n_threads.y - 1)/n_threads.y;
	
	device_normalize <<<n_blocks, n_threads>>> (d_mat_2, d_col_vec, m, n);
	
	// Copy data from the device memory to the host memory
	checkCudaErrors(cudaMemcpy(mat_2, d_mat_2, m*n*sizeof(double), cudaMemcpyDeviceToHost));

	// Free the device memory
	cudaFree(d_mat_1);
	cudaFree(d_mat_2);
	cudaFree(d_col_vec);
}

// Kernel to do reduction along columns
template<int block_size>
__global__
void device_sum_col_block(const double* const d_mat,
						  double* d_row_vec,
						  const int m,
						  const int n)
{
	// Local thead index and block index
	const int tid_x = threadIdx.x;
	const int bid_y = blockIdx.y;
	
	double my_sum = 0.0;
	for (int i = tid_x; i < m; i += block_size)
	{
		// Compute the index in the matrix
		int idx = (threadIdx.y + blockDim.y*blockIdx.y)*m + i;
		my_sum += d_mat[idx];
	}
	
	__shared__ double smem[block_size];
	
	smem[tid_x] = my_sum;

	__syncthreads();
	
	//use this for non-power of 2 block_sizes
	for (int shift = nextPowerOf2(block_size) / 2;
		 shift > 0;
		 shift >>= 1)
	{
		if (tid_x < shift && tid_x + shift < block_size)
		{
			smem[tid_x] += smem[tid_x + shift];
		}
		__syncthreads();
	}

	if (tid_x == 0)
		d_row_vec[bid_y] = smem[tid_x];
}

/*
 * Sum elements of matrix in each column
 *  m: number of rows of the matrix
 *  n: number of columns of the matrix
 */
void gpu_sum_col (const double* const mat,
                  double* row_vec,
                  const int m,
                  const int n)
{
	double *d_mat;
	double *d_row_vec;

	// Allocate the device memory
	checkCudaErrors(cudaMalloc(&d_mat, m*n*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_row_vec, n*sizeof(double)));
	
	// Copy data from the host memory to the device memory
	checkCudaErrors(cudaMemcpy(d_mat, mat, m*n*sizeof(double), cudaMemcpyHostToDevice));
	
	dim3 n_threads(0, 0);
	dim3 n_blocks(0, 0);
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_REDUCTION;
	n_threads.y = 1;
	
	// Compute the grid size
	n_blocks.x = 1;
	n_blocks.y = n;
	
	device_sum_col_block <BLOCK_SIZE_REDUCTION> <<<n_blocks, n_threads>>> (d_mat, d_row_vec, m, n);
	
	// Copy data from the device memory to the host memory
	checkCudaErrors(cudaMemcpy(row_vec, d_row_vec, n*sizeof(double), cudaMemcpyDeviceToHost));

	// Free the device memory
	cudaFree(d_mat);
	cudaFree(d_row_vec);
}

// Kernel to compute the elementwise multiplication in computing dW1
__global__
void device_elementwise_mult (const double* const d_mat_da1,
							  const double* const d_mat_a1,
							  double* d_mat_dz1,
							  const int m,
							  const int n)
{
	// Global thread index in the grid
	const int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
	const int tid_y = threadIdx.y + blockDim.y*blockIdx.y;
	
	// If the thread is not inside the matrix D, return
	if (tid_x >= m || tid_y >= n)
	{
		return;
	}
	
	// Compute the index in the matrix
	int idx = tid_y*m + tid_x;
	
	// Apply the sigmoid function
	d_mat_dz1[idx] = d_mat_da1[idx]*d_mat_a1[idx]*(1.0 - d_mat_a1[idx]);
}

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
						   const int n)
{
	double *d_mat_da1;
	double *d_mat_a1;
	double *d_mat_dz1;
	
	// Allocate the device memory
	checkCudaErrors(cudaMalloc(&d_mat_da1, m*n*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_a1, m*n*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_dz1, m*n*sizeof(double)));
	
	// Copy data from the host memory to the device memory
	checkCudaErrors(cudaMemcpy(d_mat_da1, mat_da1, m*n*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_a1, mat_a1, m*n*sizeof(double), cudaMemcpyHostToDevice));
	
	dim3 n_threads(0, 0);
	dim3 n_blocks(0, 0);
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_elementwise_mult;
	n_threads.y = BLOCK_SIZE_y_elementwise_mult;
	
	// Compute the grid size
	n_blocks.x = (m + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (n + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do the elementwise multiplication
	device_elementwise_mult <<<n_blocks, n_threads>>> (d_mat_da1, d_mat_a1, d_mat_dz1, m, n);
	
	// Copy data from the device memory to the host memory
	checkCudaErrors(cudaMemcpy(mat_dz1, d_mat_dz1, m*n*sizeof(double), cudaMemcpyDeviceToHost));
	
	// Free the device memory
	cudaFree(d_mat_da1);
	cudaFree(d_mat_a1);
	cudaFree(d_mat_dz1);
}

// Kernel to compute the elementwise subtraction
// C = A - alpha * B
__global__
void device_elementwise_subtract (const double alpha,
								  const double* const d_mat_A,
								  const double* const d_mat_B,
								  double* d_mat_C,
								  const int m,
								  const int n)
{
	// Global thread index in the grid
	const int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
	const int tid_y = threadIdx.y + blockDim.y*blockIdx.y;
	
	// If the thread is not inside the matrix D, return
	if (tid_x >= m || tid_y >= n)
	{
		return;
	}
	
	// Compute the index in the matrix
	int idx = tid_y*m + tid_x;
	
	// Apply the sigmoid function
	d_mat_C[idx] = d_mat_A[idx] - alpha*d_mat_B[idx];
}

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
							   const int n)
{
	double *d_mat_A;
	double *d_mat_B;
	double *d_mat_C;
	
	// Allocate the device memory
	checkCudaErrors(cudaMalloc(&d_mat_A, m*n*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_B, m*n*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_C, m*n*sizeof(double)));
	
	// Copy data from the host memory to the device memory
	checkCudaErrors(cudaMemcpy(d_mat_A, mat_A, m*n*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_B, mat_B, m*n*sizeof(double), cudaMemcpyHostToDevice));
	
	dim3 n_threads(0, 0);
	dim3 n_blocks(0, 0);
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_elementwise_subtract;
	n_threads.y = BLOCK_SIZE_y_elementwise_subtract;
	
	// Compute the grid size
	n_blocks.x = (m + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (n + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do the elementwise substraction
	device_elementwise_subtract <<<n_blocks, n_threads>>> (alpha, d_mat_A, d_mat_B, d_mat_C, m, n);
	
	// Copy data from the device memory to the host memory
	checkCudaErrors(cudaMemcpy(mat_C, d_mat_C, m*n*sizeof(double), cudaMemcpyDeviceToHost));
	
	// Free the device memory
	cudaFree(d_mat_A);
	cudaFree(d_mat_B);
	cudaFree(d_mat_C);
}

// Kernel to compute the diff matrix used in gpu_accel_feedforward_backprop()
__global__
void device_compute_diff (const double* const d_mat_yc,
						  const double* const d_mat_y,
						  double* d_mat_diff,
						  const int m,
						  const int n)
{
	// Global thread index in the grid
	const int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
	const int tid_y = threadIdx.y + blockDim.y*blockIdx.y;
	
	// If the thread is not inside the matrix D, return
	if (tid_x >= m || tid_y >= n)
	{
		return;
	}
	
	// Compute the index in the matrix
	int idx = tid_y*m + tid_x;
	
	// Apply the difference function
	d_mat_diff[idx] = 1.0/((double) m) * (d_mat_yc[idx] - d_mat_y[idx]);
}

/*
 * Compute the diff matrix
 * and returns a new matrix by GPU
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
					   const int n)
{
	double *d_mat_yc;
	double *d_mat_y;
	double *d_mat_diff;
	
	// Allocate the device memory
	checkCudaErrors(cudaMalloc(&d_mat_yc, m*n*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_y, m*n*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_diff, m*n*sizeof(double)));
	
	// Copy data from the host memory to the device memory
	checkCudaErrors(cudaMemcpy(d_mat_yc, mat_yc, m*n*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_y, mat_y, m*n*sizeof(double), cudaMemcpyHostToDevice));
	
	dim3 n_threads(0, 0);
	dim3 n_blocks(0, 0);
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_DIFF;
	n_threads.y = BLOCK_SIZE_y_DIFF;
	
	// Compute the grid size
	n_blocks.x = (m + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (n + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to apply the sigmoid function
	device_compute_diff <<<n_blocks, n_threads>>> (d_mat_yc, d_mat_y, d_mat_diff, m, n);
	
	// Copy data from the device memory to the host memory
	checkCudaErrors(cudaMemcpy(mat_diff, d_mat_diff, m*n*sizeof(double), cudaMemcpyDeviceToHost));
	
	// Free the device memory
	cudaFree(d_mat_yc);
	cudaFree(d_mat_y);
	cudaFree(d_mat_diff);
}

// Kernel to transpose a matrix
__global__
void device_transpose (double* d_mat_in, double* d_mat_out, const int m, const int n)
{
    const int row = threadIdx.x + blockDim.x * blockIdx.x;
    const int col = threadIdx.y + blockDim.y * blockIdx.y;
	
	if (row >= m || col >= n)
	{
		return;
	}
	
	const int idx_in = col*m + row;
	const int idx_out = row*n + col;
	
    d_mat_out[idx_out] = d_mat_in[idx_in];
}

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
					const int n)
{
	double *d_mat_1;
	double *d_mat_2;
	
	// Allocate the device memory
	checkCudaErrors(cudaMalloc(&d_mat_1, m*n*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_2, m*n*sizeof(double)));
	
	// Copy data from the host memory to the device memory
	checkCudaErrors(cudaMemcpy(d_mat_1, mat_1, m*n*sizeof(double), cudaMemcpyHostToDevice));
	
	dim3 n_threads(0, 0);
	dim3 n_blocks(0, 0);
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_transpose;
	n_threads.y = BLOCK_SIZE_y_transpose;
	
	// Compute the grid size
	n_blocks.x = (m + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (n + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do the transpose
	device_transpose <<<n_blocks, n_threads>>> (d_mat_1, d_mat_2, m, n);
	
	// Copy data from the device memory to the host memory
	checkCudaErrors(cudaMemcpy(mat_2, d_mat_2, m*n*sizeof(double), cudaMemcpyDeviceToHost));
	
	// Free the device memory
	cudaFree(d_mat_1);
	cudaFree(d_mat_2);
}

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
                            double* mat_a2, int a2_n_rows, int a2_n_cols)
{
	double* d_mat_X;
	double* d_mat_W1;
	double* d_mat_b1;
	double* d_mat_z1;
	double* d_mat_a1;
	double* d_mat_W2;
	double* d_mat_b2;
	double* d_mat_z2;
	double* d_mat_a2;
	
	dim3 n_threads(0, 0);
	dim3 n_blocks(0, 0);
	
	// Allocate the device memory
	checkCudaErrors(cudaMalloc(&d_mat_X, X_n_rows*X_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_W1, W1_n_rows*W1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_b1, b1_n_rows*b1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_z1, z1_n_rows*z1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_a1, a1_n_rows*a1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_W2, W2_n_rows*W2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_b2, b2_n_rows*b2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_z2, z2_n_rows*z2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_a2, a2_n_rows*a2_n_cols*sizeof(double)));
	
	// Copy data from the host memory to the device memory
	checkCudaErrors(cudaMemcpy(d_mat_X, mat_X, X_n_rows*X_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_W1, mat_W1, W1_n_rows*W1_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_b1, mat_b1, b1_n_rows*b1_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_W2, mat_W2, W2_n_rows*W2_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_b2, mat_b2, b2_n_rows*b2_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	
	/*
	 * Do the feedforward for z1
	 */
	
	// Set the size of the sub-block
	n_threads.x = BLOCK_SIZE_2;
	n_threads.y = BLOCK_SIZE_2;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (z1_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (z1_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do GEMM
	device_GEMM_2 <BLOCK_SIZE_2> <<<n_blocks, n_threads>>> (1.0,
														    1.0,
														    d_mat_X,
														    d_mat_W1,
														    d_mat_b1,
														    d_mat_z1,
														    X_n_rows,
														    X_n_cols,
														    W1_n_rows,
														    false,
														    true);
		
	/*
	 * Compute the sigmoid of z1
	 */
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_SIGMOID;
	n_threads.y = BLOCK_SIZE_y_SIGMOID;
	
	// Compute the grid size
	n_blocks.x = (z1_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (z1_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to apply the sigmoid function
	device_sigmoid <<<n_blocks, n_threads>>> (d_mat_z1, d_mat_a1, z1_n_rows, z1_n_cols);
	
	/*
	 * Do the feedforward for z2
	 */
	
	// Set the size of the sub-block
	n_threads.x = BLOCK_SIZE_2;
	n_threads.y = BLOCK_SIZE_2;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (z2_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (z2_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do GEMM
	device_GEMM_2 <BLOCK_SIZE_2> <<<n_blocks, n_threads>>> (1.0,
														    1.0,
														    d_mat_a1,
														    d_mat_W2,
														    d_mat_b2,
														    d_mat_z2,
														    a1_n_rows,
														    a1_n_cols,
														    W2_n_rows,
														    false,
														    true);
	
	/*
	 * Compute the softmax of z2
	 */
	
	double* d_col_vec;
	checkCudaErrors(cudaMalloc(&d_col_vec, z2_n_rows*sizeof(double)));
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_SOFTMAX;
	n_threads.y = BLOCK_SIZE_y_SOFTMAX;
	
	// Compute the grid size
	n_blocks.x = (z2_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (z2_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to compute the elementwise exponentinal function
	device_exponent <<<n_blocks, n_threads>>> (d_mat_z2, d_mat_a2, z2_n_rows, z2_n_cols);
	
	// Compute the block dimension
	n_threads.x = 1;
	n_threads.y = BLOCK_SIZE_REDUCTION;
	
	// Compute the grid size
	n_blocks.x = a2_n_rows;
	n_blocks.y = 1;
	
	device_sum_row_block <BLOCK_SIZE_REDUCTION> <<<n_blocks, n_threads>>> (d_mat_a2, d_col_vec, a2_n_rows, a2_n_cols);
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_SOFTMAX;
	n_threads.y = BLOCK_SIZE_y_SOFTMAX;
	
	// Compute the grid size
	n_blocks.x = (a2_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (a2_n_cols + n_threads.y - 1)/n_threads.y;
	
	device_normalize <<<n_blocks, n_threads>>> (d_mat_a2, d_col_vec, a2_n_rows, a2_n_cols);
	
	cudaFree(d_col_vec);
	
	// Copy data from the device memory to the host memory
	checkCudaErrors(cudaMemcpy(mat_z1, d_mat_z1, z1_n_rows*z1_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(mat_a1, d_mat_a1, a1_n_rows*a1_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(mat_z2, d_mat_z2, z2_n_rows*z2_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(mat_a2, d_mat_a2, a2_n_rows*a2_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
		
	// Free the device memory
	cudaFree(d_mat_X);
	cudaFree(d_mat_W1);
	cudaFree(d_mat_b1);
	cudaFree(d_mat_z1);
	cudaFree(d_mat_a1);
	cudaFree(d_mat_W2);
	cudaFree(d_mat_b2);
	cudaFree(d_mat_z2);
	cudaFree(d_mat_a2);
}

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
                         double* mat_db2, const int db2_n_cols)
{
	double* d_mat_diff;
	double* d_mat_X;
	double* d_mat_a1;
	double* d_mat_W1;
	double* d_mat_W2;
	double* d_mat_dz1;
	double* d_mat_da1;
	double* d_mat_dW1;
	double* d_mat_dW2;
	double* d_mat_db1;
	double* d_mat_db2;
	
	dim3 n_threads(0, 0);
	dim3 n_blocks(0, 0);
	
	// Allocate the device memory
	checkCudaErrors(cudaMalloc(&d_mat_diff, diff_n_rows*diff_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_X, X_n_rows*X_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_a1, a1_n_rows*a1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_W1, W1_n_rows*W1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_W2, W2_n_rows*W2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_dz1, diff_n_rows*W2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_da1, diff_n_rows*W2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_dW1, dW1_n_rows*dW1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_dW2, dW2_n_rows*dW2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_db1, db1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_db2, db2_n_cols*sizeof(double)));
	
	// Copy data from the host memory to the device memory
	checkCudaErrors(cudaMemcpy(d_mat_diff, mat_diff, diff_n_rows*diff_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_X, mat_X, X_n_rows*X_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_a1, mat_a1, a1_n_rows*a1_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_W1, mat_W1, W1_n_rows*W1_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_W2, mat_W2, W2_n_rows*W2_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	
	/*
	 * Compute dW2
	 */
	
	// Set the size of the sub-block
	n_threads.x = BLOCK_SIZE_2;
	n_threads.y = BLOCK_SIZE_2;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (dW2_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (dW2_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do GEMM
	device_GEMM_2 <BLOCK_SIZE_2> <<<n_blocks, n_threads>>> (1.0,
														    reg,
														    d_mat_diff,
														    d_mat_a1,
														    d_mat_W2,
														    d_mat_dW2,
														    diff_n_cols,
														    diff_n_rows,
														    a1_n_cols,
														    true,
														    false);
	
	/*
	 * Compute dW1
	 */
	
	n_threads.x = BLOCK_SIZE_2;
	n_threads.y = BLOCK_SIZE_2;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (diff_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (W2_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do GEMM
	device_GEMM_2 <BLOCK_SIZE_2> <<<n_blocks, n_threads>>> (1.0,
														    0.0,
														    d_mat_diff,
														    d_mat_W2,
														    d_mat_da1,
														    d_mat_da1,
														    diff_n_rows,
														    diff_n_cols,
														    W2_n_cols,
														    false,
														    false);

	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_elementwise_mult;
	n_threads.y = BLOCK_SIZE_y_elementwise_mult;
	
	// Compute the grid size
	n_blocks.x = (diff_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (W2_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do the elementwise multiplication
	device_elementwise_mult <<<n_blocks, n_threads>>> (d_mat_da1, d_mat_a1, d_mat_dz1, diff_n_rows, W2_n_cols);
	
	n_threads.x = BLOCK_SIZE_2;
	n_threads.y = BLOCK_SIZE_2;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (W2_n_cols + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (X_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do GEMM
	device_GEMM_2 <BLOCK_SIZE_2> <<<n_blocks, n_threads>>> (1.0,
														    reg,
														    d_mat_dz1,
														    d_mat_X,
														    d_mat_W1,
														    d_mat_dW1,
														    W2_n_cols,
														    diff_n_rows,
														    X_n_cols,
														    true,
														    false);
	
	/*
	 * Compute db1
	 */
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_REDUCTION;
	n_threads.y = 1;
	
	// Compute the grid size
	n_blocks.x = 1;
	n_blocks.y = diff_n_rows;
	
	device_sum_col_block <BLOCK_SIZE_REDUCTION> <<<n_blocks, n_threads>>> (d_mat_diff, d_mat_db2, diff_n_cols, diff_n_rows);
	
	/*
	 * Compute db2
	 */
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_REDUCTION;
	n_threads.y = 1;
	
	// Compute the grid size
	n_blocks.x = 1;
	n_blocks.y = W2_n_cols;
	
	device_sum_col_block <BLOCK_SIZE_REDUCTION> <<<n_blocks, n_threads>>> (d_mat_dz1, d_mat_db1, diff_n_rows, W2_n_cols);
	
	// Copy data from the device memory to the host memory
	checkCudaErrors(cudaMemcpy(mat_dW1, d_mat_dW1, dW1_n_rows*dW1_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(mat_dW2, d_mat_dW2, dW2_n_rows*dW2_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(mat_db1, d_mat_db1, db1_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(mat_db2, d_mat_db2, db2_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
	
	// Free the device memory
	cudaFree(d_mat_diff);
	cudaFree(d_mat_X);
	cudaFree(d_mat_a1);
	cudaFree(d_mat_W1);
	cudaFree(d_mat_W2);
	cudaFree(d_mat_dz1);
	cudaFree(d_mat_da1);
	cudaFree(d_mat_dW1);
	cudaFree(d_mat_dW2);
	cudaFree(d_mat_db1);
	cudaFree(d_mat_db2);
}

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
                                       double* mat_db2, const int db2_n_cols)
{
	double* d_mat_X;
	double* d_mat_W1;
	double* d_mat_b1;
	double* d_mat_z1;
	double* d_mat_a1;
	double* d_mat_W2;
	double* d_mat_b2;
	double* d_mat_z2;
	double* d_mat_a2;
	
	double* d_mat_dz1;
	double* d_mat_da1;
	double* d_mat_dW1;
	double* d_mat_dW2;
	double* d_mat_db1;
	double* d_mat_db2;
	
	double* d_mat_y;
	double* d_mat_diff;
	
	dim3 n_threads(0, 0);
	dim3 n_blocks(0, 0);
	
	/*
	 * Allocate the device memory
	 */
	
	checkCudaErrors(cudaMalloc(&d_mat_X, X_n_rows*X_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_W1, W1_n_rows*W1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_b1, b1_n_rows*b1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_z1, z1_n_rows*z1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_a1, a1_n_rows*a1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_W2, W2_n_rows*W2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_b2, b2_n_rows*b2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_z2, z2_n_rows*z2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_a2, a2_n_rows*a2_n_cols*sizeof(double)));
	
	checkCudaErrors(cudaMalloc(&d_mat_dz1, y_n_rows*W2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_da1, y_n_rows*W2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_dW1, dW1_n_rows*dW1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_dW2, dW2_n_rows*dW2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_db1, db1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_db2, db2_n_cols*sizeof(double)));
	
	checkCudaErrors(cudaMalloc(&d_mat_y, y_n_rows*y_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_diff, y_n_rows*y_n_cols*sizeof(double)));
	
	/*
	 * Copy data from the host memory to the device memory
	 */
	
	checkCudaErrors(cudaMemcpy(d_mat_X, mat_X, X_n_rows*X_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_W1, mat_W1, W1_n_rows*W1_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_b1, mat_b1, b1_n_rows*b1_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_W2, mat_W2, W2_n_rows*W2_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_b2, mat_b2, b2_n_rows*b2_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMemcpy(d_mat_y, mat_y, y_n_rows*y_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	
	/*
	 * Do the feedforward for z1
	 */
	
	// Set the size of the sub-block
	n_threads.x = BLOCK_SIZE_2;
	n_threads.y = BLOCK_SIZE_2;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (z1_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (z1_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do GEMM
	device_GEMM_2 <BLOCK_SIZE_2> <<<n_blocks, n_threads>>> (1.0,
														    1.0,
														    d_mat_X,
														    d_mat_W1,
														    d_mat_b1,
														    d_mat_z1,
														    X_n_rows,
														    X_n_cols,
														    W1_n_rows,
														    false,
														    true);
		
	/*
	 * Compute the sigmoid of z1
	 */
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_SIGMOID;
	n_threads.y = BLOCK_SIZE_y_SIGMOID;
	
	// Compute the grid size
	n_blocks.x = (z1_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (z1_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to apply the sigmoid function
	device_sigmoid <<<n_blocks, n_threads>>> (d_mat_z1, d_mat_a1, z1_n_rows, z1_n_cols);
	
	/*
	 * Do the feedforward for z2
	 */
	
	// Set the size of the sub-block
	n_threads.x = BLOCK_SIZE_2;
	n_threads.y = BLOCK_SIZE_2;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (z2_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (z2_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do GEMM
	device_GEMM_2 <BLOCK_SIZE_2> <<<n_blocks, n_threads>>> (1.0,
														    1.0,
														    d_mat_a1,
														    d_mat_W2,
														    d_mat_b2,
														    d_mat_z2,
														    a1_n_rows,
														    a1_n_cols,
														    W2_n_rows,
														    false,
														    true);
	
	/*
	 * Compute the softmax of z2
	 */
	
	double* d_col_vec;
	checkCudaErrors(cudaMalloc(&d_col_vec, z2_n_rows*sizeof(double)));
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_SOFTMAX;
	n_threads.y = BLOCK_SIZE_y_SOFTMAX;
	
	// Compute the grid size
	n_blocks.x = (z2_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (z2_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to compute the elementwise exponentinal function
	device_exponent <<<n_blocks, n_threads>>> (d_mat_z2, d_mat_a2, z2_n_rows, z2_n_cols);
	
	// Compute the block dimension
	n_threads.x = 1;
	n_threads.y = BLOCK_SIZE_REDUCTION;
	
	// Compute the grid size
	n_blocks.x = a2_n_rows;
	n_blocks.y = 1;
	
	device_sum_row_block <BLOCK_SIZE_REDUCTION> <<<n_blocks, n_threads>>> (d_mat_a2, d_col_vec, a2_n_rows, a2_n_cols);
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_SOFTMAX;
	n_threads.y = BLOCK_SIZE_y_SOFTMAX;
	
	// Compute the grid size
	n_blocks.x = (a2_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (a2_n_cols + n_threads.y - 1)/n_threads.y;
	
	device_normalize <<<n_blocks, n_threads>>> (d_mat_a2, d_col_vec, a2_n_rows, a2_n_cols);
	
	cudaFree(d_col_vec);
	
	/*
	 * Compute d_mat_diff
	 */
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_DIFF;
	n_threads.y = BLOCK_SIZE_y_DIFF;
	
	// Compute the grid size
	n_blocks.x = (y_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (y_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to apply the sigmoid function
	device_compute_diff <<<n_blocks, n_threads>>> (d_mat_a2, d_mat_y, d_mat_diff, y_n_rows, y_n_cols);
	
	/*
	 * Compute dW2
	 */
	
	// Set the size of the sub-block
	n_threads.x = BLOCK_SIZE_2;
	n_threads.y = BLOCK_SIZE_2;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (dW2_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (dW2_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do GEMM
	device_GEMM_2 <BLOCK_SIZE_2> <<<n_blocks, n_threads>>> (1.0,
														    reg,
														    d_mat_diff,
														    d_mat_a1,
														    d_mat_W2,
														    d_mat_dW2,
														    y_n_cols,
														    y_n_rows,
														    a1_n_cols,
														    true,
														    false);
	
	/*
	 * Compute dW1
	 */
	
	n_threads.x = BLOCK_SIZE_2;
	n_threads.y = BLOCK_SIZE_2;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (y_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (W2_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do GEMM
	device_GEMM_2 <BLOCK_SIZE_2> <<<n_blocks, n_threads>>> (1.0,
														    0.0,
														    d_mat_diff,
														    d_mat_W2,
														    d_mat_da1,
														    d_mat_da1,
														    y_n_rows,
														    y_n_cols,
														    W2_n_cols,
														    false,
														    false);

	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_elementwise_mult;
	n_threads.y = BLOCK_SIZE_y_elementwise_mult;
	
	// Compute the grid size
	n_blocks.x = (y_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (W2_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do the elementwise multiplication
	device_elementwise_mult <<<n_blocks, n_threads>>> (d_mat_da1, d_mat_a1, d_mat_dz1, y_n_rows, W2_n_cols);
	
	n_threads.x = BLOCK_SIZE_2;
	n_threads.y = BLOCK_SIZE_2;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (W2_n_cols + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (X_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do GEMM
	device_GEMM_2 <BLOCK_SIZE_2> <<<n_blocks, n_threads>>> (1.0,
														    reg,
														    d_mat_dz1,
														    d_mat_X,
														    d_mat_W1,
														    d_mat_dW1,
														    W2_n_cols,
														    y_n_rows,
														    X_n_cols,
														    true,
														    false);
	
	/*
	 * Compute db1
	 */
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_REDUCTION;
	n_threads.y = 1;
	
	// Compute the grid size
	n_blocks.x = 1;
	n_blocks.y = y_n_rows;
	
	device_sum_col_block <BLOCK_SIZE_REDUCTION> <<<n_blocks, n_threads>>> (d_mat_diff, d_mat_db2, y_n_cols, y_n_rows);
	
	/*
	 * Compute db2
	 */
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_REDUCTION;
	n_threads.y = 1;
	
	// Compute the grid size
	n_blocks.x = 1;
	n_blocks.y = W2_n_cols;
	
	device_sum_col_block <BLOCK_SIZE_REDUCTION> <<<n_blocks, n_threads>>> (d_mat_dz1, d_mat_db1, y_n_rows, W2_n_cols);
	
	/*
	 * Copy data from the device memory to the host memory
	 */
	
	checkCudaErrors(cudaMemcpy(mat_z1, d_mat_z1, z1_n_rows*z1_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(mat_a1, d_mat_a1, a1_n_rows*a1_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(mat_z2, d_mat_z2, z2_n_rows*z2_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(mat_a2, d_mat_a2, a2_n_rows*a2_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
	
	checkCudaErrors(cudaMemcpy(mat_dW1, d_mat_dW1, dW1_n_rows*dW1_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(mat_dW2, d_mat_dW2, dW2_n_rows*dW2_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(mat_db1, d_mat_db1, db1_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(mat_db2, d_mat_db2, db2_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
		
	/*
	 * Free the device memory
	 */
	
	cudaFree(d_mat_X);
	cudaFree(d_mat_W1);
	cudaFree(d_mat_b1);
	cudaFree(d_mat_z1);
	cudaFree(d_mat_a1);
	cudaFree(d_mat_W2);
	cudaFree(d_mat_b2);
	cudaFree(d_mat_z2);
	cudaFree(d_mat_a2);
	
	cudaFree(d_mat_dz1);
	cudaFree(d_mat_da1);
	cudaFree(d_mat_dW1);
	cudaFree(d_mat_dW2);
	cudaFree(d_mat_db1);
	cudaFree(d_mat_db2);
	
	cudaFree(d_mat_y);
	cudaFree(d_mat_diff);
}

/*
 * The stream version of gpu_accel_feedforward_backprop_2
 * This function is not preferred since the cost of pinning
 * host memory is large
 */
void gpu_accel_feedforward_backprop_2_w_stream (const double reg,
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
												double* mat_db2, const int db2_n_cols)
{
	double* d_mat_X;
	double* d_mat_W1;
	double* d_mat_b1;
	double* d_mat_z1;
	double* d_mat_W2;
	double* d_mat_b2;
	double* d_mat_z2;
	
	double* d_mat_dz1;
	double* d_mat_da1;
	double* d_mat_dW1;
	double* d_mat_dW2;
	double* d_mat_db1;
	double* d_mat_db2;
	
	double* d_mat_y;
	double* d_mat_diff;
	
	dim3 n_threads(0, 0);
	dim3 n_blocks(0, 0);
	
	/*
	 * Create user streams
	 */
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	
	/*
	 * Allocate the device memory
	 */
	
	checkCudaErrors(cudaMalloc(&d_mat_X, X_n_rows*X_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_W1, W1_n_rows*W1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_b1, b1_n_rows*b1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_z1, X_n_rows*W1_n_rows*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_W2, W2_n_rows*W2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_b2, b2_n_rows*b2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_z2, X_n_rows*W2_n_rows*sizeof(double)));
	
	checkCudaErrors(cudaMalloc(&d_mat_dz1, y_n_rows*W2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_da1, y_n_rows*W2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_dW1, dW1_n_rows*dW1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_dW2, dW2_n_rows*dW2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_db1, db1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_db2, db2_n_cols*sizeof(double)));
	
	checkCudaErrors(cudaMalloc(&d_mat_y, y_n_rows*y_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_diff, y_n_rows*y_n_cols*sizeof(double)));
	
	/*
	 * Pin the host memory
	 */
	
	checkCudaErrors(cudaHostRegister(mat_X, X_n_rows*X_n_cols*sizeof(double), 0));
	checkCudaErrors(cudaHostRegister(mat_W1, W1_n_rows*W1_n_cols*sizeof(double), 0));
	checkCudaErrors(cudaHostRegister(mat_b1, b1_n_rows*b1_n_cols*sizeof(double), 0));
	checkCudaErrors(cudaHostRegister(mat_W2, W2_n_rows*W2_n_cols*sizeof(double), 0));
	checkCudaErrors(cudaHostRegister(mat_b2, b2_n_rows*b2_n_cols*sizeof(double), 0));
	checkCudaErrors(cudaHostRegister(mat_y, y_n_rows*y_n_cols*sizeof(double), 0));
	
	checkCudaErrors(cudaHostRegister(mat_a2, a2_n_rows*a2_n_cols*sizeof(double), 0));
	
	checkCudaErrors(cudaHostRegister(mat_dW1, dW1_n_rows*dW1_n_cols*sizeof(double), 0));
	checkCudaErrors(cudaHostRegister(mat_dW2, dW2_n_rows*dW2_n_cols*sizeof(double), 0));
	checkCudaErrors(cudaHostRegister(mat_db1, db1_n_cols*sizeof(double), 0));
	checkCudaErrors(cudaHostRegister(mat_db2, db2_n_cols*sizeof(double), 0));
	
	/*
	 * Copy data from the host memory to the device memory
	 */
	
	checkCudaErrors(cudaMemcpyAsync(d_mat_X, mat_X, X_n_rows*X_n_cols*sizeof(double), cudaMemcpyHostToDevice, stream1));
	checkCudaErrors(cudaMemcpyAsync(d_mat_W1, mat_W1, W1_n_rows*W1_n_cols*sizeof(double), cudaMemcpyHostToDevice, stream1));
	checkCudaErrors(cudaMemcpyAsync(d_mat_b1, mat_b1, b1_n_rows*b1_n_cols*sizeof(double), cudaMemcpyHostToDevice, stream1));
	
	checkCudaErrors(cudaMemcpyAsync(d_mat_W2, mat_W2, W2_n_rows*W2_n_cols*sizeof(double), cudaMemcpyHostToDevice, stream2));
	checkCudaErrors(cudaMemcpyAsync(d_mat_b2, mat_b2, b2_n_rows*b2_n_cols*sizeof(double), cudaMemcpyHostToDevice, stream2));
	
	/*
	 * Do the feedforward for z1
	 */
	
	// Set the size of the sub-block
	n_threads.x = BLOCK_SIZE_2;
	n_threads.y = BLOCK_SIZE_2;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (X_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (W1_n_rows + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do GEMM
	device_GEMM_2 <BLOCK_SIZE_2> <<<n_blocks, n_threads, 0, stream1>>> (1.0,
														    1.0,
														    d_mat_X,
														    d_mat_W1,
														    d_mat_b1,
														    d_mat_z1,
														    X_n_rows,
														    X_n_cols,
														    W1_n_rows,
														    false,
														    true);
		
	/*
	 * Compute the sigmoid of z1
	 */
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_SIGMOID;
	n_threads.y = BLOCK_SIZE_y_SIGMOID;
	
	// Compute the grid size
	n_blocks.x = (X_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (W1_n_rows + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to apply the sigmoid function
	device_sigmoid <<<n_blocks, n_threads, 0, stream1>>> (d_mat_z1, d_mat_z1, X_n_rows, W1_n_rows);
	
	// Synchronize to make sure d_mat_W2, d_mat_b2 are already loaded
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpyAsync(d_mat_y, mat_y, y_n_rows*y_n_cols*sizeof(double), cudaMemcpyHostToDevice, stream2));
	
	/*
	 * Do the feedforward for z2
	 */
	
	// Set the size of the sub-block
	n_threads.x = BLOCK_SIZE_2;
	n_threads.y = BLOCK_SIZE_2;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (X_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (W2_n_rows + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do GEMM
	device_GEMM_2 <BLOCK_SIZE_2> <<<n_blocks, n_threads, 0, stream1>>> (1.0,
														    1.0,
														    d_mat_z1,
														    d_mat_W2,
														    d_mat_b2,
														    d_mat_z2,
														    X_n_rows,
														    W1_n_rows,
														    W2_n_rows,
														    false,
														    true);
	
	/*
	 * Compute the softmax of z2
	 */
	
	double* d_col_vec;
	checkCudaErrors(cudaMalloc(&d_col_vec, X_n_rows*sizeof(double)));
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_SOFTMAX;
	n_threads.y = BLOCK_SIZE_y_SOFTMAX;
	
	// Compute the grid size
	n_blocks.x = (X_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (W2_n_rows + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to compute the elementwise exponentinal function
	device_exponent <<<n_blocks, n_threads, 0, stream1>>> (d_mat_z2, d_mat_z2, X_n_rows, W2_n_rows);
	
	// Compute the block dimension
	n_threads.x = 1;
	n_threads.y = BLOCK_SIZE_REDUCTION;
	
	// Compute the grid size
	n_blocks.x = a2_n_rows;
	n_blocks.y = 1;
	
	device_sum_row_block <BLOCK_SIZE_REDUCTION> <<<n_blocks, n_threads, 0, stream1>>> (d_mat_z2, d_col_vec, a2_n_rows, a2_n_cols);
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_SOFTMAX;
	n_threads.y = BLOCK_SIZE_y_SOFTMAX;
	
	// Compute the grid size
	n_blocks.x = (a2_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (a2_n_cols + n_threads.y - 1)/n_threads.y;
	
	device_normalize <<<n_blocks, n_threads, 0, stream1>>> (d_mat_z2, d_col_vec, a2_n_rows, a2_n_cols);
	
	cudaFree(d_col_vec);
	
	// Synchronize to make sure d_mat_y is already loaded
	cudaDeviceSynchronize();
	
	/*
	 * Compute d_mat_diff
	 */
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_DIFF;
	n_threads.y = BLOCK_SIZE_y_DIFF;
	
	// Compute the grid size
	n_blocks.x = (y_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (y_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to apply the sigmoid function
	device_compute_diff <<<n_blocks, n_threads, 0, stream1>>> (d_mat_z2, d_mat_y, d_mat_diff, y_n_rows, y_n_cols);
	
	/*
	 * Compute dW2
	 */
	
	// Set the size of the sub-block
	n_threads.x = BLOCK_SIZE_2;
	n_threads.y = BLOCK_SIZE_2;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (dW2_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (dW2_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do GEMM
	device_GEMM_2 <BLOCK_SIZE_2> <<<n_blocks, n_threads, 0, stream1>>> (1.0,
														    reg,
														    d_mat_diff,
														    d_mat_z1,
														    d_mat_W2,
														    d_mat_dW2,
														    y_n_cols,
														    y_n_rows,
														    W1_n_rows,
														    true,
														    false);
	
	/*
	 * Compute dW1
	 */
	
	n_threads.x = BLOCK_SIZE_2;
	n_threads.y = BLOCK_SIZE_2;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (y_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (W2_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do GEMM
	device_GEMM_2 <BLOCK_SIZE_2> <<<n_blocks, n_threads, 0, stream1>>> (1.0,
														    0.0,
														    d_mat_diff,
														    d_mat_W2,
														    d_mat_da1,
														    d_mat_da1,
														    y_n_rows,
														    y_n_cols,
														    W2_n_cols,
														    false,
														    false);

	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_elementwise_mult;
	n_threads.y = BLOCK_SIZE_y_elementwise_mult;
	
	// Compute the grid size
	n_blocks.x = (y_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (W2_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do the elementwise multiplication
	device_elementwise_mult <<<n_blocks, n_threads, 0, stream1>>> (d_mat_da1, d_mat_z1, d_mat_dz1, y_n_rows, W2_n_cols);
	
	n_threads.x = BLOCK_SIZE_2;
	n_threads.y = BLOCK_SIZE_2;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (W2_n_cols + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (X_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do GEMM
	device_GEMM_2 <BLOCK_SIZE_2> <<<n_blocks, n_threads, 0, stream1>>> (1.0,
														    reg,
														    d_mat_dz1,
														    d_mat_X,
														    d_mat_W1,
														    d_mat_dW1,
														    W2_n_cols,
														    y_n_rows,
														    X_n_cols,
														    true,
														    false);
	
	/*
	 * Compute db1
	 */
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_REDUCTION;
	n_threads.y = 1;
	
	// Compute the grid size
	n_blocks.x = 1;
	n_blocks.y = y_n_rows;
	
	device_sum_col_block <BLOCK_SIZE_REDUCTION> <<<n_blocks, n_threads, 0, stream1>>> (d_mat_diff, d_mat_db2, y_n_cols, y_n_rows);
	
	/*
	 * Compute db2
	 */
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_REDUCTION;
	n_threads.y = 1;
	
	// Compute the grid size
	n_blocks.x = 1;
	n_blocks.y = W2_n_cols;
	
	device_sum_col_block <BLOCK_SIZE_REDUCTION> <<<n_blocks, n_threads, 0, stream1>>> (d_mat_dz1, d_mat_db1, y_n_rows, W2_n_cols);
	
	/*
	 * Copy data from the device memory to the host memory
	 */
	
	cudaDeviceSynchronize();
	
	checkCudaErrors(cudaMemcpyAsync(mat_dW1, d_mat_dW1, dW1_n_rows*dW1_n_cols*sizeof(double), cudaMemcpyDeviceToHost, stream2));
	checkCudaErrors(cudaMemcpyAsync(mat_a2, d_mat_z2, a2_n_rows*a2_n_cols*sizeof(double), cudaMemcpyDeviceToHost, stream2));
	checkCudaErrors(cudaMemcpyAsync(mat_db1, d_mat_db1, db1_n_cols*sizeof(double), cudaMemcpyDeviceToHost, stream2));
	checkCudaErrors(cudaMemcpyAsync(mat_db2, d_mat_db2, db2_n_cols*sizeof(double), cudaMemcpyDeviceToHost, stream2));
	checkCudaErrors(cudaMemcpyAsync(mat_dW2, d_mat_dW2, dW2_n_rows*dW2_n_cols*sizeof(double), cudaMemcpyDeviceToHost, stream2));
		
	/*
	 * Free the device memory
	 */
	
	cudaFree(d_mat_X);
	cudaFree(d_mat_W1);
	cudaFree(d_mat_b1);
	cudaFree(d_mat_z1);
	cudaFree(d_mat_W2);
	cudaFree(d_mat_b2);
	cudaFree(d_mat_z2);
	
	cudaFree(d_mat_dz1);
	cudaFree(d_mat_da1);
	cudaFree(d_mat_dW1);
	cudaFree(d_mat_dW2);
	cudaFree(d_mat_db1);
	cudaFree(d_mat_db2);
	
	cudaFree(d_mat_y);
	cudaFree(d_mat_diff);
	
	/*
	 * Unpin the host memory
	 */
	
	checkCudaErrors(cudaHostUnregister(mat_X));
	checkCudaErrors(cudaHostUnregister(mat_W1));
	checkCudaErrors(cudaHostUnregister(mat_b1));
	checkCudaErrors(cudaHostUnregister(mat_W2));
	checkCudaErrors(cudaHostUnregister(mat_b2));
	checkCudaErrors(cudaHostUnregister(mat_y));
	
	checkCudaErrors(cudaHostUnregister(mat_a2));
	
	checkCudaErrors(cudaHostUnregister(mat_dW1));
	checkCudaErrors(cudaHostUnregister(mat_dW2));
	checkCudaErrors(cudaHostUnregister(mat_db1));
	checkCudaErrors(cudaHostUnregister(mat_db2));
	
	/*
	 * Destroy user streams
	 */
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);	
}

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
                                       double* mat_db2, const int db2_n_cols)
{
	double* d_mat_X;
	double* d_mat_W1;
	double* d_mat_b1;
	double* d_mat_z1;
	double* d_mat_W2;
	double* d_mat_b2;
	double* d_mat_z2;
	
	double* d_mat_dz1;
	double* d_mat_da1;
	double* d_mat_dW1;
	double* d_mat_dW2;
	double* d_mat_db1;
	double* d_mat_db2;
	
	double* d_mat_y;
	double* d_mat_diff;
	
	double* d_mat_W1_t;
	double* d_mat_W2_t;
	double* d_mat_diff_t;
	double* d_mat_dz1_t;
	
	dim3 n_threads(0, 0);
	dim3 n_blocks(0, 0);

	/*
	 * Allocate the device memory
	 */
	
	checkCudaErrors(cudaMalloc(&d_mat_X, X_n_rows*X_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_W1, W1_n_rows*W1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_b1, b1_n_rows*b1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_z1, X_n_rows*W1_n_rows*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_W2, W2_n_rows*W2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_b2, b2_n_rows*b2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_z2, X_n_rows*W2_n_rows*sizeof(double)));
	
	checkCudaErrors(cudaMalloc(&d_mat_dz1, y_n_rows*W2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_da1, y_n_rows*W2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_dW1, dW1_n_rows*dW1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_dW2, dW2_n_rows*dW2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_db1, db1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_db2, db2_n_cols*sizeof(double)));
	
	checkCudaErrors(cudaMalloc(&d_mat_y, y_n_rows*y_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_diff, y_n_rows*y_n_cols*sizeof(double)));
	
	checkCudaErrors(cudaMalloc(&d_mat_W1_t, W1_n_rows*W1_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_W2_t, W2_n_rows*W2_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_diff_t, y_n_rows*y_n_cols*sizeof(double)));
	checkCudaErrors(cudaMalloc(&d_mat_dz1_t, y_n_rows*W2_n_cols*sizeof(double)));
	
	/*
	 * Copy data from the host memory to the device memory
	 */
	
	checkCudaErrors(cudaMemcpy(d_mat_X, mat_X, X_n_rows*X_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_W1, mat_W1, W1_n_rows*W1_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_b1, mat_b1, b1_n_rows*b1_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_W2, mat_W2, W2_n_rows*W2_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_mat_b2, mat_b2, b2_n_rows*b2_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMemcpy(d_mat_y, mat_y, y_n_rows*y_n_cols*sizeof(double), cudaMemcpyHostToDevice));
	
	/*
	 * Do the feedforward for z1
	 */
	
	// Transpose d_mat_W1
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_transpose;
	n_threads.y = BLOCK_SIZE_y_transpose;
	
	// Compute the grid size
	n_blocks.x = (W1_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (W1_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do the transpose
	device_transpose <<<n_blocks, n_threads>>> (d_mat_W1, d_mat_W1_t, W1_n_rows, W1_n_cols);
	
	// Set the size of each block
	n_threads.x = BLOCK_SIZE_x_3;
	n_threads.y = BLOCK_SIZE_y_3;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (X_n_rows + n_threads.x*n_threads.y - 1)/(n_threads.x*n_threads.y);
	n_blocks.y = (W1_n_rows + n_threads.x - 1)/n_threads.x;
	
	// Launch the kernel to do GEMM
	device_GEMM_3 <BLOCK_SIZE_x_3, BLOCK_SIZE_y_3> <<<n_blocks, n_threads>>> (1.0,
																			  1.0,
																			  d_mat_X,
																			  d_mat_W1_t,
																			  d_mat_b1,
																			  d_mat_z1,
																			  X_n_rows,
																			  X_n_cols,
																			  W1_n_rows,
																			  false,
																			  false);
	
	/*
	 * Compute the sigmoid of z1
	 */
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_SIGMOID;
	n_threads.y = BLOCK_SIZE_y_SIGMOID;
	
	// Compute the grid size
	n_blocks.x = (X_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (W1_n_rows + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to apply the sigmoid function
	device_sigmoid <<<n_blocks, n_threads>>> (d_mat_z1, d_mat_z1, X_n_rows, W1_n_rows);
	
	/*
	 * Do the feedforward for z2
	 */
	
	// Transpose d_mat_W2
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_transpose;
	n_threads.y = BLOCK_SIZE_y_transpose;
	
	// Compute the grid size
	n_blocks.x = (W2_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (W2_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do the transpose
	device_transpose <<<n_blocks, n_threads>>> (d_mat_W2, d_mat_W2_t, W2_n_rows, W2_n_cols);
	
	// Set the size of each block
	n_threads.x = BLOCK_SIZE_x_3;
	n_threads.y = BLOCK_SIZE_y_3;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (X_n_rows + n_threads.x*n_threads.y - 1)/(n_threads.x*n_threads.y);
	n_blocks.y = (W2_n_rows + n_threads.x - 1)/n_threads.x;
	
	// Launch the kernel to do GEMM
	device_GEMM_3 <BLOCK_SIZE_x_3, BLOCK_SIZE_y_3> <<<n_blocks, n_threads>>> (1.0,
																			  1.0,
																			  d_mat_z1,
																			  d_mat_W2_t,
																			  d_mat_b2,
																			  d_mat_z2,
																			  X_n_rows,
																			  W1_n_rows,
																			  W2_n_rows,
																			  false,
																			  false);

	/*
	 * Compute the softmax of z2
	 */
	
	double* d_col_vec;
	checkCudaErrors(cudaMalloc(&d_col_vec, X_n_rows*sizeof(double)));
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_SOFTMAX;
	n_threads.y = BLOCK_SIZE_y_SOFTMAX;
	
	// Compute the grid size
	n_blocks.x = (X_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (W2_n_rows + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to compute the elementwise exponentinal function
	device_exponent <<<n_blocks, n_threads>>> (d_mat_z2, d_mat_z2, X_n_rows, W2_n_rows);
	
	// Compute the block dimension
	n_threads.x = 1;
	n_threads.y = BLOCK_SIZE_REDUCTION;
	
	// Compute the grid size
	n_blocks.x = a2_n_rows;
	n_blocks.y = 1;
	
	device_sum_row_block <BLOCK_SIZE_REDUCTION> <<<n_blocks, n_threads>>> (d_mat_z2, d_col_vec, a2_n_rows, a2_n_cols);
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_SOFTMAX;
	n_threads.y = BLOCK_SIZE_y_SOFTMAX;
	
	// Compute the grid size
	n_blocks.x = (a2_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (a2_n_cols + n_threads.y - 1)/n_threads.y;
	
	device_normalize <<<n_blocks, n_threads>>> (d_mat_z2, d_col_vec, a2_n_rows, a2_n_cols);
	
	cudaFree(d_col_vec);

	/*
	 * Compute d_mat_diff
	 */
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_DIFF;
	n_threads.y = BLOCK_SIZE_y_DIFF;
	
	// Compute the grid size
	n_blocks.x = (y_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (y_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to apply the sigmoid function
	device_compute_diff <<<n_blocks, n_threads>>> (d_mat_z2, d_mat_y, d_mat_diff, y_n_rows, y_n_cols);

	/*
	 * Compute dW2
	 */
	
	// Transpose d_mat_diff
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_transpose;
	n_threads.y = BLOCK_SIZE_y_transpose;
	
	// Compute the grid size
	n_blocks.x = (y_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (y_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do the transpose
	device_transpose <<<n_blocks, n_threads>>> (d_mat_diff, d_mat_diff_t, y_n_rows, y_n_cols);
	
	// Set the size of each block
	n_threads.x = BLOCK_SIZE_x_3;
	n_threads.y = BLOCK_SIZE_y_3;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (dW2_n_rows + n_threads.x*n_threads.y - 1)/(n_threads.x*n_threads.y);
	n_blocks.y = (dW2_n_cols + n_threads.x - 1)/n_threads.x;
	
	// Launch the kernel to do GEMM
	device_GEMM_3 <BLOCK_SIZE_x_3, BLOCK_SIZE_y_3> <<<n_blocks, n_threads>>> (1.0,
																			  reg,
																			  d_mat_diff_t,
																			  d_mat_z1,
																			  d_mat_W2,
																			  d_mat_dW2,
																			  y_n_cols,
																			  y_n_rows,
																			  W1_n_rows,
																			  false,
																			  false);

	/*
	 * Compute dW1
	 */
	
	// Set the size of each block
	n_threads.x = BLOCK_SIZE_x_3;
	n_threads.y = BLOCK_SIZE_y_3;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (y_n_rows + n_threads.x*n_threads.y - 1)/(n_threads.x*n_threads.y);
	n_blocks.y = (W2_n_cols + n_threads.x - 1)/n_threads.x;
	
	// Launch the kernel to do GEMM
	device_GEMM_3 <BLOCK_SIZE_x_3, BLOCK_SIZE_y_3> <<<n_blocks, n_threads>>> (1.0,
																			  0.0,
																			  d_mat_diff,
																			  d_mat_W2,
																			  d_mat_da1,
																			  d_mat_da1,
																			  y_n_rows,
																			  y_n_cols,
																			  W2_n_cols,
																			  false,
																			  false);

	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_elementwise_mult;
	n_threads.y = BLOCK_SIZE_y_elementwise_mult;
	
	// Compute the grid size
	n_blocks.x = (y_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (W2_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do the elementwise multiplication
	device_elementwise_mult <<<n_blocks, n_threads>>> (d_mat_da1, d_mat_z1, d_mat_dz1, y_n_rows, W2_n_cols);
	
	// Transpose d_mat_dz1
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_x_transpose;
	n_threads.y = BLOCK_SIZE_y_transpose;
	
	// Compute the grid size
	n_blocks.x = (y_n_rows + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (W2_n_cols + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel to do the transpose
	device_transpose <<<n_blocks, n_threads>>> (d_mat_dz1, d_mat_dz1_t, y_n_rows, W2_n_cols);
	
	// Set the size of each block
	n_threads.x = BLOCK_SIZE_x_3;
	n_threads.y = BLOCK_SIZE_y_3;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (W2_n_cols + n_threads.x*n_threads.y - 1)/(n_threads.x*n_threads.y);
	n_blocks.y = (X_n_cols + n_threads.x - 1)/n_threads.x;
	
	// Launch the kernel to do GEMM
	device_GEMM_3 <BLOCK_SIZE_x_3, BLOCK_SIZE_y_3> <<<n_blocks, n_threads>>> (1.0,
																			  reg,
																			  d_mat_dz1_t,
																			  d_mat_X,
																			  d_mat_W1,
																			  d_mat_dW1,
																			  W2_n_cols,
																			  y_n_rows,
																			  X_n_cols,
																			  false,
																			  false);

	/*
	 * Compute db1
	 */
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_REDUCTION;
	n_threads.y = 1;
	
	// Compute the grid size
	n_blocks.x = 1;
	n_blocks.y = y_n_rows;
	
	device_sum_col_block <BLOCK_SIZE_REDUCTION> <<<n_blocks, n_threads>>> (d_mat_diff, d_mat_db2, y_n_cols, y_n_rows);

	/*
	 * Compute db2
	 */
	
	// Compute the block dimension
	n_threads.x = BLOCK_SIZE_REDUCTION;
	n_threads.y = 1;
	
	// Compute the grid size
	n_blocks.x = 1;
	n_blocks.y = W2_n_cols;
	
	device_sum_col_block <BLOCK_SIZE_REDUCTION> <<<n_blocks, n_threads>>> (d_mat_dz1, d_mat_db1, y_n_rows, W2_n_cols);

	/*
	 * Copy data from the device memory to the host memory
	 */
	checkCudaErrors(cudaMemcpy(mat_a2, d_mat_z2, X_n_rows*W2_n_rows*sizeof(double), cudaMemcpyDeviceToHost));	
	checkCudaErrors(cudaMemcpy(mat_dW1, d_mat_dW1, dW1_n_rows*dW1_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(mat_dW2, d_mat_dW2, dW2_n_rows*dW2_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(mat_db1, d_mat_db1, db1_n_cols*sizeof(double), cudaMemcpyDeviceToHost));	
	checkCudaErrors(cudaMemcpy(mat_db2, d_mat_db2, db2_n_cols*sizeof(double), cudaMemcpyDeviceToHost));
	
	/*
	 * Free the device memory
	 */
	
	cudaFree(d_mat_X);
	cudaFree(d_mat_W1);
	cudaFree(d_mat_b1);
	cudaFree(d_mat_z1);
	cudaFree(d_mat_W2);
	cudaFree(d_mat_b2);
	cudaFree(d_mat_z2);
	
	cudaFree(d_mat_dz1);
	cudaFree(d_mat_da1);
	cudaFree(d_mat_dW1);
	cudaFree(d_mat_dW2);
	cudaFree(d_mat_db1);
	cudaFree(d_mat_db2);
	
	cudaFree(d_mat_y);
	cudaFree(d_mat_diff);
	
	cudaFree(d_mat_W1_t);
	cudaFree(d_mat_W2_t);
	cudaFree(d_mat_diff_t);
	cudaFree(d_mat_dz1_t);
}
