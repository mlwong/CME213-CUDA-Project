#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>

#define BLOCK_SIZE 32

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

__global__
void device_GEMM_1(const double alpha,
				   const double beta,
				   const double* const d_mat_A,
				   const double* const d_mat_B,
				   const double* const d_mat_C,
				   double* d_mat_D,
				   const int m,
				   const int n,
				   const int l)
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
	
	// sum is used to store the element of A*B
	// that is computed by the thread
	double sum = 0.0;
	
	for (int k = 0; k < n; k++)
	{
		int idx_A = k*m + tid_x;
		int idx_B = tid_y*n + k;
		
		sum += d_mat_A[idx_A]*d_mat_B[idx_B];
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
 * Algorithm 1 to use one thread to calculate element in matrix D
 * natively
 * m: number of rows of A
 * n: number of columns of A
 * l: number of columns of B
 */
void gpu_GEMM_1(const double alpha,
                const double beta,
                const double* const mat_A,
                const double* const mat_B,
                const double* const mat_C,
                double* mat_D,
			    const int m,
			    const int n,
			    const int l)
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
	const int block_dim_x = 64u;
	const int block_dim_y = 8u;
	
	// Compute the block dimension
	n_threads.x = block_dim_x;
	n_threads.y = block_dim_y;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (m + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (l + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel
	device_GEMM_1 <<<n_blocks, n_threads>>> (alpha, beta, d_mat_A, d_mat_B, d_mat_C, d_mat_D, m, n, l);
	
	// Copy data from the device memory to the host memory
	checkCudaErrors(cudaMemcpy(mat_D, d_mat_D, m*l*sizeof(double), cudaMemcpyDeviceToHost));
	
	// Free the device memory
	cudaFree(d_mat_A);
	cudaFree(d_mat_B);
	cudaFree(d_mat_C);
	cudaFree(d_mat_D);
}

template <int block_size>
__global__
void device_GEMM_2(const double alpha,
				   const double beta,
				   const double* const d_mat_A,
				   const double* const d_mat_B,
				   const double* const d_mat_C,
				   double* d_mat_D,
				   const int m,
				   const int n,
				   const int l)
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
	
	// Index of the first sub-matrix of A processed by the block
	int mat_A_begin = block_size*bid_x;
	
	// Step size used to iterate through the sub-matrices of A
	int mat_A_step  = block_size*m;
	
	// Index of the first sub-matrix of B processed by the block
	int mat_B_begin = n*block_size*bid_y;
	
	// Index of the last sub-matrix of B process by the blcok
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
		// Declaration of the shared memory used to store the
		// sub-matrix of A
		__shared__ double mat_A_shared[block_size][block_size];
		
		// Declaration of the shared memory used to store the
		// sub-matrix of B
		__shared__ double mat_B_shared[block_size][block_size];
		
		// Load the matrices from device memory to shared memory;
		// Each threat loads one element of each sub-matrix
		
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
				//sum += mat_A_shared[tid_y][k]*mat_B_shared[k][tid_x];
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

/*
 * Algorithm 2 to use a blocking algorithm and shared memory
 * m: number of rows of A
 * n: number of columns of A
 * l: number of columns of B
 */
void gpu_GEMM_2(const double alpha,
                const double beta,
                const double* const mat_A,
                const double* const mat_B,
                const double* const mat_C,
                double* mat_D,
			    const int m,
			    const int n,
			    const int l)
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
	n_threads.x = BLOCK_SIZE;
	n_threads.y = BLOCK_SIZE;
	
	// Assume each dimension of the block is less than 65536
	// and compute the grid size
	n_blocks.x = (m + n_threads.x - 1)/n_threads.x;
	n_blocks.y = (l + n_threads.y - 1)/n_threads.y;
	
	// Launch the kernel
	device_GEMM_2 <BLOCK_SIZE> <<<n_blocks, n_threads>>> (alpha, beta, d_mat_A, d_mat_B, d_mat_C, d_mat_D, m, n, l);
	
	// Copy data from the device memory to the host memory
	checkCudaErrors(cudaMemcpy(mat_D, d_mat_D, m*l*sizeof(double), cudaMemcpyDeviceToHost));
	
	// Free the device memory
	cudaFree(d_mat_A);
	cudaFree(d_mat_B);
	cudaFree(d_mat_C);
	cudaFree(d_mat_D);
}
