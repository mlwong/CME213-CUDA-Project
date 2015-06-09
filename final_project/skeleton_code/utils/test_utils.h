#ifndef UTILS_TEST_UTILS_H_
#define UTILS_TEST_UTILS_H_

#include <armadillo>
#include <stdlib.h> 
#include <timer.h>

#include "common.h"
#include "../gpu_func.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define MAX_ULPS_DIFF 10000000
union Double_t
{
	Double_t (double num) : d(num) {}
	bool Negative() const {return (i >> 63) != 0; }
	int64_t i;
	double d;
};

bool almost_equal_ulps (double A, double B, int maxUlpsDiff)
{
	Double_t uA(A);
	Double_t uB(B);
	// Different signs means they do not match
	if (uA.Negative() != uB.Negative()) {
		// check for equality to make sure +0 == -0
		if (A == B)
			return true;
		return false;
	}
	// Find the difference in ULPs
	int ulpsDiff = abs (uA.i - uB.i);
	if (ulpsDiff <= maxUlpsDiff)
		return true;
	return false;
}


/*
	almost_equal_matrix compares an Armadillo matrix with
	a memory space representing a matrix. A matrix may be
	stored in memory in a row-major way (e.g. A[row][column]
	in C) or a column-major way (as Armadillo and Fortran
	does).

	Example:
		arma::mat m = 0.0001 * arma::randn (100, 1000);
		arma::mat b = m + (double) 0.00000000000001l;
		double *b_memptr = b.memptr ();	
		// retrieve a raw pointer to the underlying memory space of arma::mat b,
		// mat is stored in a column major way, you can use any pointer to 
		// a raw memory space with almost_equal_matrix
	    bool same = almost_equal_matrix (m, b_memptr, true);
	    std::cout << "Compare result: " << same << std::endl;
*/
bool almost_equal_matrix (const arma::mat& M, 
						  const double* const memptr, 
						  bool column_major) {
	int n_rows = M.n_rows;
	int n_cols = M.n_cols;
	int idx;

	for (int j = 0; j < n_cols; j++) {
		for (int i = 0; i < n_rows; i++) {
			// compute the idx based on column_major or 
			idx = (column_major == true)?(j*n_rows+i):(i*n_cols+j);
			double A = M(i,j);
			double B = memptr[idx];
			if (almost_equal_ulps (A,B,MAX_ULPS_DIFF) == false)
				return false;
		}
	}
	return true;
}

/* Test the gpu_sigmoid function */
void test_gpu_sigmoid (int m, int n)
{
	// Generate random values for matrix A
	arma::mat A = arma::randn (m,n);
	arma::mat B(m,n);
	arma::mat B_gpu(m,n);
	
	double start, end;
	
	start = MPI_Wtime();
	sigmoid (A, B);
	end = MPI_Wtime();
	std::cout << "  CPU sigmoid speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_sigmoid (A.memptr(), B_gpu.memptr(), m, n);
	end = MPI_Wtime();
	std::cout << "  gpu_sigmoid() speed: " << end - start << std::endl;
	
	if (almost_equal_matrix(B, B_gpu.memptr(), true))
	{
		std::cout << "  Test on gpu_sigmoid() PASSED for m = "
				  << m << ", "
				  << "n = " << n << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_sigmoid() FAILED for m = "
				  << m << ", "
				  << "n = " << n << std::endl;
		exit(1);
	}
}

/* Test the gpu_softmax function */
void test_gpu_softmax (int m, int n)
{
	// Generate random values for matrix A
	arma::mat A = arma::randn (m,n);
	arma::mat B(m,n);
	arma::mat B_gpu(m,n);
	
	double start, end;
	
	start = MPI_Wtime();
	softmax (A, B);
	end = MPI_Wtime();
	std::cout << "  CPU softmax speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_softmax (A.memptr(), B_gpu.memptr(), m, n);
	end = MPI_Wtime();
	std::cout << "  gpu_softmax() speed: " << end - start << std::endl;
	
	if (almost_equal_matrix(B, B_gpu.memptr(), true))
	{
		std::cout << "  Test on gpu_softmax() PASSED for m = "
				  << m << ", "
				  << "n = " << n << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_softmax() FAILED for m = "
				  << m << ", "
				  << "n = " << n << std::endl;
		exit(1);
	}
}

/* Test the gpu_sum_col function */
void test_gpu_sum_col(int m, int n)
{
	// Generate random values for matrix A
	arma::mat A = arma::randn (m,n);
	arma::rowvec B(n);
	arma::rowvec B_gpu(n);
	
	double start, end;
	
	start = MPI_Wtime();
	B = arma::sum (A, 0);
	end = MPI_Wtime();
	std::cout << "  CPU column reduction speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_sum_col (A.memptr(), B_gpu.memptr(), m, n);
	end = MPI_Wtime();
	std::cout << "  gpu_sum_col() speed: " << end - start << std::endl;
	
	if (almost_equal_matrix(B, B_gpu.memptr(), true))
	{
		std::cout << "  Test on gpu_sum_col() PASSED for m = "
				  << m << ", "
				  << "n = " << n << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_sum_col() FAILED for m = "
				  << m << ", "
				  << "n = " << n << std::endl;
		exit(1);
	}
}

/* Test the gpu_elementwise_mult function */
void test_gpu_elementwise_mult (int m, int n)
{
	// Generate random values for matrices A and B
	arma::mat A = arma::randn (m,n);
	arma::mat B = arma::randn (m,n);
	arma::mat C(m,n);
	arma::mat C_gpu(m,n);
	
	double start, end;
	
	start = MPI_Wtime();
	C = A % B % (1 - B);
	end = MPI_Wtime();
	std::cout << "  CPU elementwise multiplication speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_elementwise_mult(A.memptr(), B.memptr(), C_gpu.memptr(), m, n);
	end = MPI_Wtime();
	std::cout << "  gpu_elementwise_mult() speed: " << end - start << std::endl;
	
	if (almost_equal_matrix(C, C_gpu.memptr(), true))
	{
		std::cout << "  Test on gpu_elementwise_mult() PASSED for m = "
				  << m << ", "
				  << "n = " << n << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_elementwise_mult() FAILED for m = "
				  << m << ", "
				  << "n = " << n << std::endl;
		exit(1);
	}
}

/* Test the gpu_elementwise_gpu_elementwise_subtract function */
void test_gpu_elementwise_subtract (int m, int n)
{
	// Generate random values for matrices A and B
	arma::mat A = arma::randn (m,n);
	arma::mat B = arma::randn (m,n);
	arma::mat C(m,n);
	arma::mat C_gpu(m,n);
	
	double start, end;
	
	start = MPI_Wtime();
	C = A - 2.0*B;
	end = MPI_Wtime();
	std::cout << "  CPU elementwise subtraction speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_elementwise_subtract(2.0, A.memptr(), B.memptr(), C_gpu.memptr(), m, n);
	end = MPI_Wtime();
	std::cout << "  gpu_elementwise_subtract() speed: " << end - start << std::endl;
	
	if (almost_equal_matrix(C, C_gpu.memptr(), true))
	{
		std::cout << "  Test on gpu_elementwise_subtract() PASSED for m = "
				  << m << ", "
				  << "n = " << n << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_elementwise_subtract() FAILED for m = "
				  << m << ", "
				  << "n = " << n << std::endl;
		exit(1);
	}
}

/* Test the gpu_compute_diff function */
void test_gpu_compute_diff (int m, int n)
{
	// Generate random values for matrices A and B
	arma::mat A = arma::randn (m,n);
	arma::mat B = arma::randn (m,n);
	arma::mat C(m,n);
	arma::mat C_gpu(m,n);
	
	double start, end;
	
	start = MPI_Wtime();
	C = (1.0 / m) * (A - B);
	end = MPI_Wtime();
	std::cout << "  CPU elementwise substraction speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_compute_diff(A.memptr(), B.memptr(), C_gpu.memptr(), m, n);
	end = MPI_Wtime();
	std::cout << "  gpu_compute_diff() speed: " << end - start << std::endl;
	
	if (almost_equal_matrix(C, C_gpu.memptr(), true))
	{
		std::cout << "  Test on gpu_compute_diff() PASSED for m = "
				  << m << ", "
				  << "n = " << n << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_compute_diff() FAILED for m = "
				  << m << ", "
				  << "n = " << n << std::endl;
		exit(1);
	}
}

/* Test the gpu_transpose function */
void test_gpu_transpose (int m, int n)
{
	// Generate random values for matrix A
	arma::mat A = arma::randn (m,n);
	arma::mat B(n, m);
	arma::mat B_gpu(n, m);
	
	double start, end;
	
	start = MPI_Wtime();
	B = A.t();
	end = MPI_Wtime();
	std::cout << "  CPU transpose speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_transpose (A.memptr(), B_gpu.memptr(), A.n_rows, A.n_cols);
	end = MPI_Wtime();
	std::cout << "  gpu_transpose() speed: " << end - start << std::endl;
	
	if (almost_equal_matrix(B, B_gpu.memptr(), true))
	{
		std::cout << "  Test on gpu_transpose() PASSED for m = "
				  << m << ", "
				  << "n = " << n << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_transpose() FAILED for m = "
				  << m << ", "
				  << "n = " << n << std::endl;
		exit(1);
	}
}

/* Test the gpu_GEMM_0 function */
void test_gpu_GEMM_0a (int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, and C
	arma::mat A = arma::randn (m,n);
	arma::mat B = arma::randn (n,l);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_0(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, false);
	
	D = alpha*A*B + beta*C;
	
	if (almost_equal_matrix(D, D_gpu.memptr(), true))
	{
		
		std::cout << "  Test on gpu_GEMM_0() PASSED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = false, transpose_B = false"
				  << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_GEMM_0() FAILED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = false, transpose_B = false"
				  << std::endl;
				  
		exit(1);
	}
}

/* Test the gpu_GEMM_0 function */
void test_gpu_GEMM_0b (int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, and C
	arma::mat A = arma::randn (n,m);
	arma::mat B = arma::randn (n,l);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_0(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, false);
	
	D = alpha*A.t()*B + beta*C;
	
	if (almost_equal_matrix(D, D_gpu.memptr(), true))
	{
		
		std::cout << "  Test on gpu_GEMM_0() PASSED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = true, transpose_B = false"
				  << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_GEMM_0() FAILED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = true, transpose_B = false"
				  << std::endl;
				  
		exit(1);
	}
}

/* Test the gpu_GEMM_0 function */
void test_gpu_GEMM_0c (int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, and C
	arma::mat A = arma::randn (m,n);
	arma::mat B = arma::randn (l,n);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_0(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, true);
	
	D = alpha*A*B.t() + beta*C;
	
	if (almost_equal_matrix(D, D_gpu.memptr(), true))
	{
		
		std::cout << "  Test on gpu_GEMM_0() PASSED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = false, transpose_B = true"
				  << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_GEMM_0() FAILED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = false, transpose_B = true"
				  << std::endl;
				  
		exit(1);
	}
}

/* Test the gpu_GEMM_0 function */
void test_gpu_GEMM_0d (int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, and C
	arma::mat A = arma::randn (n,m);
	arma::mat B = arma::randn (l,n);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_0(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, true);
	
	D = alpha*A.t()*B.t() + beta*C;
	
	if (almost_equal_matrix(D, D_gpu.memptr(), true))
	{
		
		std::cout << "  Test on gpu_GEMM_0() PASSED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = true, transpose_B = true"
				  << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_GEMM_0() FAILED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = true, transpose_B = true"
				  << std::endl;
				  
		exit(1);
	}
}

/* Test the gpu_GEMM_1 function */
void test_gpu_GEMM_1a (int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, and C
	arma::mat A = arma::randn (m,n);
	arma::mat B = arma::randn (n,l);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_1(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, false);
	
	D = alpha*A*B + beta*C;
	
	if (almost_equal_matrix(D, D_gpu.memptr(), true))
	{
		
		std::cout << "  Test on gpu_GEMM_1() PASSED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = false, transpose_B = false"
				  << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_GEMM_1() FAILED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = false, transpose_B = false"
				  << std::endl;
				  
		exit(1);
	}
}

/* Test the gpu_GEMM_1 function */
void test_gpu_GEMM_1b (int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, and C
	arma::mat A = arma::randn (n,m);
	arma::mat B = arma::randn (n,l);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_1(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, false);
	
	D = alpha*A.t()*B + beta*C;

	if (almost_equal_matrix(D, D_gpu.memptr(), true))
	{
		
		std::cout << "  Test on gpu_GEMM_1() PASSED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = true, transpose_B = false"
				  << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_GEMM_1() FAILED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = true, transpose_B = false"
				  << std::endl;
				  
		exit(1);
	}
}

/* Test the gpu_GEMM_1 function */
void test_gpu_GEMM_1c (int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, and C
	arma::mat A = arma::randn (m,n);
	arma::mat B = arma::randn (l,n);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_1(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, true);
	
	D = alpha*A*B.t() + beta*C;

	if (almost_equal_matrix(D, D_gpu.memptr(), true))
	{
		
		std::cout << "  Test on gpu_GEMM_1() PASSED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = false, transpose_B = true"
				  << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_GEMM_1() FAILED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = false, transpose_B = true"
				  << std::endl;
				  
		exit(1);
	}
}

/* Test the gpu_GEMM_1 function */
void test_gpu_GEMM_1d (int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, and C
	arma::mat A = arma::randn (n,m);
	arma::mat B = arma::randn (l,n);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_1(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, true);
	
	D = alpha*A.t()*B.t() + beta*C;

	if (almost_equal_matrix(D, D_gpu.memptr(), true))
	{
		
		std::cout << "  Test on gpu_GEMM_1() PASSED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = true, transpose_B = true"
				  << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_GEMM_1() FAILED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = true, transpose_B = true"
				  << std::endl;
				  
		exit(1);
	}
}

/* Test the gpu_GEMM_2 function */
void test_gpu_GEMM_2a (int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, and C
	arma::mat A = arma::randn (m,n);
	arma::mat B = arma::randn (n,l);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_2(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, false);
	
	D = alpha*A*B + beta*C;
	
	if (almost_equal_matrix(D, D_gpu.memptr(), true))
	{
		
		std::cout << "  Test on gpu_GEMM_2() PASSED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = false, transpose_B = false"
				  << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_GEMM_2() FAILED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = false, transpose_B = false"
				  << std::endl;
				  
		exit(1);
	}
}

/* Test the gpu_GEMM_2 function */
void test_gpu_GEMM_2b (int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, and C
	arma::mat A = arma::randn (n,m);
	arma::mat B = arma::randn (n,l);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_2(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, false);
	
	D = alpha*A.t()*B + beta*C;
	
	if (almost_equal_matrix(D, D_gpu.memptr(), true))
	{
		
		std::cout << "  Test on gpu_GEMM_2() PASSED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = true, transpose_B = false"
				  << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_GEMM_2() FAILED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = true, transpose_B = false"
				  << std::endl;
				  
		exit(1);
	}
}

/* Test the gpu_GEMM_2 function */
void test_gpu_GEMM_2c (int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, and C
	arma::mat A = arma::randn (m,n);
	arma::mat B = arma::randn (l,n);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_2(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, true);
	
	D = alpha*A*B.t() + beta*C;
	
	if (almost_equal_matrix(D, D_gpu.memptr(), true))
	{
		
		std::cout << "  Test on gpu_GEMM_2() PASSED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = false, transpose_B = true"
				  << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_GEMM_2() FAILED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = false, transpose_B = true"
				  << std::endl;
				  
		exit(1);
	}
}

/* Test the gpu_GEMM_2 function */
void test_gpu_GEMM_2d (int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, and C
	arma::mat A = arma::randn (n,m);
	arma::mat B = arma::randn (l,n);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_2(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, true);
	
	D = alpha*A.t()*B.t() + beta*C;
	
	if (almost_equal_matrix(D, D_gpu.memptr(), true))
	{
		
		std::cout << "  Test on gpu_GEMM_2() PASSED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = true, transpose_B = true"
				  << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_GEMM_2() FAILED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = true, transpose_B = true"
				  << std::endl;
				  
		exit(1);
	}
}

/* Test the gpu_GEMM_3 function */
void test_gpu_GEMM_3a (int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, and C
	arma::mat A = arma::randn (m,n);
	arma::mat B = arma::randn (n,l);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_3(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, false);
	
	D = alpha*A*B + beta*C;
	
	if (almost_equal_matrix(D, D_gpu.memptr(), true))
	{
		
		std::cout << "  Test on gpu_GEMM_3() PASSED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = false, transpose_B = false"
				  << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_GEMM_3() FAILED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = false, transpose_B = false"
				  << std::endl;
				  
		exit(1);
	}
}

/* Test the gpu_GEMM_3 function */
void test_gpu_GEMM_3b (int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, and C
	arma::mat A = arma::randn (n,m);
	arma::mat B = arma::randn (n,l);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_3(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, false);
	
	D = alpha*A.t()*B + beta*C;
	
	if (almost_equal_matrix(D, D_gpu.memptr(), true))
	{
		
		std::cout << "  Test on gpu_GEMM_3() PASSED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = true, transpose_B = false"
				  << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_GEMM_3() FAILED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = true, transpose_B = false"
				  << std::endl;
				  
		exit(1);
	}
}

/* Test the gpu_GEMM_3 function */
void test_gpu_GEMM_3c (int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, and C
	arma::mat A = arma::randn (m,n);
	arma::mat B = arma::randn (l,n);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_3(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, true);
	
	D = alpha*A*B.t() + beta*C;
	
	if (almost_equal_matrix(D, D_gpu.memptr(), true))
	{
		
		std::cout << "  Test on gpu_GEMM_3() PASSED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = false, transpose_B = true"
				  << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_GEMM_3() FAILED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = false, transpose_B = true"
				  << std::endl;
				  
		exit(1);
	}
}

/* Test the gpu_GEMM_3 function */
void test_gpu_GEMM_3d (int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, and C
	arma::mat A = arma::randn (n,m);
	arma::mat B = arma::randn (l,n);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_3(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, true);
	
	D = alpha*A.t()*B.t() + beta*C;
	
	if (almost_equal_matrix(D, D_gpu.memptr(), true))
	{
		
		std::cout << "  Test on gpu_GEMM_3() PASSED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = true, transpose_B = true"
				  << std::endl;
	}
	else
	{
		std::cout << "  Test on gpu_GEMM_3() FAILED for m = "
				  << m << ", "
				  << "n = " << n << ", "
				  << "l = " << l << ", "
				  << "transpose_A = true, transpose_B = true"
				  << std::endl;
				  
		exit(1);
	}
}

/* Test the speed of different GPU GEMM functions*/
void test_speed_GEMM (int m, int n, int l)
{
	
	std::cout << "  Matrices dimensions: m = "
			  << m
			  << ", n = "
			  << n
			  << ", l = "
			  << l
			  << std::endl;
	
	std::cout << std::endl;
			  
	double start, end;
	
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	std::cout << "  For transpose_A = false, transpose_B = false: " << std::endl;
	
	// Generate random values for matrices A, B, and C
	arma::mat A = arma::randn (m,n);
	arma::mat B = arma::randn (n,l);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m,l);
	arma::mat D_gpu(m,l);
	
	start = MPI_Wtime();
	D = alpha*A*B + beta*C;
	end = MPI_Wtime();
	std::cout << "  CPU GEMM speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_GEMM_0(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, false);
	end = MPI_Wtime();
	std::cout << "  gpu_GEMM_0() speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_GEMM_1(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, false);
	end = MPI_Wtime();
	std::cout << "  gpu_GEMM_1() speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_GEMM_2(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, false);
	end = MPI_Wtime();
	std::cout << "  gpu_GEMM_2() speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_GEMM_3(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, false);
	end = MPI_Wtime();
	std::cout << "  gpu_GEMM_3() speed: " << end - start << std::endl;
	
	std::cout << std::endl;
	
	std::cout << "  For transpose_A = true, transpose_B = false: " << std::endl;
	
	// Generate random values for matrices A, B and C
	A = arma::randn (n,m);
	B = arma::randn (n,l);
	C = arma::randn (m,l);
	
	start = MPI_Wtime();
	D = alpha*A.t()*B + beta*C;
	end = MPI_Wtime();
	std::cout << "  CPU GEMM speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_GEMM_0(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, false);
	end = MPI_Wtime();
	std::cout << "  gpu_GEMM_0() speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_GEMM_1(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, false);
	end = MPI_Wtime();
	std::cout << "  gpu_GEMM_1() speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_GEMM_2(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, false);
	end = MPI_Wtime();
	std::cout << "  gpu_GEMM_2() speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_GEMM_3(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, false);
	end = MPI_Wtime();
	std::cout << "  gpu_GEMM_3() speed: " << end - start << std::endl;
	
	std::cout << std::endl;
	
	std::cout << "  For transpose_A = false, transpose_B = true: " << std::endl;
	
	// Generate random values for matrices A, B and C
	A = arma::randn (m,n);
	B = arma::randn (l,n);
	C = arma::randn (m,l);
	
	start = MPI_Wtime();
	D = alpha*A*B.t() + beta*C;
	end = MPI_Wtime();
	std::cout << "  CPU GEMM speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_GEMM_0(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, true);
	end = MPI_Wtime();
	std::cout << "  gpu_GEMM_0() speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_GEMM_1(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, true);
	end = MPI_Wtime();
	std::cout << "  gpu_GEMM_1() speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_GEMM_2(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, true);
	end = MPI_Wtime();
	std::cout << "  gpu_GEMM_2() speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_GEMM_3(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, true);
	end = MPI_Wtime();
	std::cout << "  gpu_GEMM_3() speed: " << end - start << std::endl;
	
	std::cout << std::endl;
	
	std::cout << "  For transpose_A = true, transpose_B = true: " << std::endl;
	
	// Generate random values for matrices A, B and C
	A = arma::randn (n,m);
	B = arma::randn (l,n);
	C = arma::randn (m,l);
	
	start = MPI_Wtime();
	D = alpha*A.t()*B.t() + beta*C;
	end = MPI_Wtime();
	std::cout << "  CPU GEMM speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_GEMM_0(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, true);
	end = MPI_Wtime();
	std::cout << "  gpu_GEMM_0() speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_GEMM_1(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, true);
	end = MPI_Wtime();
	std::cout << "  gpu_GEMM_1() speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_GEMM_2(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, true);
	end = MPI_Wtime();
	std::cout << "  gpu_GEMM_2() speed: " << end - start << std::endl;
	
	start = MPI_Wtime();
	gpu_GEMM_3(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, true);
	end = MPI_Wtime();
	std::cout << "  gpu_GEMM_3() speed: " << end - start << std::endl;
}

#endif
