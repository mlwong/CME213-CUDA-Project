#ifndef UTILS_TEST_UTILS_H_
#define UTILS_TEST_UTILS_H_

#include <armadillo>
#include <stdlib.h> 
#include <timer.h>

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

/* Test the gpu_GEMM_1 function */
bool test_gpu_GEMM_1a(int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, C and D
	arma::mat A = arma::randn (m,n);
	arma::mat B = arma::randn (n,l);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_1(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, false);
	
	D = alpha*A*B + beta*C;
	
	return almost_equal_matrix(D, D_gpu.memptr(), true);
	

}

/* Test the gpu_GEMM_1 function */
bool test_gpu_GEMM_1b(int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, C and D
	arma::mat A = arma::randn (n,m);
	arma::mat B = arma::randn (n,l);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_1(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, false);
	
	D = alpha*A.t()*B + beta*C;

	return almost_equal_matrix(D, D_gpu.memptr(), true);
}

/* Test the gpu_GEMM_1 function */
bool test_gpu_GEMM_1c(int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, C and D
	arma::mat A = arma::randn (m,n);
	arma::mat B = arma::randn (l,n);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_1(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, true);
	
	D = alpha*A*B.t() + beta*C;

	return almost_equal_matrix(D, D_gpu.memptr(), true);
}

/* Test the gpu_GEMM_1 function */
bool test_gpu_GEMM_1d(int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, C and D
	arma::mat A = arma::randn (n,m);
	arma::mat B = arma::randn (l,n);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_1(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, true);
	
	D = alpha*A.t()*B.t() + beta*C;

	return almost_equal_matrix(D, D_gpu.memptr(), true);
}

/* Test the gpu_GEMM_2 function */
bool test_gpu_GEMM_2a(int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, C and D
	arma::mat A = arma::randn (m,n);
	arma::mat B = arma::randn (n,l);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_2(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, false);
	
	D = alpha*A*B + beta*C;
	
	return almost_equal_matrix(D, D_gpu.memptr(), true);
}

/* Test the gpu_GEMM_2 function */
bool test_gpu_GEMM_2b(int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, C and D
	arma::mat A = arma::randn (n,m);
	arma::mat B = arma::randn (n,l);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_2(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, false);
	
	D = alpha*A.t()*B + beta*C;
	
	return almost_equal_matrix(D, D_gpu.memptr(), true);
}

/* Test the gpu_GEMM_2 function */
bool test_gpu_GEMM_2c(int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, C and D
	arma::mat A = arma::randn (m,n);
	arma::mat B = arma::randn (l,n);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_2(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, false, true);
	
	D = alpha*A*B.t() + beta*C;
	
	return almost_equal_matrix(D, D_gpu.memptr(), true);
}

/* Test the gpu_GEMM_2 function */
bool test_gpu_GEMM_2d(int m, int n, int l)
{
	// Generate random values for alpha and beta
	// between -1.0 to 1.0
	srand(time(NULL));
	
	double alpha =  (double) (rand() % 2000) / 1000.0 - 1.0; 
	double beta  =  (double) (rand() % 2000) / 1000.0 - 1.0;
	
	// Generate random values for matrices A, B, C and D
	arma::mat A = arma::randn (n,m);
	arma::mat B = arma::randn (l,n);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_2(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l, true, true);
	
	D = alpha*A.t()*B.t() + beta*C;
	
	return almost_equal_matrix(D, D_gpu.memptr(), true);
}

#endif
