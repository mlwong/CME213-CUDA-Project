#ifndef UTILS_TEST_UTILS_H_
#define UTILS_TEST_UTILS_H_

#include <armadillo>

#include "../gpu_func.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define MAX_ULPS_DIFF 5120
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
bool test_gpu_GEMM_1(int m, int n, int l)
{
	double alpha = 2.0;
	double beta = 3.0;
	arma::mat A = arma::randn (m,n);
	arma::mat B = arma::randn (n,l);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_1(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l);
	
	D = alpha*A*B + beta*C;
	
	return almost_equal_matrix(D, D_gpu.memptr(), true);	
}

/* Test the gpu_GEMM_2 function */
bool test_gpu_GEMM_2(int m, int n, int l)
{
	double alpha = 2.0;
	double beta = 3.0;
	arma::mat A = arma::randn (m,n);
	arma::mat B = arma::randn (n,l);
	arma::mat C = arma::randn (m,l);
	arma::mat D(m, l);
	arma::mat D_gpu(m, l);
	
	gpu_GEMM_2(alpha, beta, A.memptr(), B.memptr(), C.memptr(), D_gpu.memptr(), m, n, l);
	
	D = alpha*A*B + beta*C;
	
	double *d_gpu = D_gpu.memptr();
	double *d_cpu = D.memptr();
	
	/*
	std::cout << "D_cpu" << std::endl;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < l; j++)
		{
			std::cout << d_cpu[j*m+i] << " ";
		}
		std::cout << std::endl;
	}
	
	std::cout << "D_gpu" << std::endl;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < l; j++)
		{
			std::cout << d_gpu[j*m+i] << " ";
		}
		std::cout << std::endl;
	}
	*/
	
	return almost_equal_matrix(D, D_gpu.memptr(), true);
}

#endif