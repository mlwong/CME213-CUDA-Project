#ifndef MATRIX_LT_HPP
#define MATRIX_LT_HPP

#include <algorithm>
#include <stdexcept>

#include "matrix.hpp"

/* a lower triangular matrix class which inherits from the pure abstract matrix class */
template <typename T>
class LTMatrix: public Matrix<T>
{
    private:
        unsigned int n = 0;
        T *data = NULL;
        T zero = (T) 0;
        
        /*
         * method to get the index in the array data
         * the method take inputs row i and column j
         */
        unsigned int getIndex(unsigned int i, unsigned int j) const
        {
            unsigned int index = i*(i + 1)/2 + j;
            return index;
        }
        
    public:
        /*
         * constructor of the lower triangluar matrix class
         * the constructor takes an input n, the size of the matrix
         */
        LTMatrix(unsigned int n)
        {
            if (n == 0)
            {
                throw std::runtime_error("ERROR: Matrix should have non-zero size!");
            }
            
            // compute the number of elements that require storage
            unsigned int num_data = (n + 1)*n/2;
            
            // initialize the the attributes of the class
            this->data = new T[num_data];
            this->n = n;
            
            for (unsigned int i = 0; i < num_data; i++)
            {
                data[i] = (T) 0;
            }
        }
        
        /*
         * method to access and modify entries in the matrix
         * the method take inputs row i and column j
         */
        T& operator() (unsigned int i, unsigned int j)
        {
            if (i >= n || j >= n)
            {
                throw std::runtime_error("ERROR: Matrix index is invalid!");
            }
            
            if (j > i)
            {
                zero = (T) 0;
                return zero;
            }
            else
            {
                return data[getIndex(i, j)];
            }
            
            return data[getIndex(i, j)];
        }
        
        /* method to return the l0 norm */
        unsigned int computel0Norm(void)
        {
            unsigned int num_data = (n + 1)*n/2;
            unsigned int l0_norm = 0;
            
            // calcuate the total number of non-zero elements below the diagonal */
            l0_norm = std::count_if(data, data + num_data, [](T i){ return (i != (T) 0); } );
            
            return l0_norm;
        }
        
        /* destructor of the class */
        ~LTMatrix()
        {
            delete[] data;
            data = NULL;
        }
};

#endif /* MATRIX_LT.HPP */
