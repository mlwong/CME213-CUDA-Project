#ifndef MATRIX_HPP
#define MATRIX_HPP

/* a pure abstract base class for general matrices */
template<typename T>
class Matrix
{
    public:
        /*
         * virtual method to access and modify entries in the matrix
         * the method take inputs row i and column j
         */
        virtual T& operator() (unsigned int i, unsigned int j) = 0;
        
        /* virtual method to return the l0 norm */
        virtual unsigned int computel0Norm(void) = 0;
};

#endif /* MATRIX.HPP */
