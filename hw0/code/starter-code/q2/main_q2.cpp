# include <iostream>

#include "matrix_lt.hpp"

int main()
{
    /*
     * Test 1:
     * test whether the initializing and element accessing operations
     * of the LT matrix class are correctly implemented
     */
    try
    {
        /* do the test for double data type */
        LTMatrix<double> A_d(10);
        
        for (unsigned int i = 0; i < 10; i++)
        {
            for (unsigned int j = 0; j <= i; j++)
            {
                if (!(A_d(i, j) == 0.0))
                {
                    throw std::runtime_error("ERROR: Initialization operation is not implemented correctly!");
                }
            }
        }
        
        /* do the test for float data type */
        LTMatrix<float> A_f(10);
        
        for (unsigned int i = 0; i < 10; i++)
        {
            for (unsigned int j = 0; j <= i; j++)
            {
                if (!(A_f(i, j) == 0.0f))
                {
                    throw std::runtime_error("ERROR: Initialization operation is not implemented correctly!");
                }
            }
        }
    }
    catch(std::exception &e)
    {
        std::cout << "Test 1 did not pass!" << std::endl;
        std::cerr << "Exception caught!" << std::endl;
        std::cerr << "e.what() = " << e.what() << std::endl;
        return 1;
    }
    
    /*
     * Test 2:
     * test whether modification operation of entries are correctly
     * implemennted
     */
    try
    {
        /* do the test for double data type */
        LTMatrix<double> B_d(10);
        
        double counter1 = 1.0;
        
        for (unsigned int i = 0; i < 10; i++)
        {
            for (unsigned int j = 0; j <= i; j++)
            {
                B_d(i, j) = counter1;
                counter1++;
            }
        }
        
        double counter2 = 1.0;
        for (unsigned int i = 0; i < 10; i++)
        {
            for (unsigned int j = 0; j <= i; j++)
            {
                if (!(B_d(i, j) == counter2))
                {
                    throw std::runtime_error("ERROR: Modification operation is not implemented correctly!");
                }
                counter2++;
            }
        }
        
        /* do the test for float data type */
        LTMatrix<float> B_f(10);
        
        float counter3 = 1.0f;
        
        for (unsigned int i = 0; i < 10; i++)
        {
            for (unsigned int j = 0; j <= i; j++)
            {
                B_f(i, j) = counter3;
                counter3++;
            }
        }
        
        float counter4 = 1.0f;
        for (unsigned int i = 0; i < 10; i++)
        {
            for (unsigned int j = 0; j <= i; j++)
            {
                if (!(B_f(i, j) == counter4))
                {
                    throw std::runtime_error("ERROR: Modification operation is not implemented correctly!");
                }
                counter4++;
            }
        }
    }
    catch(std::exception &e)
    {
        std::cout << "Test 2 did not pass!" << std::endl;
        std::cerr << "Exception caught!" << std::endl;
        std::cerr << "e.what() = " << e.what() << std::endl;
        return 1;
    }
    
    /*
     * Test 3:
     * test whether the method to compute l0 norm is correctly
     * implemented
     */
    try
    {
        LTMatrix<double> C(10);
        
        if (!(C.computel0Norm() == 0))
        {
            throw std::runtime_error("ERROR: Method to compute l0 norm is not implemented correctly!");
        }
        
        C(0, 0) = 1.0;
        C(3, 2) = 1.1;
        
        if (!(C.computel0Norm() == 2))
        {
            throw std::runtime_error("ERROR: Method to compute l0 norm is not implemented correctly!");
        }
    }
    catch(std::exception &e)
    {
        std::cout << "Test 3 did not pass!" << std::endl;
        std::cerr << "Exception caught!" << std::endl;
        std::cerr << "e.what() = " << e.what() << std::endl;
        return 1;
    }
    
    /*
     * Test 4:
     * test all the exceptions
     */
    try
    {
        LTMatrix<double> D(0);
        
        std::cout << "Test 4 did not pass!" << std::endl;
        std::cout << "The class can instantiate a LT matrix object which has zero size, which is not allowed!"
                  << std::endl;
        return 1;
    }
    catch(std::exception &e){}
    
    try
    {
        LTMatrix<double> D(10);
        
        D(10, 0) = 1.0;
        
        std::cout << "Test 4 did not pass!" << std::endl;
        std::cout << "The class can access entries that are not in the matrix!"
                  << std::endl;
        return 1;
    }
    catch(std::exception &e){}
    
    
    std::cout << "All tests passed!" << std::endl;
    
    return 0;
}
