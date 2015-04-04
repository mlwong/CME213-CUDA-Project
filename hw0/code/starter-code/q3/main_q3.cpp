#include <iostream>
#include <memory>
#include <string>
#include <vector>

/* TODO: Make Matrix a pure abstract class with the 
 * public method:
 *     std::string repr()
 */
class Matrix
{
  public:
    virtual std::string repr() = 0;
};

/* TODO: Modify the following function so that it 
 * inherits from the Matrix class */
class SparseMatrix: public Matrix
{
 public:
  std::string repr()
  {
    return "sparse";
  }
};

/* TODO: Modify the following function so that it 
 * inherits from the Matrix class */
class ToeplitzMatrix: public Matrix
{
 public:
  std::string repr()
  {
    return "toeplitz";
  }
};

/* TODO: This function should accept a vector of Matrices and call the repr 
 * function on each matrix, printing the result to standard out. 
 */
void PrintRepr(std::vector<std::shared_ptr<Matrix>> &matrices)
{
  for (auto &M: matrices)
  {
    std::cout << M->repr() << std::endl;
  }
};

/* TODO: Your main method should fill a vector with an instance of SparseMatrix
 * and an instance of ToeplitzMatrix and pass the resulting vector
 * to the PrintRepr function.
 */ 
int main()
{
  /* fill a vector with an instance of SparseMatrix
   *  and an instance of ToeplitzMatrix by using
   *  smart pointers
   */
  std::vector<std::shared_ptr<Matrix>> matrices;
  
  matrices.push_back(std::make_shared<SparseMatrix>());
  matrices.push_back(std::make_shared<ToeplitzMatrix>());
  
  /* pass the vector to the PrintRepr function */
  PrintRepr(matrices);
  
  return 0;
}
