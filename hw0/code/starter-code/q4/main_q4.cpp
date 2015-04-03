#include <iostream>
#include <random>
#include <set>

/* function to return the number of points in the range [lb, ub] */
int count_in_range(std::set<double> &data, double lb, double ub)
{
    std::set<double>::iterator itlow = data.lower_bound(lb);
    std::set<double>::iterator itup = data.upper_bound(ub);
    
    int num_of_pt_in_range = std::distance(itlow, itup);
    
    return num_of_pt_in_range;
}

int main()
{
    std::set<double> data;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (unsigned int i = 0; i < 1000; ++i)
    {
        data.insert(distribution(generator));
    }
    
    double lb = 2.0;
    double ub = 10.0;
    
    int num_of_pt_in_range = count_in_range(data, lb, ub);
    
    std::cout << "Number of points in the range ["
              << lb
              << ", "
              << ub
              << "] out of "
              << 1000
              << " points: "
              << num_of_pt_in_range
              << std::endl;
    
    return 0;
}