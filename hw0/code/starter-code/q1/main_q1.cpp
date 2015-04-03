#include <cstdlib>

/* function to swap two arrays of the same size */
void swap(double **a, double **b)
{
    double *temp = *a;
    *a = *b;
    *b = temp;
}

int main()
{
    double *a = NULL;
    double *b = NULL;
    
    a = (double *)malloc(1000000*sizeof(double));
    b = (double *)malloc(1000000*sizeof(double));
    
    swap(&a, &b);
    
    free(a);
    a = NULL;
    
    free(b);
    b = NULL;
    
    return 0;
}
