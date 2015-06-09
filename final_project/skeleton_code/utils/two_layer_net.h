#ifndef UTILS_TWO_LAYER_NET_H_
#define UTILS_TWO_LAYER_NET_H_

#include <armadillo>
#include <cmath>
#include <iostream>

extern double t_scatter;
extern double t_all_reduce;
extern double t_gpu;
extern double t_update_nn;
extern double t_batch;
extern double t_print;
extern double t_auto_test;

class TwoLayerNet
{
    public:
	const int num_layers = 2;
	// H[i] is the number of neurons in layer i (where i=0 implies input layer)
	std::vector<int> H;
	// Weights of the neural network
	// W[i] are the weights of the i^th layer
	std::vector<arma::mat> W;
	// Biases of the neural network
	// b[i] is the row vector biases of the i^th layer
	std::vector<arma::rowvec> b;
    
	TwoLayerNet (std::vector<int> _H) {
	    W.resize (num_layers);
	    b.resize (num_layers);
	    H = _H;
    
	    for (int i = 0; i < num_layers; i++) {
		arma::arma_rng::set_seed(arma::arma_rng::seed_type(i));
		W[i] = 0.0001 * arma::randn (H[i+1], H[i]);
		b[i].zeros(H[i+1]);
	    }
	}
};

/*
 * Do the feedforward by using CPU and the Armadillo library
 */
void feedforward (TwoLayerNet &nn,
		  const arma::mat& X,
		  struct cache& bpcache);

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
double loss (TwoLayerNet &nn,
	     const arma::mat& yc,
	     const arma::mat& y,
	     double reg);

/*
 * Do the backpropagation by using CPU and the Armadillo library
 */
void backprop (TwoLayerNet &nn,
	       const arma::mat& y,
	       double reg,
	       const struct cache& bpcache,
	       struct grads& bpgrads);

/* 
 * Computes the numerical gradient
 */
void numgrad (TwoLayerNet &nn,
	      const arma::mat& X,
	      const arma::mat& y,
	      double reg,
	      struct grads& numgrads);

/*
 * Train the neural network &nn
 */
void train (std::vector<TwoLayerNet> &nn,
            const arma::mat& X,
            const arma::mat& y,
            double learning_rate,
            double reg,
            const int epochs,
            const int batch_size,
            bool grad_check,
            int print_every);

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict (TwoLayerNet &nn,
	      const arma::mat& X,
	      arma::mat& label);

/*
 * Do the backpropagation by using the GPU GEMM algorithm
 * In this version, only GEMM is done on GPU
 */
void gpu_backprop_1 (TwoLayerNet &nn,
                     const arma::mat& y,
                     double reg,
                     const struct cache& bpcache,
                     struct grads& bpgrads);

/*
 * Do the backpropagation by using the GPU GEMM algorithm
 * In this version, the whole backpropagation process is done
 * on GPU to minimize communication cost between GEMM's
 */
void gpu_backprop_2 (TwoLayerNet &nn,
                     const arma::mat& y,
                     double reg,
                     const struct cache& bpcache,
                     struct grads& bpgrads);

/*
 * Do the feedwoard and backpropagation entirely on GPU
 * Since this function combines both the feedforward and
 * backpropagation algorithm, some communication cost over
 * the the PCI express is saved such as the cost of transfering
 * the data of sub-matrix X, matrices W1 and W2
 * the second GEMM algorithm is used
 */
void gpu_feedforward_backprop_1 (TwoLayerNet &nn,
                                 const arma::mat& X,
                                 const arma::mat& y,
                                 double reg,
                                 struct cache& cache,
                                 struct grads& bpgrads);

/*
 * Do the feedwoard and backpropagation entirely on GPU
 * Compared to gpu_accel_feedforward_backprop_1, this function
 * further minimizes the communication cost. Redundant communication
 * such as transferring back data of z1, a1, z2 from GPU
 * Also, the third GEMM algorithm, which is faster, is used
 */
void gpu_feedforward_backprop_2 (TwoLayerNet &nn,
                                 arma::mat& X,
                                 arma::mat& y,
                                 double reg,
                                 struct cache& cache,
                                 struct grads& bpgrads);

/*
 * The first version of parallel_train
 * Train the neural network &nn of rank 0 in parallel. MPI is implemented
 * The feedforward and backprop algorithm are separated
 * GEMM_1 is used
 */
void parallel_train_1 (std::vector<TwoLayerNet> &nn,
                       const arma::mat& X,
                       const arma::mat& y,
                       double learning_rate,
                       double reg, 
                       const int epochs,
                       const int batch_size,
                       bool grad_check,
                       int print_every);

/*
 * The second version of parallel_train
 * Train the neural network &nn of rank 0 in parallel. MPI is implemented
 * The feedforward and backprop algorithm (gpu_accel_feedforward_backprop_1)
 * are combined
 * GEMM_2 is used
 */
void parallel_train_2 (std::vector<TwoLayerNet> &nn,
                       const arma::mat& X,
                       const arma::mat& y,
                       double learning_rate,
                       double reg, 
                       const int epochs,
                       const int batch_size,
                       bool grad_check,
                       int print_every);

/*
 * The thrid version of parallel_train
 * Train the neural network &nn of rank 0 in parallel. MPI is implemented
 * The feedforward and backprop algorithm (gpu_accel_feedforward_backprop_2)
 * are combined with minimum communication
 * Also, scattering of X is done outside the iteration
 * GEMM_3 is used
 */
void parallel_train_3 (std::vector<TwoLayerNet> &nn,
                       const arma::mat& X,
                       const arma::mat& y,
                       double learning_rate,
                       double reg, 
                       const int epochs,
                       const int batch_size,
                       bool grad_check,
                       int print_every);

/*
 * The fourth version of parallel_train
 * Train the neural network &nn of rank 0 in parallel. MPI is implemented
 * The feedforward and backprop algorithm (gpu_accel_feedforward_backprop_2)
 * are combined with minimum communication
 * Also, scattering of X is done outside the iteration
 * GEMM_3 is used
 */
void parallel_train_4 (std::vector<TwoLayerNet> &nn,
                       const arma::mat& X,
                       const arma::mat& y,
                       double learning_rate,
                       double reg, 
                       const int epochs,
                       const int batch_size,
                       bool grad_check,
                       int print_every);

#endif