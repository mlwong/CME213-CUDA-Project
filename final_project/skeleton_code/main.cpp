#include <iostream>
#include <iomanip> 
#include <cassert>
#include <mpi.h>
#include <armadillo>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <unistd.h>

#include "utils/mnist.h"
#include "utils/two_layer_net.h"
#include "utils/test_utils.h"
#include "utils/common.h"
#include "gpu_func.h"

#define FILE_TRAIN_IMAGES "data/train-images-idx3-ubyte"
#define FILE_TRAIN_LABELS "data/train-labels-idx1-ubyte"
#define FILE_TEST_IMAGES "data/t10k-images-idx3-ubyte"
#define FILE_TEST_OUTPUT "test_output.out"
#define NUM_TRAIN 60000
#define IMAGE_SIZE 784  // 28 x 28
#define NUM_CLASSES 10
#define NUM_TEST 10000

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

// #define UNIT_TEST

int main (int argc, char *argv[]) {
	
	double t_reading_and_computing_start = 0.0;
	double t_reading_and_computing_end = 0.0;
	double t_reading_and_computing = 0.0;
	
	// initialize
	int num_procs = 0, rank = 0;
	MPI_Init (&argc, &argv);
	MPI_SAFE_CALL (MPI_Comm_size (MPI_COMM_WORLD, &num_procs));
	MPI_SAFE_CALL (MPI_Comm_rank (MPI_COMM_WORLD, &rank));

	// assign a GPU device to each MPI proc
	int nDevices;
	cudaGetDeviceCount (&nDevices);
	if (nDevices < num_procs)
	{
		std::cerr << "Please allocate at least as many GPUs as\
			the number of MPI procs." << std::endl;
	}
	checkCudaErrors (cudaSetDevice(rank));
	
#ifdef UNIT_TEST
	// Test whether the GPU GEMM functions are implemented correctly
	if (rank == 0)
	{
		std::cout << "Testing gpu_GEMM_0(): " << std::endl;
		
		// Test 1 on the gpu_GEMM_0 function
		test_gpu_GEMM_0a(80, 70, 90);
		
		// Test 2 on the gpu_GEMM_0 function
		test_gpu_GEMM_0a(250, 400, 150);
		
		// Test 3 on the gpu_GEMM_0 function
		test_gpu_GEMM_0b(80, 70, 90);
		
		// Test 4 on the gpu_GEMM_0 function
		test_gpu_GEMM_0b(250, 400, 150);
		
		// Test 5 on the gpu_GEMM_0 function
		test_gpu_GEMM_0c(80, 70, 90);
		
		// Test 6 on the gpu_GEMM_0 function
		test_gpu_GEMM_0c(250, 400, 150);
		
		// Test 7 on the gpu_GEMM_0 function
		test_gpu_GEMM_0d(80, 70, 90);
		
		// Test 8 on the gpu_GEMM_0 function
		test_gpu_GEMM_0d(250, 400, 150);	
		
		std::cout << std::endl;
		
		std::cout << "Testing gpu_GEMM_1(): " << std::endl;
		
		// Test 1 on the gpu_GEMM_1 function
		test_gpu_GEMM_1a(80, 70, 90);
		
		// Test 2 on the gpu_GEMM_1 function
		test_gpu_GEMM_1a(250, 400, 150);
		
		// Test 3 on the gpu_GEMM_1 function
		test_gpu_GEMM_1b(80, 70, 90);
		
		// Test 4 on the gpu_GEMM_1 function
		test_gpu_GEMM_1b(250, 400, 150);
		
		// Test 5 on the gpu_GEMM_1 function
		test_gpu_GEMM_1c(80, 70, 90);
		
		// Test 6 on the gpu_GEMM_1 function
		test_gpu_GEMM_1c(250, 400, 150);
		
		// Test 7 on the gpu_GEMM_1 function
		test_gpu_GEMM_1d(80, 70, 90);
		
		// Test 8 on the gpu_GEMM_1 function
		test_gpu_GEMM_1d(250, 400, 150);
		
		std::cout << std::endl;
		
		std::cout << "Testing gpu_GEMM_2(): " << std::endl;
		
		// Test 1 on the gpu_GEMM_2 function
		test_gpu_GEMM_2a(80, 70, 90);
		
		// Test 2 on the gpu_GEMM_2 function
		test_gpu_GEMM_2a(250, 400, 150);
		
		// Test 3 on the gpu_GEMM_2 function
		test_gpu_GEMM_2b(80, 70, 90);
		
		// Test 4 on the gpu_GEMM_2 function
		test_gpu_GEMM_2b(250, 400, 150);
		
		// Test 5 on the gpu_GEMM_2 function
		test_gpu_GEMM_2c(80, 70, 90);
		
		// Test 6 on the gpu_GEMM_2 function
		test_gpu_GEMM_2c(250, 400, 150);
		
		// Test 7 on the gpu_GEMM_2 function
		test_gpu_GEMM_2d(80, 70, 90);
		
		// Test 8 on the gpu_GEMM_2 function
		test_gpu_GEMM_2d(250, 400, 150);
		
		std::cout << std::endl;
		
		std::cout << "Testing gpu_GEMM_3(): " << std::endl;
		
		// Test 1 on the gpu_GEMM_3 function
		test_gpu_GEMM_3a(80, 70, 90);
		
		// Test 2 on the gpu_GEMM_3 function
		test_gpu_GEMM_3a(250, 400, 150);
		
		// Test 3 on the gpu_GEMM_3 function
		test_gpu_GEMM_3b(80, 70, 90);
		
		// Test 4 on the gpu_GEMM_3 function
		test_gpu_GEMM_3b(250, 400, 150);
		
		// Test 5 on the gpu_GEMM_3 function
		test_gpu_GEMM_3c(80, 70, 90);
		
		// Test 6 on the gpu_GEMM_3 function
		test_gpu_GEMM_3c(250, 400, 150);
		
		// Test 7 on the gpu_GEMM_3 function
		test_gpu_GEMM_3d(80, 70, 90);
		
		// Test 8 on the gpu_GEMM_3 function
		test_gpu_GEMM_3d(250, 400, 150);
		
		std::cout << std::endl;
	}
#endif
	
#ifdef UNIT_TEST
	if (rank == 0)
	{
		std::cout << "Testing gpu_sigmoid(): " << std::endl;
		test_gpu_sigmoid(1000, 1000);
		std::cout << std::endl;
		
		std::cout << "Testing gpu_softmax(): " << std::endl;
		test_gpu_softmax(1000, 1000);
		std::cout << std::endl;
		
		std::cout << "Testing gpu_sum_col(): " << std::endl;
		test_gpu_sum_col(1000, 1000);
		std::cout << std::endl;
		
		std::cout << "Testing gpu_elementwise_mult(): " << std::endl;
		test_gpu_elementwise_mult(1000, 1000);
		std::cout << std::endl;
		
		std::cout << "Testing gpu_elementwise_subtract(): " << std::endl;
		test_gpu_elementwise_subtract(1000, 1000);
		std::cout << std::endl;
		
		std::cout << "Testing gpu_compute_diff(): " << std::endl;
		test_gpu_compute_diff(1000, 1000);
		std::cout << std::endl;
		
		std::cout << "Testing gpu_transpose(): " << std::endl;
		test_gpu_transpose(1000, 1000);
		std::cout << std::endl;
	}
#endif
	
#ifdef UNIT_TEST
	if (rank == 0)
	{
		int m, n, l;
		m = 1000;
		n = 1000;
		l = 1000;
		std::cout << "Speed testing of different GEMM algorithms: " << std::endl;
		
		test_speed_GEMM(m, n, l);
		std::cout << std::endl;
	}
#endif

	// reads in options
	std::vector<int> H(3);
	double reg = 1e-4;
	double learning_rate = 0.05;
	int num_epochs = 20;
	int batch_size = 800;
	int num_neuron = 1000;
	int run_seq = 0;

	int option = 0;
	while ((option = getopt(argc, argv, "n:r:l:e:b:s")) != -1) {
		switch (option) {
			case 'n': num_neuron = atoi(optarg); break;
			case 'r': reg = atof(optarg); break;
			case 'l': learning_rate = atof(optarg); break;
			case 'e': num_epochs = atoi(optarg); break;
			case 'b': batch_size = atoi(optarg); break;
			case 's': run_seq = 1; break;
		}
	}

	H[0] = IMAGE_SIZE;
	H[1] = num_neuron;
	H[2] = NUM_CLASSES;

	arma::mat x_train, y_train, label_train, x_dev, y_dev, label_dev, x_test;
	TwoLayerNet nn (H);

	if (rank == 0) {
		std::cout << "num_neuron=" << num_neuron << ", reg=" << reg << ", learning_rate=" << learning_rate
				  << ", num_epochs=" << num_epochs << ", batch_size=" << batch_size << ", num_procs=" << num_procs << std::endl;
	    // Read MNIST images into Armadillo mat vector
	    arma::mat x (NUM_TRAIN, IMAGE_SIZE);
	    // label_train contains the prediction for each 
	    arma::colvec label = arma::zeros<arma::colvec>(NUM_TRAIN);
	    // y_train is the matrix of one-hot label vectors where only y[c] = 1, 
	    // where c is the right class.
	    arma::mat y = arma::zeros<arma::mat>(NUM_TRAIN, NUM_CLASSES);
		
		std::cout << "Loading training data..." << std::endl;
		
		t_reading_and_computing_start = MPI_Wtime();
		
	    read_mnist (FILE_TRAIN_IMAGES, x);
	    read_mnist_label (FILE_TRAIN_LABELS, label);
	    label_to_y (label, NUM_CLASSES, y);

	    /* Print stats of training data */
	    std::cout << "Training data stats..." << std::endl;
	    std::cout << "Size of x_train, N =  " << x.n_rows << std::endl;
	    std::cout << "Size of label_train = " << label.size() << std::endl;

	    assert (x.n_rows == NUM_TRAIN && x.n_cols == IMAGE_SIZE);
	    assert (label.size() == NUM_TRAIN);

	    /* Split into train set and dev set, you should use train set to train your
	       neural network and dev set to evaluate its precision */
	    int dev_size = (int) (0.1 * NUM_TRAIN);
	    x_train = x.rows (0, NUM_TRAIN-dev_size);
	    y_train = y.rows (0, NUM_TRAIN-dev_size);
	    label_train = label.rows (0, NUM_TRAIN-dev_size);

	    x_dev = x.rows (NUM_TRAIN-dev_size, NUM_TRAIN - 1);
	    y_dev = y.rows (NUM_TRAIN-dev_size, NUM_TRAIN - 1);
	    label_dev = label.rows (NUM_TRAIN-dev_size, NUM_TRAIN - 1);

	    /* Load the test data, we will compare the prediction of your trained neural 
	       network with test data label to evalute its precision */
	    x_test = arma::zeros (NUM_TEST, IMAGE_SIZE);
	    read_mnist (FILE_TEST_IMAGES, x_test);
		
		t_reading_and_computing_end = MPI_Wtime();
		
		t_reading_and_computing += (t_reading_and_computing_end -
								    t_reading_and_computing_start);
	}

	std::vector<TwoLayerNet> seq_nn;
	if ((rank == 0) && (run_seq)) {
		seq_nn.push_back(nn);
    	std::cout << "Start Sequential Training" << std::endl;
    	double start = MPI_Wtime();
    	train (seq_nn, x_train, y_train, learning_rate, reg, num_epochs, batch_size, false, -1);
    	double end = MPI_Wtime();
    	std::cout << "Time for Sequential Training: " << end - start << " seconds" << std::endl;
    	arma::vec label_dev_pred;
    	predict (seq_nn.back(), x_dev, label_dev_pred);
    	double prec = precision (label_dev_pred, label_dev);
    	std::cout << std::fixed << std::setprecision(10) << "Precision on dev set for sequential training = " << prec << std::endl;
    }
    
	std::vector<TwoLayerNet> par_nn;
	// if (!run_seq)
	{
		/* Train the Neural Network in Parallel*/
		if (rank == 0) 
			std::cout << "Start Parallel Training" << std::endl;
		double start = MPI_Wtime();
		t_reading_and_computing_start = MPI_Wtime();
		
		par_nn.push_back(nn);
		
		/* ---- Parallel Training ---- */
		parallel_train_3 (par_nn, x_train, y_train, learning_rate, reg, num_epochs, batch_size, false, -1);
		
		double end = MPI_Wtime();
		t_reading_and_computing_end = MPI_Wtime();
		t_reading_and_computing += (t_reading_and_computing_end -
								    t_reading_and_computing_start);
		if (rank == 0)
		{
			std::cout << "Time for Parallel Training (excluding loading data): " << end - start << " seconds" << std::endl;
			std::cout << "Time for Parallel Training (including loading data): " << t_reading_and_computing << " seconds" << std::endl;
			std::cout << "Time for scattering: " << std::endl
			          << t_scatter << " seconds" << std::endl;
			std::cout << "Time for all-reducing: " << std::endl
			          << t_all_reduce << " seconds" << std::endl;
			std::cout << "Time for GPU computing (including allocating data on CPU and transfering data over PCI express: " << std::endl
			          << t_gpu << " seconds" << std::endl;
			std::cout << "Time for updating neural network: " << std::endl
			          << t_update_nn << " seconds" << std::endl;
			std::cout << "Time for memory allocating and copying for sub-batching on CPU: " << std::endl
			           << t_batch << " seconds" << std::endl;
			std::cout << "Time for displaying: " << std::endl
			           << t_print << " seconds" << std::endl;
			std::cout << "Time for storing data in auto-testing: " << std::endl
			           << t_auto_test << " seconds" << std::endl;
		}
	
		/* Make sure after training process, rank 0's neural network is up to date */
		if (rank == 0) {
			arma::vec label_dev_pred;
			predict (par_nn.back(), x_dev, label_dev_pred);
			double prec = precision (label_dev_pred, label_dev);
			std::cout << std::fixed << std::setprecision(10) << "Precision on dev set for parallel training = " << prec << std::endl;
			arma::vec label_test_pred;
			predict (par_nn.back(), x_test, label_test_pred);
			save_label (FILE_TEST_OUTPUT, label_test_pred);
		}
		
	}
	
	/*
	 * Auto-testing of code. W and b of both sequential and paralle runs are compared
	 */
	
	if ((rank == 0) && (run_seq))
	{
		std::cout << "'epoch'  ";
		for (int i = 0; i < par_nn[0].W.size(); i++)
		{
			std::cout << "'L2 error of W" << i+1 << "' ";
		}
		for (int i = 0; i < par_nn[0].b.size(); i++)
		{
			std::cout << "'L2 error of b" << i+1 << "' ";
		}
		std::cout << std::endl;
		
		for (int n = 1; n < par_nn.size(); n++)
		{
			std::cout << n << " ";
			for (int i = 0; i < par_nn[n].W.size(); i++)
			{
				arma::mat W_diff = par_nn[n].W[i] - seq_nn[n].W[i];
				W_diff = abs(W_diff);
				double error = norm(W_diff);
				
				std::cout << error << " ";
			}
			
			for (int i = 0; i < par_nn[n].b.size(); i++)
			{
				arma::mat b_diff = par_nn[n].b[i] - seq_nn[n].b[i];
				b_diff = abs(b_diff);
				double error = norm(b_diff);
				
				std::cout << error << " ";
			}
			std::cout << std::endl;
		}
	}
	
    MPI_Finalize ();
    return 0;
}
