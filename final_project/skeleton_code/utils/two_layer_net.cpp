#include "two_layer_net.h"

#include <armadillo>
#include "common.h"
#include "../gpu_func.h"
#include "mpi.h"

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

double norms (TwoLayerNet &nn) {
      double norm_sum = 0;

      for (int i = 0; i < nn.num_layers; ++i)  {
        norm_sum += arma::accu (arma::square (nn.W[i]));
      }

      return norm_sum;
}

void feedforward (TwoLayerNet &nn, const arma::mat& X, struct cache& cache)
{
  cache.z.resize(2);
  cache.a.resize(2);

  // std::cout << W[0].n_rows << "\n";
  assert (X.n_cols == nn.W[0].n_cols);
  cache.X = X;
  int N = X.n_rows;

  arma::mat z1 = X * nn.W[0].t() + arma::repmat(nn.b[0], N, 1);
  cache.z[0] = z1;

  // std::cout << "Computing a1 " << "\n";
  arma::mat a1;
  sigmoid (z1, a1);
  cache.a[0] = a1;

  // std::cout << "Computing z2 " << "\n";
  assert (a1.n_cols == nn.W[1].n_cols);
  arma::mat z2 = a1 * nn.W[1].t() + arma::repmat(nn.b[1], N, 1);
  cache.z[1] = z2;

  // std::cout << "Computing a2 " << "\n";
  arma::mat a2;
  softmax (z2, a2);
  cache.a[1] = cache.yc = a2;
}

void gpu_feedforward(TwoLayerNet &nn, const arma::mat& X, struct cache& cache)
{
    cache.z.resize(2);
    cache.a.resize(2);
    
    // std::cout << W[0].n_rows << "\n";
    assert (X.n_cols == nn.W[0].n_cols);
    cache.X = X;
    int N = X.n_rows;
    
    double* mat_X = cache.X.memptr();
    arma::mat W1_t = nn.W[0].t();
    double* mat_W1_t = W1_t.memptr();
    arma::mat b1 = arma::repmat(nn.b[0], N, 1);
    double* mat_b1 = b1.memptr();
    arma::mat z1(X.n_rows, W1_t.n_cols);
    
    gpu_GEMM_1(1.0, 1.0, mat_X, mat_W1_t, mat_b1, z1.memptr(), X.n_rows, X.n_cols, W1_t.n_cols);
    //arma::mat z1 = X * nn.W[0].t() + arma::repmat(nn.b[0], N, 1);
    cache.z[0] = z1;
    
    // std::cout << "Computing a1 " << "\n";
    arma::mat a1;
    sigmoid (z1, a1);
    cache.a[0] = a1;
    
    // std::cout << "Computing z2 " << "\n";
    assert (a1.n_cols == nn.W[1].n_cols);
    
    double* mat_a1 = a1.memptr();
    arma::mat W2_t = nn.W[1].t();
    double *mat_W2_t = W2_t.memptr();
    arma::mat b2 = arma::repmat(nn.b[1], N, 1);
    double *mat_b2 = b2.memptr();
    arma::mat z2(a1.n_rows, W2_t.n_cols);
    
    gpu_GEMM_1(1.0, 1.0, mat_a1, mat_W2_t, mat_b2, z2.memptr(), a1.n_rows, a1.n_cols, W2_t.n_cols);
    //arma::mat z2 = a1 * nn.W[1].t() + arma::repmat(nn.b[1], N, 1);
    cache.z[1] = z2;
    
    // std::cout << "Computing a2 " << "\n";
    arma::mat a2;
    softmax (z2, a2);
    cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : N x C one-hot row vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop (TwoLayerNet &nn, const arma::mat& y, double reg, const struct cache& bpcache, struct grads& bpgrads)
{
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_rows;

  // std::cout << "backprop " << bpcache.yc << "\n";
  arma::mat diff = (1.0 / N) * (bpcache.yc - y);
  bpgrads.dW[1] = diff.t() * bpcache.a[0] + reg * nn.W[1];
  bpgrads.db[1] = arma::sum (diff, 0);
  arma::mat da1 = diff * nn.W[1];

  arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  bpgrads.dW[0] = dz1.t() * bpcache.X + reg * nn.W[0];
  bpgrads.db[0] = arma::sum(dz1, 0);
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : N x C one-hot row vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void gpu_backprop(TwoLayerNet &nn, const arma::mat& y, double reg, const struct cache& bpcache, struct grads& bpgrads)
{
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_rows;

  // std::cout << "backprop " << bpcache.yc << "\n";
  arma::mat diff = (1.0 / N) * (bpcache.yc - y);
  
  arma::mat diff_t = diff.t();
  double* mat_diff_t = diff_t.memptr();
  const double* mat_a1 = bpcache.a[0].memptr();
  double* mat_W2 = nn.W[1].memptr();
  bpgrads.dW[1].set_size(nn.W[1].n_rows, nn.W[1].n_cols);
  double* mat_dW2 = bpgrads.dW[1].memptr();
  gpu_GEMM_1(1.0, reg, mat_diff_t, mat_a1, mat_W2, mat_dW2, diff_t.n_rows, diff_t.n_cols, bpcache.a[0].n_cols);
  //bpgrads.dW[1] = diff.t() * bpcache.a[0] + reg * nn.W[1];
  
  bpgrads.db[1] = arma::sum (diff, 0);
  
  arma::mat da1(diff.n_rows, nn.W[1].n_cols);
  double* mat_da1 = da1.memptr();
  double* mat_diff = diff.memptr();
  gpu_GEMM_1(1.0, 0.0, mat_diff, mat_W2, mat_da1, mat_da1, diff.n_rows, diff.n_cols, nn.W[1].n_cols);
  //arma::mat da1 = diff * nn.W[1];

  arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  arma::mat dz1_t = dz1.t();
  double* mat_dz1_t = dz1_t.memptr();
  const double* mat_X = bpcache.X.memptr();
  double* mat_W1 = nn.W[0].memptr();
  bpgrads.dW[0].set_size(nn.W[0].n_rows, nn.W[0].n_cols);
  double* mat_dW1 = bpgrads.dW[0].memptr();
  gpu_GEMM_1(1.0, reg, mat_dz1_t, mat_X, mat_W1, mat_dW1, dz1_t.n_rows, dz1_t.n_cols, bpcache.X.n_cols);
  //bpgrads.dW[0] = dz1.t() * bpcache.X + reg * nn.W[0];
  
  bpgrads.db[0] = arma::sum(dz1, 0);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
double loss (TwoLayerNet &nn, const arma::mat& yc, const arma::mat& y, double reg)
{
  int N = yc.n_rows;
  double ce_sum = -arma::accu (arma::log (yc.elem (arma::find (y == 1))));

  double data_loss = ce_sum / N;
  double reg_loss = 0.5 * reg * norms(nn);
  double loss = data_loss + reg_loss;
  // std::cout << "Loss: " << loss << "\n";
  return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict (TwoLayerNet &nn, const arma::mat& X, arma::mat& label)
{
  struct cache fcache;
  feedforward (nn, X, fcache);
  label.set_size (X.n_rows);

  for (int i = 0; i < X.n_rows; ++i) {
    arma::uword row, col;
    fcache.yc.row(i).max (row, col);
    label(i) = col;
  }
}

/* 
 * Computes the numerical gradient
 */
void numgrad (TwoLayerNet &nn, const arma::mat& X, const arma::mat& y, double reg, struct grads& numgrads)
{
  double h = 0.00001;
  struct cache numcache;
  numgrads.dW.resize(nn.num_layers);
  numgrads.db.resize(nn.num_layers);

  for (int i = 0; i < nn.num_layers; ++i) {
    numgrads.dW[i].resize (nn.W[i].n_rows, nn.W[i].n_cols);
    for (int j = 0; j < nn.W[i].n_rows; ++j) {
      for (int k = 0; k < nn.W[i].n_cols; ++k) {
        double oldval = nn.W[i](j,k);
        nn.W[i](j, k) = oldval + h;
        feedforward (nn, X, numcache);
        double fxph = loss (nn, numcache.yc, y, reg);
        nn.W[i](j, k) = oldval - h;
        feedforward (nn, X, numcache);
        double fxnh = loss (nn, numcache.yc, y, reg);
        numgrads.dW[i](j, k) = (fxph - fxnh) / (2*h);
        nn.W[i](j, k) = oldval;
      }
    }
  }

   for (int i = 0; i < nn.num_layers; ++i) {
    numgrads.db[i].resize (nn.b[i].n_rows, nn.b[i].n_cols);
    for (int j = 0; j < nn.b[i].size(); ++j) {
      double oldval = nn.b[i](j);
      nn.b[i](j) = oldval + h;
      feedforward (nn, X, numcache);
      double fxph = loss (nn, numcache.yc, y, reg);
      nn.b[i](j) = oldval - h;
      feedforward (nn, X, numcache);
      double fxnh = loss (nn, numcache.yc, y, reg);
      numgrads.db[i](j) = (fxph - fxnh) / (2*h);
      nn.b[i](j) = oldval;
    }
  }
}

/*
 * Train the neural network &nn
 */
void train (TwoLayerNet &nn, const arma::mat& X, const arma::mat& y, double learning_rate, double reg, 
    const int epochs, const int batch_size, bool grad_check, int print_every)
{
  int N = X.n_rows;
  int iter = 0;

  for (int epoch = 0 ; epoch < epochs; ++epoch) {
    std::cout << "At epoch " << epoch << std::endl;
    
    int num_batches = (int) ceil ( N / (float) batch_size);    

    for (int batch = 0; batch < num_batches; ++batch) {
      int last_row = std::min ((batch + 1)*batch_size, N-1);
      arma::mat X_batch = X.rows (batch * batch_size, last_row);
      arma::mat y_batch = y.rows (batch * batch_size, last_row);

      struct cache bpcache;
      feedforward (nn, X_batch, bpcache);
      
      struct grads bpgrads;
      backprop (nn, y_batch, reg, bpcache, bpgrads);

      if (print_every > 0 && iter % print_every == 0) {
       if (grad_check) {
          struct grads numgrads;
          numgrad (nn, X_batch, y_batch, reg, numgrads);
          assert (gradcheck (numgrads, bpgrads));
        }
        std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" << epochs << " = " << loss (nn, bpcache.yc, y_batch, reg) << "\n";
      }

      // Gradient descent step
      for (int i = 0; i < nn.W.size(); ++i) {
        nn.W[i] -= learning_rate * bpgrads.dW[i];
      }

      for (int i = 0; i < nn.b.size(); ++i) {
        nn.b[i] -= learning_rate * bpgrads.db[i];
      }

      iter++;
    }    
  }    
}

/*
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation 
 * should mainly be in this function.
 */
void parallel_train (TwoLayerNet &nn,
                     const arma::mat& X,
                     const arma::mat& y,
                     double learning_rate,
                     double reg, 
                     const int epochs,
                     const int batch_size,
                     bool grad_check,
                     int print_every)
{
    /*
     * Possible Implementation:
     * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
     * 2. compute each sub-batch of images' contribution to network coefficient updates
     * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
     * 4. update local network coefficient at each node
     */
    
    /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
       for storing elements in a column major way. Or you can allocate your own array
       memory space and store the elements in a row major way. Remember to update the
       Armadillo matrices in TwoLayerNet &nn of rank 0 before returning from the function. */
    
    int rank, num_procs;
    MPI_SAFE_CALL (MPI_Comm_size (MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL (MPI_Comm_rank (MPI_COMM_WORLD, &rank));
    
    int X_num_rows = (rank == 0)? X.n_rows : 0;
    int X_num_cols = (rank == 0)? X.n_cols : 0;
    int y_num_cols = (rank == 0)? y.n_cols : 0;
    
    // Broadcast the number of rows of X, number of columns of y and number of columns of y
    // to all nodes from rank 0
    MPI_SAFE_CALL (MPI_Bcast (&X_num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL (MPI_Bcast (&X_num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL (MPI_Bcast (&y_num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD));
    
    /* Subdivide input into batches and send the batches to each MPI node from rank 0 */
    
    // Compute the number of batches in each MPI process
    int batches_per_proc = (int) ceil (X_num_rows / (double) num_procs);
    
    int X_process_num_rows = std::min(batches_per_proc, X_num_rows - rank*batches_per_proc);

    const double* mat_X = X.memptr();
    arma::mat X_process(batches_per_proc, X_num_cols);
    double* mat_X_process = X_process.memptr();
    
    for (int i = 0; i < X_num_cols; i++)
    {
        MPI_SAFE_CALL(
            MPI_Scatter(mat_X + i*X_num_rows,
                        batches_per_proc,
                        MPI_DOUBLE,
                        mat_X_process + i*batches_per_proc,
                        batches_per_proc,
                        MPI_DOUBLE,
                        0,
                        MPI_COMM_WORLD));
    }
    
    X_process.resize(X_process_num_rows, X_num_cols);
    
    /* Subdivide the one-hot vectors into batches and send the batches to each MPI node from rank 0 */
    
    int y_process_num_rows = X_process_num_rows;
    
    const double* mat_y = y.memptr();
    arma::mat y_process(batches_per_proc, y_num_cols);
    double* mat_y_process = y_process.memptr();
    
    for (int i = 0; i < y_num_cols; i++)
    {
        MPI_SAFE_CALL(
            MPI_Scatter(mat_y + i*X_num_rows,
                        batches_per_proc,
                        MPI_DOUBLE,
                        mat_y_process + i*batches_per_proc,
                        batches_per_proc,
                        MPI_DOUBLE,
                        0,
                        MPI_COMM_WORLD));
    }
    
    y_process.resize(y_process_num_rows, y_num_cols);
    
    int iter = 0.0;
    
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::cout << "At epoch " << epoch << std::endl;
        
        int num_sub_batches = (batches_per_proc + batch_size - 1)/batch_size;
        for (int sub_batch = 0; sub_batch < num_sub_batches; sub_batch++)
        {      
            struct grads bpgrads_global_sum;
            bpgrads_global_sum.dW.resize(nn.W.size());
            bpgrads_global_sum.db.resize(nn.b.size());
            
            for (int i = 0; i < nn.W.size(); i++)
            {
                bpgrads_global_sum.dW[i].zeros(nn.W[i].n_rows, nn.W[i].n_cols);
            }
            
            for (int i = 0; i < nn.b.size(); i++)
            {
                bpgrads_global_sum.db[i].zeros(nn.b[i].n_rows, nn.b[i].n_cols);
            }
            
            struct grads bpgrads;
            
            if (sub_batch*batch_size < X_process.n_rows)
            {
                int last_row = std::min((sub_batch + 1)*batch_size - 1, X_process_num_rows - 1);
                arma::mat X_sub_batch = X_process.rows (sub_batch*batch_size, last_row);
                arma::mat y_sub_batch = y_process.rows (sub_batch*batch_size, last_row);
                
                struct cache bpcache;
                
                gpu_feedforward (nn, X_sub_batch, bpcache);
                //feedforward (nn, X_sub_batch, bpcache);
                
                gpu_backprop (nn, y_sub_batch, reg, bpcache, bpgrads);
                //backprop (nn, y_sub_batch, reg, bpcache, bpgrads);
                
                if (print_every > 0 && iter % print_every == 0)
                {
                    if (grad_check)
                    {
                        struct grads numgrads;
                        numgrad (nn, X_sub_batch, y_sub_batch, reg, numgrads);
                        assert (gradcheck (numgrads, bpgrads));
                    }
                    std::cout << "Loss at iteration "
                              << iter
                              << " of epoch "
                              << epoch
                              << "/"
                              << epochs
                              << " of rank "
                              << rank
                              << " = "
                              << loss(nn, bpcache.yc, y_sub_batch, reg)
                              << "\n";
                }
            }
            else
            {
                for (int i = 0; i < nn.W.size(); i++)
                {
                    bpgrads.dW[i].zeros(nn.W[i].n_rows, nn.W[i].n_cols);
                }
                
                for (int i = 0; i < nn.b.size(); i++)
                {
                    bpgrads.db[i].zeros(nn.b[i].n_rows, nn.b[i].n_cols);
                }
            }
            
            // Sum up dW and dB from all process by using MPI_Allreduce
            
            for (int i = 0; i < nn.W.size(); i++)
            {
                MPI_SAFE_CALL(
                    MPI_Allreduce(bpgrads.dW[i].memptr(),
                                  bpgrads_global_sum.dW[i].memptr(),
                                  bpgrads.dW[i].n_rows*bpgrads.dW[i].n_cols,
                                  MPI_DOUBLE,
                                  MPI_SUM,
                                  MPI_COMM_WORLD));
            }
            
            for (int i = 0; i < nn.b.size(); i++)
            {
                MPI_SAFE_CALL(
                    MPI_Allreduce(bpgrads.db[i].memptr(),
                                  bpgrads_global_sum.db[i].memptr(),
                                  bpgrads.db[i].n_rows*bpgrads.db[i].n_cols,
                                  MPI_DOUBLE,
                                  MPI_SUM,
                                  MPI_COMM_WORLD));
            }
            
            // Gradient descent step
            for (int i = 0; i < nn.W.size(); i++)
            {
                nn.W[i] -= learning_rate * bpgrads_global_sum.dW[i];
            }
            
            for (int i = 0; i < nn.b.size(); i++)
            {
                nn.b[i] -= learning_rate * bpgrads_global_sum.db[i];
            }
            
            iter++;
        }
    }
}

/*
void parallel_train (TwoLayerNet &nn, const arma::mat& X, const arma::mat& y, double learning_rate, double reg, 
    const int epochs, const int batch_size, bool grad_check, int print_every)
{
  int rank, num_procs;
  MPI_SAFE_CALL (MPI_Comm_size (MPI_COMM_WORLD, &num_procs));
  MPI_SAFE_CALL (MPI_Comm_rank (MPI_COMM_WORLD, &rank));

  int N = (rank == 0)?X.n_rows:0;
  MPI_SAFE_CALL (MPI_Bcast (&N, 1, MPI_INT, 0, MPI_COMM_WORLD));
  
  for (int epoch = 0; epoch < epochs; ++epoch) {
    int num_batches = (N + batch_size - 1)/batch_size;
    for (int batch = 0; batch < num_batches; ++batch) {
    }
  }
}
*/
