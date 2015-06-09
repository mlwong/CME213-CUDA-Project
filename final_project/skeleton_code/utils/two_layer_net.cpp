#include "two_layer_net.h"

#include <armadillo>
#include "common.h"
#include "mpi.h"
#include "../gpu_func.h"

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

double t_scatter;
double t_all_reduce;
double t_gpu;
double t_update_nn;
double t_batch;
double t_print;
double t_auto_test;

double norms (TwoLayerNet &nn)
{
    double norm_sum = 0;
    
    for (int i = 0; i < nn.num_layers; ++i)
    {
        norm_sum += arma::accu (arma::square (nn.W[i]));
    }
    
    return norm_sum;
}

/*
 * Do the feedforward by using CPU and the Armadillo library
 */
void feedforward (TwoLayerNet &nn,
                  const arma::mat& X,
                  struct cache& cache)
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

/*
 * Do the feedforward by using the GPU GEMM algorithm
 * In this version, only GEMM is done on GPU
 */
void gpu_feedforward_1 (TwoLayerNet &nn,
                        const arma::mat& X,
                        struct cache& cache)
{
    cache.z.resize(2);
    cache.a.resize(2);
    
    // std::cout << W[0].n_rows << "\n";
    assert (X.n_cols == nn.W[0].n_cols);
    cache.X = X;
    
    int N = X.n_rows;
    
    arma::mat W1_t = nn.W[0].t();
    arma::mat W2_t = nn.W[1].t();
    arma::mat b1 = arma::repmat(nn.b[0], N, 1);
    arma::mat b2 = arma::repmat(nn.b[1], N, 1);
    arma::mat z1(X.n_rows, W1_t.n_cols);
    arma::mat a1(z1.n_rows, z1.n_cols);
    arma::mat z2(a1.n_rows, W2_t.n_cols);
    arma::mat a2(z2.n_rows, z2.n_cols);
    
    double* mat_X = cache.X.memptr();
    double* mat_W1_t = W1_t.memptr();
    double* mat_b1 = b1.memptr();
    
    gpu_GEMM_1(1.0, 1.0, mat_X, mat_W1_t, mat_b1, z1.memptr(), X.n_rows, X.n_cols, W1_t.n_cols, false, false);
    // arma::mat z1 = X * nn.W[0].t() + arma::repmat(nn.b[0], N, 1);
    
    // std::cout << "Computing a1 " << "\n";
    sigmoid (z1, a1);

    // std::cout << "Computing z2 " << "\n";
    assert (a1.n_cols == nn.W[1].n_cols);
    
    double* mat_a1 = a1.memptr();
    double *mat_W2_t = W2_t.memptr();
    double *mat_b2 = b2.memptr();
    
    gpu_GEMM_1(1.0, 1.0, mat_a1, mat_W2_t, mat_b2, z2.memptr(), a1.n_rows, a1.n_cols, W2_t.n_cols, false, false);
    // arma::mat z2 = a1 * nn.W[1].t() + arma::repmat(nn.b[1], N, 1);
    
    // std::cout << "Computing a2 " << "\n";
    softmax (z2, a2);
    
    cache.z[0] = z1;
    cache.z[1] = z2;
    cache.a[0] = a1;
    cache.a[1] = cache.yc = a2;
}

/*
 * Do the feedforward by using the GPU GEMM algorithm
 * In this version, the whole feedforward process is done
 * on GPU to minimize communication cost between GEMM's
 */
void gpu_feedforward_2 (TwoLayerNet &nn,
                        const arma::mat& X,
                        struct cache& cache)
{    
    cache.z.resize(2);
    cache.a.resize(2);
    
    // std::cout << W[0].n_rows << "\n";
    assert (X.n_cols == nn.W[0].n_cols);
    cache.X = X;
    int N = X.n_rows;
    
    arma::mat b1 = arma::repmat(nn.b[0], N, 1);
    arma::mat z1(X.n_rows, nn.W[0].n_rows);
    arma::mat a1(z1.n_rows, z1.n_cols);
    
    arma::mat b2 = arma::repmat(nn.b[1], N, 1);
    arma::mat z2(a1.n_rows, nn.W[1].n_rows);
    arma::mat a2(z2.n_rows, z2.n_cols);
    
    gpu_accel_feedforward (cache.X.memptr(), cache.X.n_rows, cache.X.n_cols,
                           nn.W[0].memptr(), nn.W[0].n_rows, nn.W[0].n_cols,
                           b1.memptr(), b1.n_rows, b1.n_cols,
                           z1.memptr(), z1.n_rows, z1.n_cols,
                           a1.memptr(), a1.n_rows, a1.n_cols,
                           nn.W[1].memptr(), nn.W[1].n_rows, nn.W[1].n_cols,
                           b2.memptr(), b2.n_rows, b2.n_cols,
                           z2.memptr(), z2.n_rows, z2.n_cols,
                           a2.memptr(), a2.n_rows, a2.n_cols);
                           
    cache.z[0] = z1;
    cache.a[0] = a1;
    cache.z[1] = z2;
    cache.a[1] = cache.yc = a2;
}

/*
 * Do the backpropagation by using CPU and the Armadillo library
 * 
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : N x C one-hot row vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop (TwoLayerNet &nn,
               const arma::mat& y,
               double reg,
               const struct cache& bpcache,
               struct grads& bpgrads)
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
 * Do the backpropagation by using the GPU GEMM algorithm
 * In this version, only GEMM is done on GPU
 */
void gpu_backprop_1 (TwoLayerNet &nn,
                     const arma::mat& y,
                     double reg,
                     const struct cache& bpcache,
                     struct grads& bpgrads)
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
    gpu_GEMM_1(1.0, reg, mat_diff_t, mat_a1, mat_W2, mat_dW2, diff_t.n_rows, diff_t.n_cols, bpcache.a[0].n_cols, false, false);
    // bpgrads.dW[1] = diff.t() * bpcache.a[0] + reg * nn.W[1];
    
    bpgrads.db[1] = arma::sum (diff, 0);
    
    arma::mat da1(diff.n_rows, nn.W[1].n_cols);
    double* mat_da1 = da1.memptr();
    double* mat_diff = diff.memptr();
    gpu_GEMM_1(1.0, 0.0, mat_diff, mat_W2, mat_da1, mat_da1, diff.n_rows, diff.n_cols, nn.W[1].n_cols, false, false);
    //arma::mat da1 = diff * nn.W[1];
    
    arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);
    
    arma::mat dz1_t = dz1.t();
    double* mat_dz1_t = dz1_t.memptr();
    const double* mat_X = bpcache.X.memptr();
    double* mat_W1 = nn.W[0].memptr();
    bpgrads.dW[0].set_size(nn.W[0].n_rows, nn.W[0].n_cols);
    double* mat_dW1 = bpgrads.dW[0].memptr();
    gpu_GEMM_1(1.0, reg, mat_dz1_t, mat_X, mat_W1, mat_dW1, dz1_t.n_rows, dz1_t.n_cols, bpcache.X.n_cols, false, false);
    // bpgrads.dW[0] = dz1.t() * bpcache.X + reg * nn.W[0];
    
    bpgrads.db[0] = arma::sum(dz1, 0);
}

/*
 * Do the backpropagation by using the GPU GEMM algorithm
 * In this version, the whole backpropagation process is done
 * on GPU to minimize communication cost between GEMM's
 */
void gpu_backprop_2 (TwoLayerNet &nn,
                     const arma::mat& y,
                     double reg,
                     const struct cache& bpcache,
                     struct grads& bpgrads)
{
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    int N = y.n_rows;
    
    // std::cout << "backprop " << bpcache.yc << "\n";
    arma::mat diff = (1.0 / N) * (bpcache.yc - y);
    
    bpgrads.dW[0].set_size(nn.W[0].n_rows, nn.W[0].n_cols);
    bpgrads.dW[1].set_size(nn.W[1].n_rows, nn.W[1].n_cols);
    
    bpgrads.db[0].set_size(nn.W[1].n_cols);
    bpgrads.db[1].set_size(diff.n_cols);
    
    
    gpu_accel_backprop (reg,
                        diff.memptr(), diff.n_rows, diff.n_cols,
                        bpcache.X.memptr(), bpcache.X.n_rows, bpcache.X.n_cols,
                        bpcache.a[0].memptr(), bpcache.a[0].n_rows, bpcache.a[0].n_cols,
                        nn.W[0].memptr(), nn.W[0].n_rows, nn.W[0].n_cols,
                        nn.W[1].memptr(), nn.W[1].n_rows, nn.W[1].n_cols,
                        bpgrads.dW[0].memptr(), bpgrads.dW[0].n_rows, bpgrads.dW[0].n_cols,
                        bpgrads.dW[1].memptr(), bpgrads.dW[1].n_rows, bpgrads.dW[1].n_cols,
                        bpgrads.db[0].memptr(), bpgrads.db[0].n_cols,
                        bpgrads.db[1].memptr(), bpgrads.db[1].n_cols);
}

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
                                 struct grads& bpgrads)
{
    cache.z.resize(2);
    cache.a.resize(2);
    
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    
    // std::cout << W[0].n_rows << "\n";
    assert (X.n_cols == nn.W[0].n_cols);
    cache.X = X;
    int N = X.n_rows;
    
    arma::mat b1 = arma::repmat(nn.b[0], N, 1);
    arma::mat z1(X.n_rows, nn.W[0].n_rows);
    arma::mat a1(z1.n_rows, z1.n_cols);
    
    arma::mat b2 = arma::repmat(nn.b[1], N, 1);
    arma::mat z2(a1.n_rows, nn.W[1].n_rows);
    arma::mat a2(z2.n_rows, z2.n_cols);
    
    bpgrads.dW[0].set_size(nn.W[0].n_rows, nn.W[0].n_cols);
    bpgrads.dW[1].set_size(nn.W[1].n_rows, nn.W[1].n_cols);
    
    bpgrads.db[0].set_size(nn.W[1].n_cols);
    bpgrads.db[1].set_size(y.n_cols);
    
    gpu_accel_feedforward_backprop_1 (reg,
                                      cache.X.memptr(), cache.X.n_rows, cache.X.n_cols,
                                      y.memptr(), y.n_rows, y.n_cols,
                                      nn.W[0].memptr(), nn.W[0].n_rows, nn.W[0].n_cols,
                                      b1.memptr(), b1.n_rows, b1.n_cols,
                                      z1.memptr(), z1.n_rows, z1.n_cols,
                                      a1.memptr(), a1.n_rows, a1.n_cols,
                                      nn.W[1].memptr(), nn.W[1].n_rows, nn.W[1].n_cols,
                                      b2.memptr(), b2.n_rows, b2.n_cols,
                                      z2.memptr(), z2.n_rows, z2.n_cols,
                                      a2.memptr(), a2.n_rows, a2.n_cols,
                                      bpgrads.dW[0].memptr(), bpgrads.dW[0].n_rows, bpgrads.dW[0].n_cols,
                                      bpgrads.dW[1].memptr(), bpgrads.dW[1].n_rows, bpgrads.dW[1].n_cols,
                                      bpgrads.db[0].memptr(), bpgrads.db[0].n_cols,
                                      bpgrads.db[1].memptr(), bpgrads.db[1].n_cols);
                           
    cache.z[0] = z1;
    cache.a[0] = a1;
    cache.z[1] = z2;
    cache.a[1] = cache.yc = a2;
}

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
                                 struct grads& bpgrads)
{
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    
    // std::cout << W[0].n_rows << "\n";
    assert (X.n_cols == nn.W[0].n_cols);
    int N = X.n_rows;
    
    arma::mat b1 = arma::repmat(nn.b[0], N, 1);
    
    arma::mat b2 = arma::repmat(nn.b[1], N, 1);
    arma::mat a2(X.n_rows, nn.W[1].n_rows);
    
    bpgrads.dW[0].set_size(nn.W[0].n_rows, nn.W[0].n_cols);
    bpgrads.dW[1].set_size(nn.W[1].n_rows, nn.W[1].n_cols);
    
    bpgrads.db[0].set_size(nn.W[1].n_cols);
    bpgrads.db[1].set_size(y.n_cols);
    
    gpu_accel_feedforward_backprop_2 (reg,
                                      X.memptr(), X.n_rows, X.n_cols,
                                      y.memptr(), y.n_rows, y.n_cols,
                                      nn.W[0].memptr(), nn.W[0].n_rows, nn.W[0].n_cols,
                                      b1.memptr(), b1.n_rows, b1.n_cols,
                                      nn.W[1].memptr(), nn.W[1].n_rows, nn.W[1].n_cols,
                                      b2.memptr(), b2.n_rows, b2.n_cols,
                                      a2.memptr(), a2.n_rows, a2.n_cols,
                                      bpgrads.dW[0].memptr(), bpgrads.dW[0].n_rows, bpgrads.dW[0].n_cols,
                                      bpgrads.dW[1].memptr(), bpgrads.dW[1].n_rows, bpgrads.dW[1].n_cols,
                                      bpgrads.db[0].memptr(), bpgrads.db[0].n_cols,
                                      bpgrads.db[1].memptr(), bpgrads.db[1].n_cols);
                           

    cache.yc = a2;
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
double loss (TwoLayerNet &nn,
             const arma::mat& yc,
             const arma::mat& y,
             double reg)
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
void predict (TwoLayerNet &nn,
              const arma::mat& X,
              arma::mat& label)
{
    struct cache fcache;
    feedforward (nn, X, fcache);
    label.set_size (X.n_rows);
    
    for (int i = 0; i < X.n_rows; ++i)
    {
        arma::uword row, col;
        fcache.yc.row(i).max (row, col);
        label(i) = col;
    }
}

/* 
 * Computes the numerical gradient
 */
void numgrad (TwoLayerNet &nn,
              const arma::mat& X,
              const arma::mat& y,
              double reg,
              struct grads& numgrads)
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
void train (std::vector<TwoLayerNet> &nn,
            const arma::mat& X,
            const arma::mat& y,
            double learning_rate,
            double reg,
            const int epochs,
            const int batch_size,
            bool grad_check,
            int print_every)
{
    int N = X.n_rows;
    int iter = 0;
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::cout << "At epoch " << epoch << std::endl;
        
        int num_batches = (int) ceil ( N / (float) batch_size);
        
        for (int batch = 0; batch < num_batches; ++batch)
        {
            int last_row = std::min ((batch + 1)*batch_size-1, N-1);
            arma::mat X_batch = X.rows (batch * batch_size, last_row);
            arma::mat y_batch = y.rows (batch * batch_size, last_row);
          
            struct cache bpcache;
            struct grads bpgrads;
            
            feedforward (nn[0], X_batch, bpcache);
            backprop (nn[0], y_batch, reg, bpcache, bpgrads);
          
            if (print_every > 0 && iter % print_every == 0)
            {
                if (grad_check)
                {
                    struct grads numgrads;
                    numgrad (nn[0], X_batch, y_batch, reg, numgrads);
                    assert (gradcheck (numgrads, bpgrads));
                }
                std::cout << "Loss at iter "
                          << iter
                          << " and batch "
                          << batch
                          << " of epoch "
                          << epoch
                          << "/"
                          << epochs-1
                          << " = "
                          << loss (nn[0], bpcache.yc, y_batch, reg)
                          << "\n";
            }
            
            // Gradient descent step
            for (int i = 0; i < nn[0].W.size(); ++i)
            {
                nn[0].W[i] -= learning_rate * bpgrads.dW[i];
            }
          
            for (int i = 0; i < nn[0].b.size(); ++i)
            {
                nn[0].b[i] -= learning_rate * bpgrads.db[i];
            }
            
            iter++;
        }
        
        nn.push_back(nn[0]);
    }
}

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
                       int print_every)
{   
    /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
       for storing elements in a column major way. Or you can allocate your own array
       memory space and store the elements in a row major way. Remember to update the
       Armadillo matrices in TwoLayerNet &nn of rank 0 before returning from the function. */
    
    t_scatter    = 0.0;
    t_all_reduce = 0.0;
    t_gpu        = 0.0;
    t_update_nn  = 0.0;
    t_batch      = 0.0;
    t_print      = 0.0;
    t_auto_test  = 0.0;
    
    double t_start, t_end;
    
    int rank, num_procs;
    MPI_SAFE_CALL (MPI_Comm_size (MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL (MPI_Comm_rank (MPI_COMM_WORLD, &rank));
    
    int X_num_rows = (rank == 0)? X.n_rows : 0;
    int X_num_cols = (rank == 0)? X.n_cols : 0;
    int y_num_cols = (rank == 0)? y.n_cols : 0;
    
    // Broadcast the number of rows of X, number of columns of X and number of columns of y
    // to all nodes from rank 0
    MPI_SAFE_CALL (MPI_Bcast (&X_num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL (MPI_Bcast (&X_num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL (MPI_Bcast (&y_num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD));
    
    int iter = 0;
    
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        t_start = MPI_Wtime();
        
        if (rank == 0)
        {
            std::cout << "At epoch " << epoch << std::endl;
        }
        
        t_end = MPI_Wtime();
        
        t_print += (t_end - t_start);
        
        int num_batches = (int) ceil ( X_num_rows / (float) batch_size);
        
        if (num_procs > 1)
        {
            for (int batch = 0; batch < num_batches; batch++)
            {
                /*
                 * Possible Implementation:
                 * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
                 * 2. compute each sub-batch of images' contribution to network coefficient updates
                 * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
                 * 4. update local network coefficient at each node
                 */
                
                t_start = MPI_Wtime();
                
                int last_row = std::min ((batch + 1)*batch_size-1, X_num_rows-1);
                
                arma::mat X_batch(last_row - batch*batch_size + 1, X_num_cols);
                arma::mat y_batch(last_row - batch*batch_size + 1, y_num_cols);
                
                arma::mat X_sub_batch;
                arma::mat y_sub_batch;
                
                t_end = MPI_Wtime();
                    
                t_batch += (t_end - t_start);
                
                if (rank == 0)
                {
                    X_batch = X.rows (batch * batch_size, last_row);
                    y_batch = y.rows (batch * batch_size, last_row);
                }
                
                t_start = MPI_Wtime();
                
                /* Subdivide batches into sub-batches and send the sub-batches to each MPI node from rank 0 */
                
                // Compute the number of inputs (number of rows of X) for each MPI process
                int sub_batch_size = (int) ceil (X_batch.n_rows / (float) num_procs);
                
                int X_sub_batch_num_rows = std::min(sub_batch_size, (int) X_batch.n_rows - rank*sub_batch_size);
                
                const double* mat_X_batch = X_batch.memptr();
                X_sub_batch.set_size(sub_batch_size, X_num_cols);
                double* mat_X_sub_batch = X_sub_batch.memptr();
                
                t_end = MPI_Wtime();
                
                t_batch += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                for (int i = 0; i < X_num_cols; i++)
                {
                    MPI_SAFE_CALL(
                        MPI_Scatter(mat_X_batch + i*X_batch.n_rows,
                                    sub_batch_size,
                                    MPI_DOUBLE,
                                    mat_X_sub_batch + i*sub_batch_size,
                                    sub_batch_size,
                                    MPI_DOUBLE,
                                    0,
                                    MPI_COMM_WORLD));
                }
                
                t_end = MPI_Wtime();
                
                t_scatter += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                X_sub_batch.resize(X_sub_batch_num_rows, X_num_cols);
                
                /* Subdivide the one-hot vectors into sub-batches and send the sub-batches to each MPI node from rank 0 */
                
                int y_sub_batch_num_rows = X_sub_batch_num_rows;
                
                const double* mat_y_batch = y_batch.memptr();
                y_sub_batch.set_size(sub_batch_size, y_num_cols);
                double* mat_y_sub_batch = y_sub_batch.memptr();
                
                t_end = MPI_Wtime();
                
                t_batch += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                for (int i = 0; i < y_num_cols; i++)
                {
                    MPI_SAFE_CALL(
                        MPI_Scatter(mat_y_batch + i*y_batch.n_rows,
                                    sub_batch_size,
                                    MPI_DOUBLE,
                                    mat_y_sub_batch + i*sub_batch_size,
                                    sub_batch_size,
                                    MPI_DOUBLE,
                                    0,
                                    MPI_COMM_WORLD));
                }
                
                t_end = MPI_Wtime();
                
                t_scatter += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                y_sub_batch.resize(y_sub_batch_num_rows, y_num_cols);
                
                t_end = MPI_Wtime();
            
                t_batch += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                struct grads bpgrads_global_sum;
                bpgrads_global_sum.dW.resize(nn[0].W.size());
                bpgrads_global_sum.db.resize(nn[0].b.size());
                
                for (int i = 0; i < nn[0].W.size(); i++)
                {
                    bpgrads_global_sum.dW[i].zeros(nn[0].W[i].n_rows, nn[0].W[i].n_cols);
                }
                
                for (int i = 0; i < nn[0].b.size(); i++)
                {
                    bpgrads_global_sum.db[i].zeros(nn[0].b[i].n_rows, nn[0].b[i].n_cols);
                }
                
                t_end = MPI_Wtime();
                
                t_batch += (t_end - t_start);
                
                struct grads bpgrads;
                        
                struct cache bpcache;
                
                t_start = MPI_Wtime();
                
                gpu_feedforward_1 (nn[0], X_sub_batch, bpcache);
                
                gpu_backprop_1 (nn[0], y_sub_batch, reg, bpcache, bpgrads);
                
                t_end = MPI_Wtime();
                
                t_gpu += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                if (print_every > 0 && iter % print_every == 0)
                {
                    if (grad_check)
                    {
                        struct grads numgrads;
                        numgrad (nn[0], X_batch, y_sub_batch, reg, numgrads);
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
                              << loss(nn[0], bpcache.yc, y_sub_batch, reg)
                              << "\n";
                }
                
                t_end = MPI_Wtime();
                
                t_print = t_end - t_start;
                
                // Sum up dW and dB from all processes by using MPI_Allreduce
                
                t_start = MPI_Wtime();
                
                for (int i = 0; i < nn[0].W.size(); i++)
                {
                    MPI_SAFE_CALL(
                        MPI_Allreduce(bpgrads.dW[i].memptr(),
                                      bpgrads_global_sum.dW[i].memptr(),
                                      bpgrads.dW[i].n_rows*bpgrads.dW[i].n_cols,
                                      MPI_DOUBLE,
                                      MPI_SUM,
                                      MPI_COMM_WORLD));
                }
                
                for (int i = 0; i < nn[0].b.size(); i++)
                {
                    MPI_SAFE_CALL(
                        MPI_Allreduce(bpgrads.db[i].memptr(),
                                      bpgrads_global_sum.db[i].memptr(),
                                      bpgrads.db[i].n_rows*bpgrads.db[i].n_cols,
                                      MPI_DOUBLE,
                                      MPI_SUM,
                                      MPI_COMM_WORLD));
                }
                
                t_end = MPI_Wtime();
                
                t_all_reduce += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                // Gradient descent step on the local neutral network of all the nodes
                for (int i = 0; i < nn[0].W.size(); i++)
                {
                    nn[0].W[i] -= learning_rate / num_procs * bpgrads_global_sum.dW[i];
                }
                
                for (int i = 0; i < nn[0].b.size(); i++)
                {
                    nn[0].b[i] -= learning_rate / num_procs * bpgrads_global_sum.db[i];
                }
                
                t_end = MPI_Wtime();
                
                t_update_nn += (t_end - t_start);
                
                iter++;
            }
        }
        else
        {
            for (int batch = 0; batch < num_batches; ++batch)
            {
                t_start = MPI_Wtime();
                
                int last_row = std::min ((batch + 1)*batch_size-1, X_num_rows-1);
                arma::mat X_batch = X.rows (batch * batch_size, last_row);
                arma::mat y_batch = y.rows (batch * batch_size, last_row);
                
                t_end = MPI_Wtime();
                
                t_batch += (t_end - t_start);
                
                struct cache bpcache;
                struct grads bpgrads;
                
                t_start = MPI_Wtime();
                
                gpu_feedforward_1 (nn[0], X_batch, bpcache);
                gpu_backprop_1 (nn[0], y_batch, reg, bpcache, bpgrads);
                
                t_end = MPI_Wtime();
                
                t_gpu += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                if (print_every > 0 && iter % print_every == 0)
                {
                    if (grad_check)
                    {
                        struct grads numgrads;
                        numgrad (nn[0], X_batch, y_batch, reg, numgrads);
                        assert (gradcheck (numgrads, bpgrads));
                    }
                    std::cout << "Loss at iter "
                              << iter
                              << " and batch "
                              << batch
                              << " of epoch "
                              << epoch
                              << "/"
                              << epochs-1
                              << " = "
                              << loss (nn[iter], bpcache.yc, y_batch, reg)
                              << "\n";
                }
                
                t_end = MPI_Wtime();
                
                t_print += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                // Gradient descent step
                for (int i = 0; i < nn[0].W.size(); ++i)
                {
                    nn[0].W[i] -= learning_rate * bpgrads.dW[i];
                }
              
                for (int i = 0; i < nn[0].b.size(); ++i)
                {
                    nn[0].b[i] -= learning_rate * bpgrads.db[i];
                }
                
                t_end = MPI_Wtime();
                
                t_update_nn += (t_end - t_start);
                
                iter++;
            }
        }
        
        /*
         * Record W and B at every epoch
         */
        
        t_start = MPI_Wtime();
        
        nn.push_back(nn[0]);
        
        t_end = MPI_Wtime();
        
        t_auto_test += (t_end - t_start);  
    }
}

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
                       int print_every)
{   
    /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
       for storing elements in a column major way. Or you can allocate your own array
       memory space and store the elements in a row major way. Remember to update the
       Armadillo matrices in TwoLayerNet &nn of rank 0 before returning from the function. */
    
    t_scatter    = 0.0;
    t_all_reduce = 0.0;
    t_gpu        = 0.0;
    t_update_nn  = 0.0;
    t_batch      = 0.0;
    t_print      = 0.0;
    t_auto_test  = 0.0;
    
    double t_start, t_end;
    
    int rank, num_procs;
    MPI_SAFE_CALL (MPI_Comm_size (MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL (MPI_Comm_rank (MPI_COMM_WORLD, &rank));
    
    int X_num_rows = (rank == 0)? X.n_rows : 0;
    int X_num_cols = (rank == 0)? X.n_cols : 0;
    int y_num_cols = (rank == 0)? y.n_cols : 0;
    
    // Broadcast the number of rows of X, number of columns of X and number of columns of y
    // to all nodes from rank 0
    MPI_SAFE_CALL (MPI_Bcast (&X_num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL (MPI_Bcast (&X_num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL (MPI_Bcast (&y_num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD));
    
    int iter = 0;
    
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        t_start = MPI_Wtime();
        
        if (rank == 0)
        {
            std::cout << "At epoch " << epoch << std::endl;
        }
        
        t_end = MPI_Wtime();
        
        t_print += (t_end - t_start);
        
        int num_batches = (int) ceil ( X_num_rows / (float) batch_size);
        
        if (num_procs > 1)
        {
            for (int batch = 0; batch < num_batches; batch++)
            {
                /*
                 * Possible Implementation:
                 * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
                 * 2. compute each sub-batch of images' contribution to network coefficient updates
                 * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
                 * 4. update local network coefficient at each node
                 */
                
                t_start = MPI_Wtime();
                
                int last_row = std::min ((batch + 1)*batch_size-1, X_num_rows-1);
                
                arma::mat X_batch(last_row - batch*batch_size + 1, X_num_cols);
                arma::mat y_batch(last_row - batch*batch_size + 1, y_num_cols);
                
                arma::mat X_sub_batch;
                arma::mat y_sub_batch;
                
                t_end = MPI_Wtime();
                    
                t_batch += (t_end - t_start);
                
                if (rank == 0)
                {
                    X_batch = X.rows (batch * batch_size, last_row);
                    y_batch = y.rows (batch * batch_size, last_row);
                }
                
                t_start = MPI_Wtime();
                
                /* Subdivide batches into sub-batches and send the sub-batches to each MPI node from rank 0 */
                
                // Compute the number of inputs (number of rows of X) for each MPI process
                int sub_batch_size = (int) ceil (X_batch.n_rows / (float) num_procs);
                
                int X_sub_batch_num_rows = std::min(sub_batch_size, (int) X_batch.n_rows - rank*sub_batch_size);
                
                const double* mat_X_batch = X_batch.memptr();
                X_sub_batch.set_size(sub_batch_size, X_num_cols);
                double* mat_X_sub_batch = X_sub_batch.memptr();
                
                t_end = MPI_Wtime();
                
                t_batch += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                for (int i = 0; i < X_num_cols; i++)
                {
                    MPI_SAFE_CALL(
                        MPI_Scatter(mat_X_batch + i*X_batch.n_rows,
                                    sub_batch_size,
                                    MPI_DOUBLE,
                                    mat_X_sub_batch + i*sub_batch_size,
                                    sub_batch_size,
                                    MPI_DOUBLE,
                                    0,
                                    MPI_COMM_WORLD));
                }
                
                t_end = MPI_Wtime();
                
                t_scatter += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                X_sub_batch.resize(X_sub_batch_num_rows, X_num_cols);
                
                /* Subdivide the one-hot vectors into sub-batches and send the sub-batches to each MPI node from rank 0 */
                
                int y_sub_batch_num_rows = X_sub_batch_num_rows;
                
                const double* mat_y_batch = y_batch.memptr();
                y_sub_batch.set_size(sub_batch_size, y_num_cols);
                double* mat_y_sub_batch = y_sub_batch.memptr();
                
                t_end = MPI_Wtime();
                
                t_batch += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                for (int i = 0; i < y_num_cols; i++)
                {
                    MPI_SAFE_CALL(
                        MPI_Scatter(mat_y_batch + i*y_batch.n_rows,
                                    sub_batch_size,
                                    MPI_DOUBLE,
                                    mat_y_sub_batch + i*sub_batch_size,
                                    sub_batch_size,
                                    MPI_DOUBLE,
                                    0,
                                    MPI_COMM_WORLD));
                }
                
                t_end = MPI_Wtime();
                
                t_scatter += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                y_sub_batch.resize(y_sub_batch_num_rows, y_num_cols);
                
                t_end = MPI_Wtime();
            
                t_batch += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                struct grads bpgrads_global_sum;
                bpgrads_global_sum.dW.resize(nn[0].W.size());
                bpgrads_global_sum.db.resize(nn[0].b.size());
                
                for (int i = 0; i < nn[0].W.size(); i++)
                {
                    bpgrads_global_sum.dW[i].zeros(nn[0].W[i].n_rows, nn[0].W[i].n_cols);
                }
                
                for (int i = 0; i < nn[0].b.size(); i++)
                {
                    bpgrads_global_sum.db[i].zeros(nn[0].b[i].n_rows, nn[0].b[i].n_cols);
                }
                
                t_end = MPI_Wtime();
                
                t_batch += (t_end - t_start);
                
                struct grads bpgrads;
                        
                struct cache bpcache;
                
                t_start = MPI_Wtime();
                
                gpu_feedforward_backprop_1 (nn[0], X_sub_batch, y_sub_batch, reg, bpcache, bpgrads);
                
                t_end = MPI_Wtime();
                
                t_gpu += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                if (print_every > 0 && iter % print_every == 0)
                {
                    if (grad_check)
                    {
                        struct grads numgrads;
                        numgrad (nn[0], X_sub_batch, y_sub_batch, reg, numgrads);
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
                              << loss(nn[0], bpcache.yc, y_sub_batch, reg)
                              << "\n";
                }
                
                t_end = MPI_Wtime();
                
                t_print = t_end - t_start;
                
                // Sum up dW and dB from all processes by using MPI_Allreduce
                
                t_start = MPI_Wtime();
                
                for (int i = 0; i < nn[0].W.size(); i++)
                {
                    MPI_SAFE_CALL(
                        MPI_Allreduce(bpgrads.dW[i].memptr(),
                                      bpgrads_global_sum.dW[i].memptr(),
                                      bpgrads.dW[i].n_rows*bpgrads.dW[i].n_cols,
                                      MPI_DOUBLE,
                                      MPI_SUM,
                                      MPI_COMM_WORLD));
                }
                
                for (int i = 0; i < nn[0].b.size(); i++)
                {
                    MPI_SAFE_CALL(
                        MPI_Allreduce(bpgrads.db[i].memptr(),
                                      bpgrads_global_sum.db[i].memptr(),
                                      bpgrads.db[i].n_rows*bpgrads.db[i].n_cols,
                                      MPI_DOUBLE,
                                      MPI_SUM,
                                      MPI_COMM_WORLD));
                }
                
                t_end = MPI_Wtime();
                
                t_all_reduce += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                // Gradient descent step on the local neutral network of all the nodes
                for (int i = 0; i < nn[0].W.size(); i++)
                {
                    nn[0].W[i] -= learning_rate / num_procs * bpgrads_global_sum.dW[i];
                }
                
                for (int i = 0; i < nn[0].b.size(); i++)
                {
                    nn[0].b[i] -= learning_rate / num_procs * bpgrads_global_sum.db[i];
                }
                
                t_end = MPI_Wtime();
                
                t_update_nn += (t_end - t_start);
                
                iter++;
            }
        }
        else
        {
            for (int batch = 0; batch < num_batches; ++batch)
            {
                t_start = MPI_Wtime();
                
                int last_row = std::min ((batch + 1)*batch_size-1, X_num_rows-1);
                arma::mat X_batch = X.rows (batch * batch_size, last_row);
                arma::mat y_batch = y.rows (batch * batch_size, last_row);
                
                t_end = MPI_Wtime();
                
                t_batch += (t_end - t_start);
                
                struct cache bpcache;
                struct grads bpgrads;
                
                t_start = MPI_Wtime();
                
                gpu_feedforward_backprop_1 (nn[0], X_batch, y_batch, reg, bpcache, bpgrads);
                
                t_end = MPI_Wtime();
                
                t_gpu += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                if (print_every > 0 && iter % print_every == 0)
                {
                    if (grad_check)
                    {
                        struct grads numgrads;
                        numgrad (nn[0], X_batch, y_batch, reg, numgrads);
                        assert (gradcheck (numgrads, bpgrads));
                    }
                    std::cout << "Loss at iter "
                              << iter
                              << " and batch "
                              << batch
                              << " of epoch "
                              << epoch
                              << "/"
                              << epochs-1
                              << " = "
                              << loss (nn[iter], bpcache.yc, y_batch, reg)
                              << "\n";
                }
                
                t_end = MPI_Wtime();
                
                t_print += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                // Gradient descent step
                for (int i = 0; i < nn[0].W.size(); ++i)
                {
                    nn[0].W[i] -= learning_rate * bpgrads.dW[i];
                }
              
                for (int i = 0; i < nn[0].b.size(); ++i)
                {
                    nn[0].b[i] -= learning_rate * bpgrads.db[i];
                }
                
                t_end = MPI_Wtime();
                
                t_update_nn += (t_end - t_start);
                
                iter++;
            }
        }
        
        /*
         * Record W and B at every epoch
         */
        
        t_start = MPI_Wtime();
        
        nn.push_back(nn[0]);
        
        t_end = MPI_Wtime();
        
        t_auto_test += (t_end - t_start);  
    }
}

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
                       int print_every)
{    
    /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
       for storing elements in a column major way. Or you can allocate your own array
       memory space and store the elements in a row major way. Remember to update the
       Armadillo matrices in TwoLayerNet &nn of rank 0 before returning from the function. */
    
    t_scatter    = 0.0;
    t_all_reduce = 0.0;
    t_gpu        = 0.0;
    t_update_nn  = 0.0;
    t_batch      = 0.0;
    t_print      = 0.0;
    t_auto_test  = 0.0;
    
    double t_start, t_end;
    
    int rank, num_procs;
    MPI_SAFE_CALL (MPI_Comm_size (MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL (MPI_Comm_rank (MPI_COMM_WORLD, &rank));
    
    int X_num_rows = (rank == 0)? X.n_rows : 0;
    int X_num_cols = (rank == 0)? X.n_cols : 0;
    int y_num_cols = (rank == 0)? y.n_cols : 0;
    
    // Broadcast the number of rows of X, number of columns of X and number of columns of y
    // to all nodes from rank 0
    MPI_SAFE_CALL (MPI_Bcast (&X_num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL (MPI_Bcast (&X_num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL (MPI_Bcast (&y_num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD));
    
    /* Subdivide input into batches and send the batches to each MPI node from rank 0 */
    
    t_start = MPI_Wtime();
    
    // Compute the number of inputs (number of rows of X) for each MPI process
    int num_inputs_per_proc = (int) ceil (X_num_rows / (double) num_procs);
    int X_process_num_rows = std::min(num_inputs_per_proc, X_num_rows - rank*num_inputs_per_proc);

    const double* mat_X = X.memptr();
    arma::mat X_process(num_inputs_per_proc, X_num_cols);
    double* mat_X_process = X_process.memptr();
    
    t_end = MPI_Wtime();
    
    t_batch += (t_end - t_start);
    
    t_start = MPI_Wtime();
    
    for (int i = 0; i < X_num_cols; i++)
    {
        MPI_SAFE_CALL(
            MPI_Scatter(mat_X + i*X_num_rows,
                        num_inputs_per_proc,
                        MPI_DOUBLE,
                        mat_X_process + i*num_inputs_per_proc,
                        num_inputs_per_proc,
                        MPI_DOUBLE,
                        0,
                        MPI_COMM_WORLD));
    }
    
    t_end = MPI_Wtime();
    
    t_scatter += (t_end - t_start);
    
    t_start = MPI_Wtime();
    
    X_process.resize(X_process_num_rows, X_num_cols);
    
    /* Subdivide the one-hot vectors into batches and send the batches to each MPI node from rank 0 */
    
    int y_process_num_rows = X_process_num_rows;
    
    const double* mat_y = y.memptr();
    arma::mat y_process(num_inputs_per_proc, y_num_cols);
    double* mat_y_process = y_process.memptr();
    
    t_end = MPI_Wtime();
    
    t_batch += (t_end - t_start);
    
    t_start = MPI_Wtime();
    
    for (int i = 0; i < y_num_cols; i++)
    {
        MPI_SAFE_CALL(
            MPI_Scatter(mat_y + i*X_num_rows,
                        num_inputs_per_proc,
                        MPI_DOUBLE,
                        mat_y_process + i*num_inputs_per_proc,
                        num_inputs_per_proc,
                        MPI_DOUBLE,
                        0,
                        MPI_COMM_WORLD));
    }
    
    t_end = MPI_Wtime();
    
    t_scatter += (t_end - t_start);
    
    t_start = MPI_Wtime();
    
    y_process.resize(y_process_num_rows, y_num_cols);
    
    t_end = MPI_Wtime();
    
    t_batch += (t_end - t_start);
    
    int iter = 0;
    
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        t_start = MPI_Wtime();
        
        if (rank == 0)
        {
            std::cout << "At epoch " << epoch << std::endl;
        }
        
        t_end = MPI_Wtime();
        
        t_print += (t_end - t_start);
        
        int sub_batch_size = (batch_size + num_procs - 1)/num_procs;
        // int sub_batch_size = batch_size;
        
        int num_sub_batches = (num_inputs_per_proc + sub_batch_size - 1)/sub_batch_size;
        for (int sub_batch = 0; sub_batch < num_sub_batches; sub_batch++)
        {
            t_start = MPI_Wtime();
            
            struct grads bpgrads_global_sum;
            bpgrads_global_sum.dW.resize(nn[0].W.size());
            bpgrads_global_sum.db.resize(nn[0].b.size());
            
            for (int i = 0; i < nn[0].W.size(); i++)
            {
                bpgrads_global_sum.dW[i].zeros(nn[0].W[i].n_rows, nn[0].W[i].n_cols);
            }
            
            for (int i = 0; i < nn[0].b.size(); i++)
            {
                bpgrads_global_sum.db[i].zeros(nn[0].b[i].n_rows, nn[0].b[i].n_cols);
            }
            
            t_end = MPI_Wtime();
            
            t_batch += (t_end - t_start);
            
            // If the sub-batch is inside the sub-matrix X, do the GPU feedforward and backpop.
            // Otherwise, there is no contribution to dW and dB from this process
            
            struct cache bpcache;
            struct grads bpgrads;
            
            if (sub_batch*sub_batch_size < X_process.n_rows)
            {
                t_start = MPI_Wtime();
                
                int last_row = std::min((sub_batch + 1)*sub_batch_size - 1, X_process_num_rows - 1);
                arma::mat X_sub_batch = X_process.rows (sub_batch*sub_batch_size, last_row);
                arma::mat y_sub_batch = y_process.rows (sub_batch*sub_batch_size, last_row);
                
                t_end = MPI_Wtime();
                
                t_batch += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                gpu_feedforward_backprop_2 (nn[0], X_sub_batch, y_sub_batch, reg, bpcache, bpgrads);
                
                t_end = MPI_Wtime();
                
                t_gpu += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                if (print_every > 0 && iter % print_every == 0)
                {
                    if (grad_check)
                    {
                        struct grads numgrads;
                        numgrad (nn[0], X_sub_batch, y_sub_batch, reg, numgrads);
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
                              << loss(nn[0], bpcache.yc, y_sub_batch, reg)
                              << "\n";
                }
                
                t_end = MPI_Wtime();
                
                t_print += (t_end - t_start);
                
            }
            else
            {
                t_start = MPI_Wtime();
                
                for (int i = 0; i < nn[0].W.size(); i++)
                {
                    bpgrads.dW[i].zeros(nn[0].W[i].n_rows, nn[0].W[i].n_cols);
                }
                
                for (int i = 0; i < nn[0].b.size(); i++)
                {
                    bpgrads.db[i].zeros(nn[0].b[i].n_rows, nn[0].b[i].n_cols);
                }
                
                t_end = MPI_Wtime();
                
                t_update_nn += (t_end - t_start);
            }
            
            if (num_procs > 1)
            {
                // Sum up dW and dB from all processes by using MPI_Allreduce
                
                t_start = MPI_Wtime();
                
                for (int i = 0; i < nn[0].W.size(); i++)
                {
                    MPI_SAFE_CALL(
                        MPI_Allreduce(bpgrads.dW[i].memptr(),
                                      bpgrads_global_sum.dW[i].memptr(),
                                      bpgrads.dW[i].n_rows*bpgrads.dW[i].n_cols,
                                      MPI_DOUBLE,
                                      MPI_SUM,
                                      MPI_COMM_WORLD));
                }
                
                for (int i = 0; i < nn[0].b.size(); i++)
                {
                    MPI_SAFE_CALL(
                        MPI_Allreduce(bpgrads.db[i].memptr(),
                                      bpgrads_global_sum.db[i].memptr(),
                                      bpgrads.db[i].n_rows*bpgrads.db[i].n_cols,
                                      MPI_DOUBLE,
                                      MPI_SUM,
                                      MPI_COMM_WORLD));
                }
                
                t_end = MPI_Wtime();
                
                t_all_reduce += (t_end - t_start);
            
                t_start = MPI_Wtime();
                
                // Gradient descent step on the local neutral network of all the nodes
                for (int i = 0; i < nn[0].W.size(); i++)
                {
                    nn[0].W[i] -= (learning_rate / num_procs) * bpgrads_global_sum.dW[i];
                }
                
                for (int i = 0; i < nn[0].b.size(); i++)
                {
                    nn[0].b[i] -= (learning_rate / num_procs) * bpgrads_global_sum.db[i];
                }
                
                t_end = MPI_Wtime();
                
                t_update_nn += (t_end - t_start);
            }
            else
            {
                t_start = MPI_Wtime();
                
                // Gradient descent step on the local neutral network
                for (int i = 0; i < nn[0].W.size(); i++)
                {
                    nn[0].W[i] -= learning_rate * bpgrads.dW[i];
                }
                
                for (int i = 0; i < nn[0].b.size(); i++)
                {
                    nn[0].b[i] -= learning_rate  * bpgrads.db[i];
                }
                
                t_end = MPI_Wtime();
                
                t_update_nn += (t_end - t_start);
            }
            
            iter++;
        }
        
        /*
         * Record W and B at every epoch
         */
        
        t_start = MPI_Wtime();
        
        nn.push_back(nn[0]);
        
        t_end = MPI_Wtime();
        
        t_auto_test += (t_end - t_start);  
    }
}

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
                       int print_every)
{
    /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
       for storing elements in a column major way. Or you can allocate your own array
       memory space and store the elements in a row major way. Remember to update the
       Armadillo matrices in TwoLayerNet &nn of rank 0 before returning from the function. */
    
    t_scatter    = 0.0;
    t_all_reduce = 0.0;
    t_gpu        = 0.0;
    t_update_nn  = 0.0;
    t_batch      = 0.0;
    t_print      = 0.0;
    t_auto_test  = 0.0;
    
    double t_start, t_end;
    
    int rank, num_procs;
    MPI_SAFE_CALL (MPI_Comm_size (MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL (MPI_Comm_rank (MPI_COMM_WORLD, &rank));
    
    MPI_Request requestNUll;
    
    std::vector<MPI_Request> dW1_receive_requests(num_procs);
    std::vector<MPI_Request> dW2_receive_requests(num_procs);
    std::vector<MPI_Request> db1_receive_requests(num_procs);
    std::vector<MPI_Request> db2_receive_requests(num_procs);
    
    std::vector<arma::mat> dW1_receives(num_procs);
    std::vector<arma::mat> dW2_receives(num_procs);
    std::vector<arma::rowvec> db1_receives(num_procs);
    std::vector<arma::rowvec> db2_receives(num_procs);
    
    for (int r = 0; r < num_procs; r++)
    {                    
        dW1_receives[r].zeros(nn[0].W[0].n_rows, nn[0].W[0].n_cols);
        dW2_receives[r].zeros(nn[0].W[1].n_rows, nn[0].W[1].n_cols);
        db1_receives[r].zeros(nn[0].b[0].n_rows, nn[0].b[0].n_cols);
        db2_receives[r].zeros(nn[0].b[1].n_rows, nn[0].b[1].n_cols);
    }
    
    int X_num_rows = (rank == 0)? X.n_rows : 0;
    int X_num_cols = (rank == 0)? X.n_cols : 0;
    int y_num_cols = (rank == 0)? y.n_cols : 0;
    
    // Broadcast the number of rows of X, number of columns of X and number of columns of y
    // to all nodes from rank 0
    MPI_SAFE_CALL (MPI_Bcast (&X_num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL (MPI_Bcast (&X_num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL (MPI_Bcast (&y_num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD));
    
    /* Subdivide input into batches and send the batches to each MPI node from rank 0 */
    
    t_start = MPI_Wtime();
    
    // Compute the number of inputs (number of rows of X) for each MPI process
    int num_inputs_per_proc = (int) ceil (X_num_rows / (double) num_procs);
    int X_process_num_rows = std::min(num_inputs_per_proc, X_num_rows - rank*num_inputs_per_proc);

    const double* mat_X = X.memptr();
    arma::mat X_process(num_inputs_per_proc, X_num_cols);
    double* mat_X_process = X_process.memptr();
    
    t_end = MPI_Wtime();
    
    t_batch += (t_end - t_start);
    
    t_start = MPI_Wtime();
    
    for (int i = 0; i < X_num_cols; i++)
    {
        MPI_SAFE_CALL(
            MPI_Scatter(mat_X + i*X_num_rows,
                        num_inputs_per_proc,
                        MPI_DOUBLE,
                        mat_X_process + i*num_inputs_per_proc,
                        num_inputs_per_proc,
                        MPI_DOUBLE,
                        0,
                        MPI_COMM_WORLD));
    }
    
    t_end = MPI_Wtime();
    
    t_scatter += (t_end - t_start);
    
    t_start = MPI_Wtime();
    
    X_process.resize(X_process_num_rows, X_num_cols);
    
    /* Subdivide the one-hot vectors into batches and send the batches to each MPI node from rank 0 */
    
    int y_process_num_rows = X_process_num_rows;
    
    const double* mat_y = y.memptr();
    arma::mat y_process(num_inputs_per_proc, y_num_cols);
    double* mat_y_process = y_process.memptr();
    
    t_end = MPI_Wtime();
    
    t_batch += (t_end - t_start);
    
    t_start = MPI_Wtime();
    
    for (int i = 0; i < y_num_cols; i++)
    {
        MPI_SAFE_CALL(
            MPI_Scatter(mat_y + i*X_num_rows,
                        num_inputs_per_proc,
                        MPI_DOUBLE,
                        mat_y_process + i*num_inputs_per_proc,
                        num_inputs_per_proc,
                        MPI_DOUBLE,
                        0,
                        MPI_COMM_WORLD));
    }
    
    t_end = MPI_Wtime();
    
    t_scatter += (t_end - t_start);
    
    t_start = MPI_Wtime();
    
    y_process.resize(y_process_num_rows, y_num_cols);
    
    t_end = MPI_Wtime();
    
    t_batch += (t_end - t_start);
    
    int iter = 0;
    
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        t_start = MPI_Wtime();
        
        if (rank == 0)
        {
            std::cout << "At epoch " << epoch << std::endl;
        }
        
        t_end = MPI_Wtime();
        
        t_print += (t_end - t_start);
        
        int sub_batch_size = (batch_size + num_procs - 1)/num_procs;
        // int sub_batch_size = batch_size;
        
        int num_sub_batches = (num_inputs_per_proc + sub_batch_size - 1)/sub_batch_size;
        for (int sub_batch = 0; sub_batch < num_sub_batches; sub_batch++)
        {
            t_start = MPI_Wtime();
            
            t_end = MPI_Wtime();
            
            t_batch += (t_end - t_start);
            
            // If the sub-batch is inside the sub-matrix X, do the GPU feedforward and backpop.
            // Otherwise, there is no contribution to dW and dB from this process
            
            struct cache bpcache;
            struct grads bpgrads;
            
            if (sub_batch*sub_batch_size < X_process.n_rows)
            {
                t_start = MPI_Wtime();
                
                int last_row = std::min((sub_batch + 1)*sub_batch_size - 1, X_process_num_rows - 1);
                arma::mat X_sub_batch = X_process.rows (sub_batch*sub_batch_size, last_row);
                arma::mat y_sub_batch = y_process.rows (sub_batch*sub_batch_size, last_row);
                
                t_end = MPI_Wtime();
                
                t_batch += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                gpu_feedforward_backprop_2 (nn[0], X_sub_batch, y_sub_batch, reg, bpcache, bpgrads);
                
                t_end = MPI_Wtime();
                
                t_gpu += (t_end - t_start);
                
                t_start = MPI_Wtime();
                
                if (print_every > 0 && iter % print_every == 0)
                {
                    if (grad_check)
                    {
                        struct grads numgrads;
                        numgrad (nn[0], X_sub_batch, y_sub_batch, reg, numgrads);
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
                              << loss(nn[0], bpcache.yc, y_sub_batch, reg)
                              << "\n";
                }
                
                t_end = MPI_Wtime();
                
                t_print += (t_end - t_start);
                
            }
            else
            {
                t_start = MPI_Wtime();
                
                for (int i = 0; i < nn[0].W.size(); i++)
                {
                    bpgrads.dW[i].zeros(nn[0].W[i].n_rows, nn[0].W[i].n_cols);
                }
                
                for (int i = 0; i < nn[0].b.size(); i++)
                {
                    bpgrads.db[i].zeros(nn[0].b[i].n_rows, nn[0].b[i].n_cols);
                }
                
                t_end = MPI_Wtime();
                
                t_update_nn += (t_end - t_start);
            }
            
            if (num_procs > 1)
            {
                // Sum up dW and dB from all processes by using MPI_Allreduce
                
                t_start = MPI_Wtime();
                
                for (int r = 0; r < num_procs; r++)
                {
                    if (r != rank)
                    {
                        MPI_Isend(bpgrads.dW[0].memptr(),
                                  bpgrads.dW[0].n_rows*bpgrads.dW[0].n_cols,
                                  MPI_DOUBLE,
                                  r,
                                  0,
                                  MPI_COMM_WORLD,
                                  &requestNUll);
                        
                        MPI_Isend(bpgrads.dW[1].memptr(),
                                  bpgrads.dW[1].n_rows*bpgrads.dW[1].n_cols,
                                  MPI_DOUBLE,
                                  r,
                                  1,
                                  MPI_COMM_WORLD,
                                  &requestNUll);
                        
                        MPI_Isend(bpgrads.db[0].memptr(),
                                  bpgrads.db[0].n_rows*bpgrads.db[0].n_cols,
                                  MPI_DOUBLE,
                                  r,
                                  2,
                                  MPI_COMM_WORLD,
                                  &requestNUll);
                        
                        MPI_Isend(bpgrads.db[1].memptr(),
                                  bpgrads.db[1].n_rows*bpgrads.db[1].n_cols,
                                  MPI_DOUBLE,
                                  r,
                                  3,
                                  MPI_COMM_WORLD,
                                  &requestNUll);
                        
                    }
                }
                
                for (int r = 0; r < num_procs; r++)
                {
                    if (r != rank)
                    {
                        MPI_Irecv(dW1_receives[r].memptr(),
                                  bpgrads.dW[0].n_rows*bpgrads.dW[0].n_cols,
                                  MPI_DOUBLE,
                                  r,
                                  0,
                                  MPI_COMM_WORLD,
                                  &dW1_receive_requests[r]);
                        
                        MPI_Irecv(dW2_receives[r].memptr(),
                                  bpgrads.dW[1].n_rows*bpgrads.dW[1].n_cols,
                                  MPI_DOUBLE,
                                  r,
                                  1,
                                  MPI_COMM_WORLD,
                                  &dW2_receive_requests[r]);
                        
                        MPI_Irecv(db1_receives[r].memptr(),
                                  bpgrads.db[0].n_rows*bpgrads.db[0].n_cols,
                                  MPI_DOUBLE,
                                  r,
                                  2,
                                  MPI_COMM_WORLD,
                                  &db1_receive_requests[r]);
                        
                        MPI_Irecv(db2_receives[r].memptr(),
                                  bpgrads.db[1].n_rows*bpgrads.db[1].n_cols,
                                  MPI_DOUBLE,
                                  r,
                                  3,
                                  MPI_COMM_WORLD,
                                  &db2_receive_requests[r]);
                    }
                }
                
                for (int r = 0; r < num_procs; r++)
                {
                    if (r != rank)
                    {
                        MPI_Status status;
                        MPI_Wait(&(dW1_receive_requests[r]), &status);
                        MPI_Wait(&(dW2_receive_requests[r]), &status);
                        MPI_Wait(&(db1_receive_requests[r]), &status);
                        MPI_Wait(&(db2_receive_requests[r]), &status);
                    }
                }
                
                for (int r = 0; r < num_procs; r++)
                {
                    if (r != rank)
                    {
                        nn[0].W[0] -= learning_rate / num_procs * dW1_receives[r];
                        nn[0].W[1] -= learning_rate / num_procs * dW2_receives[r];
                        nn[0].b[0] -= learning_rate / num_procs * db1_receives[r];
                        nn[0].b[1] -= learning_rate / num_procs * db2_receives[r];
                    }
                }
                
                nn[0].W[0] -= learning_rate / num_procs * bpgrads.dW[0];
                nn[0].W[1] -= learning_rate / num_procs * bpgrads.dW[1];
                nn[0].b[0] -= learning_rate / num_procs * bpgrads.db[0];
                nn[0].b[1] -= learning_rate / num_procs * bpgrads.db[1];
                
                t_end = MPI_Wtime();
                
                t_all_reduce += (t_end - t_start);
            }
            else
            {
                t_start = MPI_Wtime();
                
                // Gradient descent step on the local neutral network
                for (int i = 0; i < nn[0].W.size(); i++)
                {
                    nn[0].W[i] -= learning_rate * bpgrads.dW[i];
                }
                
                for (int i = 0; i < nn[0].b.size(); i++)
                {
                    nn[0].b[i] -= learning_rate  * bpgrads.db[i];
                }
                
                t_end = MPI_Wtime();
                
                t_update_nn += (t_end - t_start);
            }
            
            iter++;
        }
        
        /*
         * Record W and B at every epoch
         */
        
        t_start = MPI_Wtime();
        
        nn.push_back(nn[0]);
        
        t_end = MPI_Wtime();
        
        t_auto_test += (t_end - t_start);  
    }
}
