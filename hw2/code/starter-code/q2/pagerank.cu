#include <algorithm> // using min() function

/* Each kernel handles the update of one pagerank score. In other 
 * words, each kernel handles one row of the update:
 * 
 *      pi(t+1) = (1/2) A pi(t) + (1/2N)
 * 
 */
__global__
void device_graph_propagate(const uint *graph_indices
                          , const uint *graph_edges
                          , const float *graph_nodes_in
                          , float *graph_nodes_out
                          , const float *inv_edges_per_node
                          , int num_nodes)
{
  // TODO: fill in the kernel code here
  
  const uint tid = blockIdx.x*blockDim.x + threadIdx.x;

  for (uint i = tid; i < num_nodes; i += gridDim.x*blockDim.x)
  {
    float sum = 0;
    
    for (uint idx = graph_indices[i]; idx < graph_indices[i + 1]; idx++)
    {
      uint j = graph_edges[idx];
      sum += inv_edges_per_node[j]*graph_nodes_in[j];
    }
    
    graph_nodes_out[i] = 0.5f*sum + 0.5f/((float) num_nodes);
  }
}

/* This function executes a specified number of iterations of the
 * pagerank algorithm. The variables are:
 *
 * h_graph_indices, h_graph_edges: 
 *     These arrays describe the indices of the neighbors of node i.
 *     Specifically, node i is adjacent to all nodes in the range
 *     h_graph_edges[h_graph_indices[i] ... h_graph_indices[i+1]].
 *     
 * h_node_values_input:
 *     An initial guess of pi(0).
 *
 * h_gpu_node_values_output:
 *     Output array for the pagerank vector.
 *
 * h_inv_edges_per_node:
 *     The i'th element in this array is the reciprocal of the 
 *     out degree of the i'th node.
 * 
 * nr_iterations:
 *     The number of iterations to run the pagerank algorithm for.
 * 
 * num_nodes: 
 *     The number of nodes in the whole graph (ie N).
 *
 * avg_edges:
 *     The average number of edges in the graph. You are guaranteed
 *     that the whole graph has num_nodes * avg_edges edges.
 * 
 */
double device_graph_iterate(const uint *h_graph_indices
                          , const uint *h_graph_edges
                          , const float *h_node_values_input
                          , float *h_gpu_node_values_output
                          , const float *h_inv_edges_per_node
                          , int nr_iterations
                          , int num_nodes
                          , int avg_edges)
{
  // TODO: allocate GPU memory
  
  uint *d_graph_indices = NULL;
  uint *d_graph_edges = NULL;
  
  // the two pagerank vectors which switch roll on every iteration
  float *d_pi_1 = NULL;
  float *d_pi_2 = NULL;
  
  float *d_inv_edges_per_node = NULL;
  
  // calculate the total number of edges
  int num_edges = num_nodes*avg_edges;
  
  int num_Malloc = 5;
  cudaError_t err[num_Malloc];
  
  for (int i = 0; i < num_Malloc; i++)
  {
   err[i] = cudaSuccess;
  }
  
  err[0] = cudaMalloc(&d_graph_indices, sizeof(uint)*(num_nodes + 1));
  err[1] = cudaMalloc(&d_graph_edges, sizeof(uint)*num_edges);
  err[2] = cudaMalloc(&d_pi_1, sizeof(float)*num_nodes);
  err[3] = cudaMalloc(&d_pi_2, sizeof(float)*num_nodes);
  err[4] = cudaMalloc(&d_inv_edges_per_node, sizeof(float)*num_nodes);
  
  // TODO: check for allocation failure
  
  if ((!d_graph_indices) || (!d_graph_edges) || (!d_pi_1) || (!d_pi_2) || (!d_inv_edges_per_node))
  {
   std::cerr << "There was allocation failure on GPU!" << std::endl;
   exit(1);
  }
  
  for (int i = 0; i < num_Malloc; i++)
  {
   if (err[i])
   {
    std::cerr << "There was allocation failure on GPU!" << std::endl;
    exit(1);
   }
  }

  // TODO: copy data to the GPU
  
  cudaMemcpy(d_graph_indices, h_graph_indices, sizeof(uint)*(num_nodes + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_graph_edges, h_graph_edges, sizeof(uint)*num_edges, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pi_1, h_node_values_input, sizeof(float)*num_nodes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_inv_edges_per_node, h_inv_edges_per_node, sizeof(float)*num_nodes, cudaMemcpyHostToDevice);
  
  start_timer(&timer);

  const int block_size = 192;
  
  // TODO: launch your kernels the appropriate number of iterations
  
  // calculate the number of blocks per grid
  uint grid_size = std::min((uint) (num_nodes + block_size - 1)/block_size, 65535u);
  
  // do the iterations
  for (int n = 0; n < nr_iterations/2; n++)
  {
   device_graph_propagate<<<grid_size, block_size>>>(d_graph_indices,
                                                     d_graph_edges,
                                                     d_pi_1,
                                                     d_pi_2,
                                                     d_inv_edges_per_node,
                                                     num_nodes);
   
   // switch the role of the two pagerank vectors
   device_graph_propagate<<<grid_size, block_size>>>(d_graph_indices,
                                                     d_graph_edges,
                                                     d_pi_2,
                                                     d_pi_1,
                                                     d_inv_edges_per_node,
                                                     num_nodes);
  }
  
  // if we have odd number of iterations, we have to do one more iteration
  if (nr_iterations%2 == 1)
  {
   device_graph_propagate<<<grid_size, block_size>>>(d_graph_indices,
                                                     d_graph_edges,
                                                     d_pi_1,
                                                     d_pi_2,
                                                     d_inv_edges_per_node,
                                                     num_nodes);
  }
  
  check_launch("gpu graph propagate");
  double gpu_elapsed_time = stop_timer(&timer);

  // TODO: copy final data back to the host for correctness checking
  
  if (nr_iterations%2 == 0)
  {
   cudaMemcpy(h_gpu_node_values_output, d_pi_1, sizeof(float)*num_nodes, cudaMemcpyDeviceToHost);
  }
  else
  {
   cudaMemcpy(h_gpu_node_values_output, d_pi_2, sizeof(float)*num_nodes, cudaMemcpyDeviceToHost);
  }
  
  // TODO: free the memory you allocated!
  
  cudaFree(d_graph_indices);
  cudaFree(d_graph_edges);
  cudaFree(d_pi_1);
  cudaFree(d_pi_2);
  cudaFree(d_inv_edges_per_node); 
  
  return gpu_elapsed_time;
}
