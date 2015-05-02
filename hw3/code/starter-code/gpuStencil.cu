#include <math_constants.h>

#include "BC.h"

/** 
 * Calculates the next finite difference step given a
 * grid point and step lengths.
 *
 * @param curr Pointer to the grid point that should be updated.
 * @param width Number of grid points in the x dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 * @returns Grid value of next timestep.
 */
template<int order> 
__device__
float Stencil(const float *curr, int width, float xcfl, float ycfl) { 
  switch (order) {
    case 2:
      return curr[0] + xcfl * (curr[-1] + curr[1] - 2.f * curr[0]) +
          ycfl * (curr[width] + curr[-width] - 2.f * curr[0]);
    case 4:
      return curr[0] + xcfl * (- curr[2] + 16.f * curr[1] - 30.f * curr[0] +
          16.f * curr[-1] - curr[-2]) + ycfl * (- curr[2 * width] +
          16.f * curr[width] - 30.f * curr[0] + 16.f * curr[-width] -
          curr[-2 * width]);
    case 8:
      return curr[0] + xcfl * (-9.f * curr[4] + 128.f * curr[3] -
          1008.f * curr[2] + 8064.f * curr[1] - 14350.f * curr[0] +
          8064.f * curr[-1] - 1008.f * curr[-2] + 128.f * curr[-3] -
          9.f * curr[-4]) + ycfl * (-9.f * curr[4 * width] +
          128.f * curr[3 * width] - 1008.f * curr[2 * width] +
          8064.f * curr[width] - 14350.f * curr[0] +
          8064.f * curr[-width] - 1008.f * curr[-2 * width] +
          128.f * curr[-3 * width] - 9.f * curr[-4 * width]);
     default:
       printf("ERROR: Order %d not supported", order);
       return CUDART_NAN_F;
  }
}

/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be very simple and only use global memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundar).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order>
__global__
void gpuStencil(float *next, const float *curr, int gx, int nx, int ny,
                float xcfl, float ycfl)
{
  // TODO
  
  const int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
  const int tid_y = threadIdx.y + blockDim.y*blockIdx.y;
  
  // If the thread is not inside the domain, return
  if (tid_x >= nx || tid_y >= ny)
  {
    return;
  }
  
  // Compute the boarder size
  const int b = order/2;
  
  // Compute the index of the point in 1D array
  const int idx = (tid_y + b)*gx + (tid_x + b);
  
  next[idx] = Stencil<order> (&curr[idx], gx, xcfl, ycfl);
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencil kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputation(Grid &curr_grid, const simParams &params) {

  boundary_conditions BC(params);

  Grid next_grid(curr_grid);

  // TODO: Declare variables/Compute parameters.

  dim3 threads(0, 0);
  dim3 blocks(0, 0);
  
  // Set the size of each block
  const unsigned int block_dim_x = 32u;
  const unsigned int block_dim_y = 16u;
  
  int nx = params.nx();
  int ny = params.ny();
  
  int gx = params.gx();
  
  float xcfl = params.xcfl();
  float ycfl = params.ycfl();
  
  int order = params.order();
  
  // Compute the block dimension
  threads.x = block_dim_x;
  threads.y = block_dim_y;
  
  // Assume each dimension of the block is less than 65536
  // and compute the grid size
  blocks.x = ((unsigned int) nx + threads.x - 1)/threads.x;
  blocks.y = ((unsigned int) ny + threads.y - 1)/threads.y;
  
  event_pair timer;
  start_timer(&timer);
  for (int i = 0; i < params.iters(); ++i) {

    // update the values on the boundary only
    BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

    // TODO: Apply stencil.
    
    switch (order)
    {
      case 2:
        gpuStencil<2><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_,
                                           gx, nx, ny, xcfl, ycfl);
        break;
      case 4:
        gpuStencil<4><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_,
                                           gx, nx, ny, xcfl, ycfl);
        break;
      case 8:
        gpuStencil<8><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_,
                                           gx, nx, ny, xcfl, ycfl);
        break;
      default:
        printf("ERROR: Order %d not supported", order);
        exit(1);
    }
    
    check_launch("gpuStencil");

    Grid::swap(curr_grid, next_grid);
  }

  return stop_timer(&timer);
}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size (blockDim.y * numYPerStep) * blockDim.x. Each thread
 * should calculate at most numYPerStep updates. It should still only use
 * global memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundar).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order, int numYPerStep>
__global__
void gpuStencilLoop(float *next, const float *curr, int gx, int nx, int ny,
                    float xcfl, float ycfl)
{
  // TODO
  
  const int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
  const int tid_y = threadIdx.y + numYPerStep*blockDim.y*blockIdx.y;
  
  // If the thread is not inside the domain, return
  if (tid_x >= nx)
  {
    return;
  }
  
  // Compute the boarder size
  const int b = order/2;
  
  for (int i = 0; i < numYPerStep; i++)
  {
    if (tid_y + i*blockDim.y >= ny)
    {
      return;
    }
    const int idx = (tid_y + i*blockDim.y + b)*gx + (tid_x + b);
    next[idx] = Stencil<order> (&curr[idx], gx, xcfl, ycfl);
  }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilLoop kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationLoop(Grid &curr_grid, const simParams &params) {

  boundary_conditions BC(params);

  Grid next_grid(curr_grid);
  // TODO
  // TODO: Declare variables/Compute parameters.
  
  dim3 threads(0, 0);
  dim3 blocks(0, 0);
  
  // Set the size of each block
  const unsigned int block_dim_x = 64u;
  const unsigned int block_dim_y = 8u;
  const unsigned int numYPerStep = 8u;
  
  int nx = params.nx();
  int ny = params.ny();
  
  int gx = params.gx();
  
  float xcfl = params.xcfl();
  float ycfl = params.ycfl();
  
  int order = params.order();
  
  // Compute the block dimension
  threads.x = block_dim_x;
  threads.y = block_dim_y;
  
  // Assume each dimension of the block is less than 65536
  // and compute the grid size
  blocks.x = ((unsigned int) nx + threads.x - 1)/threads.x;
  blocks.y = ((unsigned int) ny + threads.y*numYPerStep - 1)/(threads.y*numYPerStep);

  event_pair timer;
  start_timer(&timer);

  for (int i = 0; i < params.iters(); ++i) {

    // update the values on the boundary only
    BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

    // TODO: Apply stencil.
    
    switch (order)
    {
      case 2:
        gpuStencilLoop<2, numYPerStep><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_,
                                                            gx, nx, ny, xcfl, ycfl);
        break;
      case 4:
        gpuStencilLoop<4, numYPerStep><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_,
                                                            gx, nx, ny, xcfl, ycfl);
        break;
      case 8:
        gpuStencilLoop<8, numYPerStep><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_,
                                                            gx, nx, ny, xcfl, ycfl);
        break;
      default:
        printf("ERROR: Order %d not supported", order);
        exit(1);
    }

    check_launch("gpuStencilLoop");

    Grid::swap(curr_grid, next_grid);
  }

  return stop_timer(&timer);
}

/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size side * side using shared memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param gy Number of grid points in the y dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int side, int order>
__global__
void gpuShared(float *next, const float *curr, int gx, int gy,
               float xcfl, float ycfl)
{
  // TODO
  
  const int lane_x = threadIdx.x;
  const int lane_y = threadIdx.y;
  
  // Compute the boarder size
  const int b = order/2;
  
  __shared__ float smem_block[side*side];
  
  const int global_col = lane_x + (blockDim.x - order)*blockIdx.x;
  
  // Load the data into shared memory
  if (global_col < gx)
  {
    const int y_lim = min(side, gy - (side - order)*blockIdx.y);
    for (int i = lane_y; i < y_lim; i += blockDim.y)
    {
        const int global_row = i + (side - order)*blockIdx.y;
        const int global_idx = global_row*gx + global_col;
        const int smem_block_idx = i*blockDim.x + lane_x;
        
        smem_block[smem_block_idx] = curr[global_idx];      
    }
  }
  
  __syncthreads();
  
  // Do the computation here
  if (global_col < gx - b)
  {
    if (lane_x >= b && lane_x < blockDim.x - b)
    {
      const int y_lim = min(side - b, gy - b - (side - order)*blockIdx.y);
      for (int i = lane_y + b; i < y_lim; i += blockDim.y)
      {
        const int global_row = i + (side - order)*blockIdx.y;
        const int global_idx = global_row*gx + global_col;
        const int smem_block_idx = i*blockDim.x + lane_x;
        
        next[global_idx] = Stencil<order> (&smem_block[smem_block_idx], side, xcfl, ycfl);
      }
    }
  }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuShared kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
template<int order>
double gpuComputationShared(Grid &curr_grid, const simParams &params) {

  boundary_conditions BC(params);

  Grid next_grid(curr_grid);
 
  // TODO: Declare variables/Compute parameters.
  
  dim3 threads(0, 0);
  dim3 blocks(0, 0);
  
  // Set the size of each block
  const unsigned int block_dim_x = 64u;
  const unsigned int block_dim_y = 8u;
  const unsigned int smem_side = 64u;
  
  int nx = params.nx();
  int ny = params.ny();
  
  int gx = params.gx();
  int gy = params.gy();
  
  float xcfl = params.xcfl();
  float ycfl = params.ycfl();
  
  // Compute the block dimension
  threads.x = block_dim_x;
  threads.y = block_dim_y;
  
  // Assume each dimension of the block is less than 65536
  // and compute the grid size
  blocks.x = ((unsigned int) nx + threads.x - (unsigned int) order - 1)/(threads.x - (unsigned int) order);
  blocks.y = ((unsigned int) ny + smem_side - (unsigned int) order - 1)/(smem_side - (unsigned int) order);
  
  event_pair timer;
  start_timer(&timer);

  for (int i = 0; i < params.iters(); ++i) {

    // update the values on the boundary only
    BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

    // TODO: Apply stencil.
    
    switch (order)
    {
      case 2:
        gpuShared<smem_side, 2><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_,
                                                     gx, gy, xcfl, ycfl);
        break;
      case 4:
        gpuShared<smem_side, 4><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_,
                                                     gx, gy, xcfl, ycfl);
        break;
      case 8:
        gpuShared<smem_side, 8><<<blocks, threads>>>(next_grid.dGrid_, curr_grid.dGrid_,
                                                     gx, gy, xcfl, ycfl);
        break;
      default:
        printf("ERROR: Order %d not supported", order);
        exit(1);
    }
    
    check_launch("gpuShared");

    Grid::swap(curr_grid, next_grid);
  }

  return stop_timer(&timer);
}

