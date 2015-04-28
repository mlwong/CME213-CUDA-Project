// ADDED:
#include <algorithm> // using min() function

// Repeating from the tutorial, just in case you haven't looked at it.

// "kernels" or __global__ functions are the entry points to code that executes on the GPU
// The keyword __global__ indicates to the compiler that this function is a GPU entry point.
// __global__ functions must return void, and may only be called or "launched" from code that
// executes on the CPU.

// This kernel implements a per element shift
// by naively loading one byte and shifting it
__global__ void shift_char(const unsigned char *input_array
                         , unsigned char *output_array
                         , unsigned char shift_amount
                         , unsigned int array_length)
{
  // TODO: fill in
  
  const unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

  for (unsigned int i = tid; i < array_length; i += gridDim.x*blockDim.x)
  {
    output_array[i] = input_array[i] + shift_amount;
  }
  
}

//Here we load 4 bytes at a time instead of just 1
//to improve the bandwidth due to a better memory
//access pattern
__global__ void shift_int(const unsigned int *input_array
                        , unsigned int *output_array
                        , unsigned int shift_amount
                        , unsigned int array_length) 
{
  // TODO: fill in
  
  const unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

  for (unsigned int i = tid; i < array_length; i += gridDim.x*blockDim.x)
  {
    output_array[i] = input_array[i] + shift_amount;
  }
  
}

//Here we go even further and load 8 bytes
//does it make a further improvement? No
__global__ void shift_int2(const uint2 *input_array
                         , uint2 *output_array
                         , unsigned int shift_amount
                         , unsigned int array_length) 
{
  // TODO: fill in
  
  const unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

  for (unsigned int i = tid; i < array_length; i += gridDim.x*blockDim.x)
  {
    output_array[i].x = input_array[i].x + shift_amount;
    output_array[i].y = input_array[i].y + shift_amount;
  }
}

//the following three kernels launch their respective kernels
//and report the time it took for the kernel to run

double doGPUShiftChar(const unsigned char *d_input
                    , unsigned char *d_output
                    , unsigned char shift_amount
                    , unsigned int text_size
                    , unsigned int block_size)
{
  // TODO: compute your grid dimensions
  
  unsigned int grid_size = std::min((text_size + block_size - 1)/block_size, (unsigned int) 65535);
  
  event_pair timer;
  start_timer(&timer);

  // TODO: launch kernel
  
  shift_char<<<grid_size, block_size>>>(d_input, d_output, shift_amount, text_size);
  
  check_launch("gpu shift cipher char");
  return stop_timer(&timer);
}

double doGPUShiftUInt(const unsigned char *d_input
                    , unsigned char *d_output
                    , unsigned char shift_amount
                    , unsigned int text_size
                    , unsigned int block_size)
{
  // TODO: compute your grid dimensions
  
  unsigned int num_uint = (text_size + 3)/4;
  unsigned int grid_size = std::min((num_uint + block_size - 1)/block_size, (unsigned int) 65535);
  
  // TODO: compute 4 byte shift value
  
  unsigned int int_shift_amount = ((shift_amount << 24) | (shift_amount << 16) | (shift_amount << 8) | shift_amount);
  
  event_pair timer;
  start_timer(&timer);

  // TODO: launch kernel
  
  shift_int<<<grid_size, block_size>>>((const unsigned int *)d_input, (unsigned int *)d_output, int_shift_amount, num_uint);
  
  check_launch("gpu shift cipher uint");
  return stop_timer(&timer);
}

double doGPUShiftUInt2(const unsigned char *d_input
                     , unsigned char *d_output
                     , unsigned char shift_amount
                     , unsigned int text_size
                     , unsigned int block_size)
{
  // TODO: compute your grid dimensions
  
  unsigned int num_uint2 = (text_size + 7)/8;
  unsigned int grid_size = std::min((num_uint2 + block_size - 1)/block_size, 65535u);
  
  
  // TODO: compute 4 byte shift value
  
  unsigned int int_shift_amount = ((shift_amount << 24) | (shift_amount << 16) | (shift_amount << 8) | shift_amount);
  
  event_pair timer;
  start_timer(&timer);

  // TODO: launch kernel
  
  shift_int2<<<grid_size, block_size>>>((const uint2 *)d_input, (uint2 *)d_output, int_shift_amount, num_uint2);
  
  check_launch("gpu shift cipher uint2");
  return stop_timer(&timer);
}
