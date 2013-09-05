#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define IDX_DATA_TYPE %s
#define THREADS_PER_BLOCK %s

__global__ void scan_reshuffle(uint8_t* mark_table,
                          IDX_DATA_TYPE* sorted_indices,
                          IDX_DATA_TYPE* sorted_indices_out,
                          int range,
                          int n_active_threads,
                          int n_samples,
                          int split_idx,
                          int stride
                          ){
  int indices_offset = blockIdx.x * stride;
  int range_begin = threadIdx.x * range;
  int range_end = (range_begin +  range < n_samples)? range_begin + range : n_samples;
  int reg_pos = 0;
  int out_pos;
  int right_pos = indices_offset + split_idx + 1;
  uint8_t side;
  
  __shared__ IDX_DATA_TYPE shared_pos_table[THREADS_PER_BLOCK];
  
  if(threadIdx.x >= n_active_threads)
    return;
  
  if(threadIdx.x < n_active_threads - 1){
    for(int i = range_begin; i < range_end; ++i)
      reg_pos += mark_table[sorted_indices[indices_offset + i]];
    
    shared_pos_table[threadIdx.x + 1] = reg_pos;
  }
  else
    shared_pos_table[0] = 0;
  
  __syncthreads();
  
  if(threadIdx.x == 0){
    for(int i = 1; i < n_active_threads; ++i)
      shared_pos_table[i] += shared_pos_table[i-1];
  }
 
  __syncthreads();

  reg_pos = shared_pos_table[threadIdx.x]; 

  for(int i = range_begin; i < range_end; ++i){
    side = mark_table[sorted_indices[indices_offset + i]];  
    out_pos = (side == 1)? indices_offset + reg_pos : right_pos + i - reg_pos;
      
    sorted_indices_out[out_pos] = sorted_indices[indices_offset + i];
      
    reg_pos += side; 
  }
}

