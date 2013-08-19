#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define IDX_DATA_TYPE %s
#define THREADS_PER_BLOCK %s
#define WARP_SIZE 32
#define WARP_MASK 0x1f


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
  int old_pos;
  int out_pos;
  int right_pos = indices_offset + split_idx + 1;
  uint8_t side;
  int lane_id = threadIdx.x & WARP_MASK;
  int warp_id = threadIdx.x / WARP_SIZE;
  int n;

  __shared__ IDX_DATA_TYPE shared_pos_table[THREADS_PER_BLOCK / WARP_SIZE];
  
  if(threadIdx.x >= n_active_threads)
    return;
  
  if(threadIdx.x < n_active_threads)
    for(int i = range_begin; i < range_end; ++i)
      reg_pos += mark_table[sorted_indices[indices_offset + i]];
  
  old_pos = reg_pos;
  
  for(int i = 1; i <= WARP_SIZE; i *= 2){
    n = __shfl_up(reg_pos, i);
    if(lane_id >= i)
      reg_pos += n;
  }

  reg_pos -= old_pos;

  if(lane_id == WARP_SIZE - 1)
    shared_pos_table[warp_id] = reg_pos;

  __syncthreads();
  
  if(threadIdx.x == 0)
    for(int i = 1; i < THREADS_PER_BLOCK / WARP_SIZE; ++i)
      shared_pos_table[i] += shared_pos_table[i-1];
  
  __syncthreads();
  
  if(warp_id > 0)
    reg_pos += shared_pos_table[warp_id - 1];
  
  for(int i = range_begin; i < range_end; ++i){
    side = mark_table[sorted_indices[indices_offset + i]];  
    out_pos = (side == 1)? indices_offset + reg_pos : right_pos + i - reg_pos;
    sorted_indices_out[out_pos] = sorted_indices[indices_offset + i];  
    reg_pos += side; 
  }
}

