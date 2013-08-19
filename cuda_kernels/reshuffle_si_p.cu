#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define IDX_DATA_TYPE %s

__global__ void reshuffle(uint8_t* mark_table,
                          IDX_DATA_TYPE* sorted_indices,
                          IDX_DATA_TYPE* sorted_indices_out,
                          IDX_DATA_TYPE* pos_table,
                          int range,
                          int n_active_threads,
                          int n_samples,
                          int split_idx,
                          int stride
                          ){

    int indices_offset = blockIdx.x * stride;
    int pos_offset = blockDim.x * blockIdx.x;
    int range_begin = threadIdx.x * range;
    int range_end = (range_begin +  range < n_samples)? range_begin + range : n_samples;
    int reg_pos = 0;
    int out_pos;
    int right_pos = indices_offset + split_idx + 1; 

    uint8_t side;

    if(threadIdx.x >= n_active_threads)
      return;
    
    reg_pos = pos_table[pos_offset + threadIdx.x];

    for(int i = range_begin; i < range_end; ++i){
      side = mark_table[sorted_indices[indices_offset + i]];  
      out_pos = (side == 1)? indices_offset + reg_pos : right_pos + i - reg_pos;
      
      sorted_indices_out[out_pos] = sorted_indices[indices_offset + i];
      
      reg_pos += side; 
    }
}






