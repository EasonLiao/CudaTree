#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define IDX_DATA_TYPE %s
#define THREADS_PER_BLOCK %s

texture<char, 1> tex_mark;

__global__ void scan_reshuffle(uint8_t* mark_table,
                          IDX_DATA_TYPE* sorted_indices,
                          IDX_DATA_TYPE* sorted_indices_out,
                          int n_samples,
                          int split_idx,
                          int stride
                          ){  
  uint32_t indices_offset = blockIdx.x * stride;
  IDX_DATA_TYPE reg_pos = 0;
  uint32_t out_pos;
  uint32_t right_pos = indices_offset + split_idx + 1;
  uint8_t side;
  int n;

  __shared__ IDX_DATA_TYPE last_sum;
  __shared__ IDX_DATA_TYPE shared_pos_table[THREADS_PER_BLOCK];
  

  if(threadIdx.x == 0)
    last_sum = 0;
  
  for(IDX_DATA_TYPE i = threadIdx.x; i < n_samples; i += blockDim.x){
    side = tex1Dfetch(tex_mark, sorted_indices[indices_offset + i]);//mark_table[sorted_indices[indices_offset + i]];
    reg_pos = side;
    
    shared_pos_table[threadIdx.x] = reg_pos;

    __syncthreads();
     
    for(uint16_t s = 1; s < blockDim.x; s *= 2){
      if(threadIdx.x >= s){
        n = shared_pos_table[threadIdx.x - s];
      }
      else 
        n = 0;

      __syncthreads();
      shared_pos_table[threadIdx.x] += n;

      __syncthreads();
    }


    __syncthreads();
    
    reg_pos = shared_pos_table[threadIdx.x] + last_sum; 
    out_pos = (side == 1)? indices_offset + reg_pos - 1 : right_pos + i - reg_pos ;
    sorted_indices_out[out_pos] = sorted_indices[indices_offset + i];  
    
    __syncthreads();
    
    if(threadIdx.x == blockDim.x - 1)
      last_sum = reg_pos; 
  }
}

