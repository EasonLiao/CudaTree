#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define IDX_DATA_TYPE %s
#define SAMPLE_DATA_TYPE %s
#define LABEL_DATA_TYPE %s
#define THREADS_PER_BLOCK %s
#define WARP_SIZE 32
#define WARP_MASK 0x1f

__global__ void scan_reshuffle(uint8_t* mark_table,
                          IDX_DATA_TYPE* sorted_indices,
                          SAMPLE_DATA_TYPE* sorted_samples,
                          LABEL_DATA_TYPE* sorted_labels,
                          IDX_DATA_TYPE* sorted_indices_out,
                          SAMPLE_DATA_TYPE* sorted_samples_out,
                          LABEL_DATA_TYPE* sorted_labels_out,
                          int n_samples,
                          int split_idx,
                          int stride
                          ){
  
  int indices_offset = blockIdx.x * stride;
  int reg_pos = 0;
  int out_pos;
  int right_pos = indices_offset + split_idx + 1;
  uint8_t side;
  int lane_id = threadIdx.x & WARP_MASK;
  int warp_id = threadIdx.x / WARP_SIZE;
  int n;

  __shared__ IDX_DATA_TYPE last_sum;
  __shared__ IDX_DATA_TYPE shared_pos_table[THREADS_PER_BLOCK / WARP_SIZE];
  

  if(threadIdx.x == 0)
    last_sum = 0;
  
  for(int i = threadIdx.x; i < n_samples; i += blockDim.x){
    side = mark_table[sorted_indices[indices_offset + i]];
    reg_pos = side;

    for(int s = 1; s <= WARP_SIZE; s *= 2){
      n = __shfl_up(reg_pos, s);
      if(lane_id >= s)
        reg_pos += n;
    }

    if(lane_id == WARP_SIZE - 1)
      shared_pos_table[warp_id] = reg_pos;
   
    __syncthreads();
   
    if(threadIdx.x == 0)
      for(int l = 1; l < blockDim.x / WARP_SIZE - 1; ++l)
        shared_pos_table[l] += shared_pos_table[l-1];

    __syncthreads();

    if(warp_id > 0)
      reg_pos += shared_pos_table[warp_id - 1];
      
    reg_pos += last_sum; 
    
    __syncthreads();
    
    out_pos = (side == 1)? indices_offset + reg_pos - 1 : right_pos + i - reg_pos ;
    sorted_indices_out[out_pos] = sorted_indices[indices_offset + i];  
    sorted_samples_out[out_pos] = sorted_samples[indices_offset + i];  
    sorted_labels_out[out_pos] = sorted_labels[indices_offset + i];  
    
    if(threadIdx.x == blockDim.x - 1)
      last_sum = reg_pos; 
  }
}

