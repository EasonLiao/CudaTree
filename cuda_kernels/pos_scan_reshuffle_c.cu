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
  int n;

  __shared__ IDX_DATA_TYPE last_sum;
  __shared__ IDX_DATA_TYPE shared_pos_table[THREADS_PER_BLOCK];

  if(threadIdx.x == 0)
    last_sum = 0;
  
  for(int i = threadIdx.x; i < n_samples; i += blockDim.x){
    side = mark_table[sorted_indices[indices_offset + i]];
    reg_pos = side;
    
    shared_pos_table[threadIdx.x] = reg_pos;
    
    __syncthreads();

    for(int s = 1; s < blockDim.x; s *= 2){
      if(threadIdx.x >= s){
        n = shared_pos_table[threadIdx.x - s];
      }
      else 
        n = 0;

      __syncthreads();
      shared_pos_table[threadIdx.x] += n;

      __syncthreads();
    }
    
    /*
    if(threadIdx.x == 0)
      for(int i = 1; i < blockDim.x; ++i)
        shared_pos_table[i] += shared_pos_table[i-1];
    */
    
    __syncthreads();
    
    
    reg_pos = shared_pos_table[threadIdx.x] + last_sum;
    
    out_pos = (side == 1)? indices_offset + reg_pos - 1 : right_pos + i - reg_pos ;
    sorted_indices_out[out_pos] = sorted_indices[indices_offset + i];  
    sorted_samples_out[out_pos] = sorted_samples[indices_offset + i];  
    sorted_labels_out[out_pos] = sorted_labels[indices_offset + i];  

    __syncthreads();

    if(threadIdx.x == blockDim.x - 1)
      last_sum = reg_pos; 
  }
}

