#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define IDX_DATA_TYPE %s
#define THREADS_PER_BLOCK %s

__global__ void scan_reshuffle(
                          uint8_t* mark_table,
                          uint8_t* si_idx,
                          IDX_DATA_TYPE* sorted_indices_1,
                          IDX_DATA_TYPE* sorted_indices_2,
                          IDX_DATA_TYPE* begin_end_idx,
                          IDX_DATA_TYPE* split,
                          uint16_t* feature_idx,
                          uint16_t n_features,
                          int stride
                          ){  
  /*
  if(blockIdx.x ==3 && blockIdx.y == 0)
    printf("griddim.x:%%d griddim.y:%%d n_features: %%d, stride: %%d  %%d %%d %%d\n", 
        gridDim.x, gridDim.y, n_features, stride, begin_end_idx[2 * blockIdx.x], split[blockIdx.x], begin_end_idx[2 * blockIdx.x + 1]);
  */

  __shared__ IDX_DATA_TYPE last_sum;
  __shared__ IDX_DATA_TYPE shared_pos_table[THREADS_PER_BLOCK];
  uint16_t feature_table_idx = feature_idx[blockIdx.x];
  IDX_DATA_TYPE reg_start_idx = begin_end_idx[2 * blockIdx.x];
  IDX_DATA_TYPE reg_stop_idx = begin_end_idx[2 * blockIdx.x + 1];
  IDX_DATA_TYPE reg_split_idx = split[blockIdx.x];
  int n;

  
  IDX_DATA_TYPE *p_sorted_indices_in;
  IDX_DATA_TYPE *p_sorted_indices_out;

  if(si_idx[blockIdx.x] == 0){
    p_sorted_indices_in = sorted_indices_1;
    p_sorted_indices_out = sorted_indices_2;
  }else{
    p_sorted_indices_in = sorted_indices_2;
    p_sorted_indices_out = sorted_indices_1;
  }

  for(uint16_t shuffle_feature_idx = blockIdx.y; shuffle_feature_idx < n_features; shuffle_feature_idx += gridDim.y){
    int offset = shuffle_feature_idx * stride;

    if(threadIdx.x == 0)
      last_sum = 0;

    for(int i = threadIdx.x + reg_start_idx; i < reg_stop_idx; i += blockDim.x){
      uint8_t side = mark_table[feature_table_idx * stride + p_sorted_indices_in[offset + i]];
      shared_pos_table[threadIdx.x] = side;
      
      for(int s = 1; s < blockDim.x; s *= 2){
        if(threadIdx.x >= s)
          n = shared_pos_table[threadIdx.x - s];
        else
          n = 0;

        shared_pos_table[threadIdx.x] += n;
      }
      
      int reg_pos = shared_pos_table[threadIdx.x] + last_sum;
      int out_pos = (side == 1)? reg_start_idx + reg_pos - 1 : reg_split_idx + 1 + i - reg_start_idx - reg_pos;
      p_sorted_indices_out[offset + out_pos] = p_sorted_indices_in[offset + i];   
    
      if(threadIdx.x == blockDim.x - 1)
        last_sum = reg_pos;
    }
  }
}

