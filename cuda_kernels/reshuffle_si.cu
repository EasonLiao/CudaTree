#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define IDX_DATA_TYPE %s


__global__ void reshuffle(uint8_t* mark_table,
                          IDX_DATA_TYPE* sorted_indices,
                          IDX_DATA_TYPE* sorted_indices_out,
                          int n_samples,
                          int split_idx,
                          int stride
                          ){
  int offset = blockIdx.x * stride;
  int left_start = 0;
  int right_start = split_idx + 1;

  for(int i = 0; i < n_samples; ++i){
    if(mark_table[sorted_indices[offset + i]] == 0){
      sorted_indices_out[offset + left_start] = sorted_indices[offset + i];
      left_start++;
    }
    else{
      sorted_indices_out[offset + right_start] = sorted_indices[offset + i];
      right_start++;
    }
  }
}






