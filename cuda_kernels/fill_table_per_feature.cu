#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define IDX_DATA_TYPE %s

__global__ void fill_table(IDX_DATA_TYPE *sorted_indices,
                          int n_samples,
                          int split_idx,
                          uint8_t  *mark_table
                          ){

    for(int i = threadIdx.x; i < n_samples; i += blockDim.x)
      if(i <= split_idx)
        mark_table[sorted_indices[i]] = 0;
      else 
        mark_table[sorted_indices[i]] = 1;
}









