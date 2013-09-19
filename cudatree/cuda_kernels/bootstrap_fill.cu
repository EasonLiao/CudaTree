#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define IDX_DATA_TYPE %s

__global__ void bootstrap_fill(
                          IDX_DATA_TYPE *rand_indices,
                          uint8_t *mark_table,
                          uint32_t n_samples,
                          uint32_t stride
                          ){
  for(IDX_DATA_TYPE i = threadIdx.x; i < stride; i += blockDim.x)
    mark_table[i] = 0;
  
  __syncthreads();
  
  for(IDX_DATA_TYPE i = threadIdx.x; i < n_samples; ++i)
    mark_table[rand_indices[i]] = 1;
}









