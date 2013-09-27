#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define THREADS_PER_BLOCK 32
#define IDX_DATA_TYPE %s
#define MAX_BLOCK_PER_FEATURE %d

__global__ void reduce(
                      float *impurity_2d,
                      float *impurity_left,
                      float *impurity_right,
                      IDX_DATA_TYPE *split_2d,
                      IDX_DATA_TYPE *split_1d, 
                      int n_block
                      ){
  __shared__ float shared_imp[MAX_BLOCK_PER_FEATURE * 2];
  uint32_t imp_offset = blockIdx.x * MAX_BLOCK_PER_FEATURE * 2;
  uint32_t split_offset = blockIdx.x * MAX_BLOCK_PER_FEATURE;
  
  for(uint16_t t= threadIdx.x; t < n_block * 2; t += blockDim.x)
    shared_imp[t] = impurity_2d[imp_offset + t];
  
  __syncthreads();
  
  if(threadIdx.x == 0){
    float min_left = 2.0;
    float min_right = 2.0;
    uint16_t min_idx = 0;
    for(uint16_t i = 0; i < n_block; ++i)
      if(min_left + min_right > shared_imp[i * 2] + shared_imp[2 * i + 1]){
        min_left = shared_imp[i * 2];
        min_right = shared_imp[2 * i + 1];
        min_idx = i;
      }
    impurity_left[blockIdx.x] = min_left;
    impurity_right[blockIdx.x] = min_right;
    split_1d[blockIdx.x] = split_2d[split_offset + min_idx];
  }
}
