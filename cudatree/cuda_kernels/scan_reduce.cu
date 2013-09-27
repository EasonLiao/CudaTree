#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define MAX_NUM_LABELS %d
#define COUNT_DATA_TYPE %s
#define MAX_BLOCK_PER_FEATURE %d

__global__ void scan_reduce(
                        COUNT_DATA_TYPE *label_total_2d,
                        int n_block
                        ){

  uint32_t offset = blockIdx.x * (MAX_BLOCK_PER_FEATURE + 1) * MAX_NUM_LABELS;
  
  for(uint16_t i = 2; i <= n_block; ++i){
    
    uint32_t last_off = (i - 1) * MAX_NUM_LABELS;
    uint32_t this_off = i * MAX_NUM_LABELS;

    for(uint16_t t = threadIdx.x; t < MAX_NUM_LABELS; t += blockDim.x)
      label_total_2d[offset + this_off + t] += label_total_2d[offset + last_off + t];
  } 
}
