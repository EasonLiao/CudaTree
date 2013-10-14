#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define IDX_DATA_TYPE %s
#define SAMPLE_DATA_TYPE %s


__global__ void feature_selector(IDX_DATA_TYPE *sorted_indices,
                          SAMPLE_DATA_TYPE *samples,
                          uint8_t* boolean_mask,
                          int n_samples,
                          int stride
                          ){
    uint32_t offset = blockIdx.x * stride;
    if(samples[offset + sorted_indices[offset]] == samples[offset + sorted_indices[offset + n_samples -1]])
      boolean_mask[blockIdx.x] = 0;
    else
      boolean_mask[blockIdx.x] = 1;

}









