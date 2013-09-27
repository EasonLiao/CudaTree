#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define IDX_DATA_TYPE %s
#define SAMPLE_DATA_TYPE %s

__global__ void get_thresholds
                            (
                            uint8_t *si_idx,
                            IDX_DATA_TYPE *sorted_indices_0,
                            IDX_DATA_TYPE *sorted_indices_1,
                            SAMPLE_DATA_TYPE *samples,
                            float *threshold_values,
                            uint16_t *min_feature_indices,
                            IDX_DATA_TYPE *min_split_idx,
                            int stride
                            ){
    
  IDX_DATA_TYPE *p_sorted_indices;
  uint8_t idx = si_idx[blockIdx.x];
  uint16_t row = min_feature_indices[blockIdx.x];
  IDX_DATA_TYPE col = min_split_idx[blockIdx.x];
  uint32_t offset = row * stride;

  if(idx == 0)
    p_sorted_indices = sorted_indices_0;
  else
    p_sorted_indices = sorted_indices_1;
   
  threshold_values[blockIdx.x] = ((float)samples[offset + p_sorted_indices[offset + col]] + 
                                    samples[offset + p_sorted_indices[offset + col + 1]]) / 2;
}









