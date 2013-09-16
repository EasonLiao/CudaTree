#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define IDX_DATA_TYPE %s

__global__ void fill_table(
                          IDX_DATA_TYPE *sorted_indices_1,
                          IDX_DATA_TYPE *sorted_indices_2,
                          uint8_t *si_idx,
                          uint16_t *feature_idx,
                          IDX_DATA_TYPE *begin_end_idx,
                          IDX_DATA_TYPE *min_split,
                          uint8_t *mark_table,
                          int stride
                          ){
  
  uint16_t reg_fidx = feature_idx[blockIdx.x];
  IDX_DATA_TYPE reg_start_idx = begin_end_idx[2 * blockIdx.x];
  IDX_DATA_TYPE reg_stop_idx = begin_end_idx[2 * blockIdx.x + 1];
  IDX_DATA_TYPE reg_split = min_split[blockIdx.x];
  
  if(reg_split == reg_stop_idx)
    return;

  uint8_t reg_si_idx = si_idx[blockIdx.x];
  IDX_DATA_TYPE* p_sorted_indices = (reg_si_idx == 0)? sorted_indices_1 : sorted_indices_2;
  uint32_t offset = reg_fidx * stride;

  for(int t = threadIdx.x + reg_start_idx; t < reg_stop_idx; t += blockDim.x){
    if(t <= reg_split)
      mark_table[p_sorted_indices[offset + t]] = 1;
    else
      mark_table[p_sorted_indices[offset + t]] = 0;
  
  }
  
}









