#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define THREADS_PER_BLOCK %d
#define MAX_NUM_LABELS %d
#define LABEL_DATA_TYPE %s
#define COUNT_DATA_TYPE %s
#define IDX_DATA_TYPE %s

__global__ void count_total(
                          IDX_DATA_TYPE *sorted_indices_1,
                          IDX_DATA_TYPE *sorted_indices_2,
                          LABEL_DATA_TYPE *labels,
                          COUNT_DATA_TYPE *label_total,
                          uint8_t *si_idx,
                          uint32_t *begin_stop_idx
                          ){
  
  IDX_DATA_TYPE *p_sorted_indices;
  IDX_DATA_TYPE reg_start_idx;
  IDX_DATA_TYPE reg_stop_idx;
  __shared__ int shared_count[MAX_NUM_LABELS];
  //__shared__ LABEL_DATA_TYPE shared_labels[THREADS_PER_BLOCK];

  for(uint16_t i = threadIdx.x; i < MAX_NUM_LABELS; i += blockDim.x)
    shared_count[i] = 0;
  
  reg_start_idx = begin_stop_idx[2 * blockIdx.x];
  reg_stop_idx = begin_stop_idx[2 * blockIdx.x + 1];
  
  uint8_t reg_si_idx = si_idx[blockIdx.x];
  if(reg_si_idx == 0)
    p_sorted_indices = sorted_indices_1;
  else 
    p_sorted_indices = sorted_indices_2;
  
  __syncthreads();

  for(IDX_DATA_TYPE i = reg_start_idx; i < reg_stop_idx; i += blockDim.x){
    IDX_DATA_TYPE idx = i + threadIdx.x;

    if(idx < reg_stop_idx)
      atomicAdd(shared_count + labels[p_sorted_indices[idx]], 1);
  }

  __syncthreads();

  for(uint16_t i = threadIdx.x; i < MAX_NUM_LABELS; i += blockDim.x)
    label_total[blockIdx.x * MAX_NUM_LABELS + i] = shared_count[i];
}
 








