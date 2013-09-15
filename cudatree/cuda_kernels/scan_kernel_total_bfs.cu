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
                          IDX_DATA_TYPE *begin_stop_idx,
                          IDX_DATA_TYPE *subset_indices,
                          int max_features
                          ){
  
  IDX_DATA_TYPE *p_sorted_indices;
  IDX_DATA_TYPE shared_start_idx;
  IDX_DATA_TYPE shared_stop_idx;
  __shared__ COUNT_DATA_TYPE shared_count[MAX_NUM_LABELS];
  __shared__ LABEL_DATA_TYPE shared_labels[THREADS_PER_BLOCK];

  for(int i = threadIdx.x; i < MAX_NUM_LABELS; i += blockDim.x)
    shared_count[i] = 0;
  
  shared_start_idx = begin_stop_idx[2 * blockIdx.x];
  shared_stop_idx = begin_stop_idx[2 * blockIdx.x + 1];
  
  uint8_t reg_si_idx = si_idx[blockIdx.x];
  if(reg_si_idx == 0)
    p_sorted_indices = sorted_indices_1;
  else 
    p_sorted_indices = sorted_indices_2;
   

  IDX_DATA_TYPE n_samples = shared_stop_idx - shared_start_idx;

  for(IDX_DATA_TYPE i = 0; i < n_samples; i += blockDim.x){
    if(i + threadIdx.x < n_samples)
      shared_labels[threadIdx.x] = labels[p_sorted_indices[shared_start_idx + i + threadIdx.x]];
    
    __syncthreads();
    
    if(threadIdx.x == 0){
      IDX_DATA_TYPE stop_pos = (i + blockDim.x  < n_samples)? blockDim.x : n_samples - i;

      for(int t = 0; t < stop_pos; ++t)
        shared_count[shared_labels[t]]++;
    }
    
    __syncthreads();
  }

  for(LABEL_DATA_TYPE i = threadIdx.x; i < MAX_NUM_LABELS; i += blockDim.x)
    label_total[blockIdx.x * MAX_NUM_LABELS + i] = shared_count[i];
}
 








