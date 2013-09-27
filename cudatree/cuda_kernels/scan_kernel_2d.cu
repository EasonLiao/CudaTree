#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define THREADS_PER_BLOCK %d
#define MAX_NUM_LABELS %d
#define LABEL_DATA_TYPE %s
#define COUNT_DATA_TYPE %s
#define IDX_DATA_TYPE %s
#define MAX_BLOCK_PER_FEATURE %d

__global__ void count_total(
                        IDX_DATA_TYPE *sorted_indices,
                        LABEL_DATA_TYPE *labels, 
                        COUNT_DATA_TYPE *label_total,
                        IDX_DATA_TYPE *subset_indices,
                        int n_range,
                        int n_samples,
                        int stride
                        ){
   
  __shared__ COUNT_DATA_TYPE shared_count[MAX_NUM_LABELS];
  __shared__ LABEL_DATA_TYPE shared_labels[THREADS_PER_BLOCK]; 
  uint32_t offset = blockIdx.x * MAX_NUM_LABELS * (MAX_BLOCK_PER_FEATURE + 1) + (blockIdx.y + 1) * MAX_NUM_LABELS;
  uint32_t subset_offset = subset_indices[blockIdx.x] * stride;
  
  IDX_DATA_TYPE start_pos = blockIdx.y * n_range;
  IDX_DATA_TYPE stop_pos = (start_pos + n_range < n_samples)? start_pos + n_range: n_samples;


  for(uint16_t i = threadIdx.x; i < MAX_NUM_LABELS; i += blockDim.x)
    shared_count[i] = 0;

  
  for(IDX_DATA_TYPE i = start_pos; i < stop_pos; i += blockDim.x){
    IDX_DATA_TYPE idx = i + threadIdx.x;
    if(idx < stop_pos)
      shared_labels[threadIdx.x] = labels[sorted_indices[subset_offset + idx]];

    __syncthreads();
    
    if(threadIdx.x == 0){
      IDX_DATA_TYPE end_pos = (i + blockDim.x < stop_pos) ? blockDim.x : stop_pos - i;
      for(IDX_DATA_TYPE t = 0; t < end_pos; ++t)
        shared_count[shared_labels[t]]++;
    }
    
    __syncthreads();

  }
  
  
  for(uint16_t i = threadIdx.x; i < MAX_NUM_LABELS; i += blockDim.x)
    label_total[offset + i] = shared_count[i];
}
