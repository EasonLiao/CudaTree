#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <assert.h>

#define THREADS_PER_BLOCK %d
#define MAX_NUM_LABELS %d
#define LABEL_DATA_TYPE %s
#define COUNT_DATA_TYPE %s
#define IDX_DATA_TYPE %s
#define MAX_BLOCK_PER_FEATURE %d
#define DEBUG %d

__global__ void count_total(
                        IDX_DATA_TYPE *sorted_indices,
                        LABEL_DATA_TYPE *labels, 
                        COUNT_DATA_TYPE *label_total_2d,
                        uint16_t *subset_indices,
                        int n_range,
                        int n_samples,
                        int stride
                        ){
   
  /* 
    Fill the label_total_2d array for each feature and each range of that feature. 
    Inputs: 
      - sorted_indices : sorted indices.
      - labels : labels.
      - subset_indices : randomly generated featureindices. determine which feature we should count. 
      - n_range : the range of each block is responsible for.
      - n_samples : number of samples for this node.
      - stride : the stride of sorted_indices.
  
    Outputs:
      - label_total_2d : the label total for each range.
  */
  __shared__ int shared_count[MAX_NUM_LABELS];
  uint32_t offset = blockIdx.x * MAX_NUM_LABELS * (MAX_BLOCK_PER_FEATURE + 1) + (blockIdx.y + 1) * MAX_NUM_LABELS;
  uint32_t subset_offset = subset_indices[blockIdx.x] * stride;
  
  IDX_DATA_TYPE start_pos = blockIdx.y * n_range;
  IDX_DATA_TYPE stop_pos = (start_pos + n_range < n_samples)? start_pos + n_range: n_samples;

  for(uint16_t i = threadIdx.x; i < MAX_NUM_LABELS; i += blockDim.x)
    shared_count[i] = 0;
  
  __syncthreads();

  for(IDX_DATA_TYPE i = start_pos; i < stop_pos; i += blockDim.x){
    IDX_DATA_TYPE idx = i + threadIdx.x;
    if(idx < stop_pos)
      atomicAdd(shared_count + labels[sorted_indices[subset_offset + idx]], 1);
  }

  #if DEBUG == 1
  __syncthreads();
  if(threadIdx.x == 0){
    int sum = 0;
    for(int i = 0; i < MAX_NUM_LABELS; ++i)
      sum += shared_count[i];
    
    assert(sum == stop_pos - start_pos);
  }
  #endif

  __syncthreads();

  for(uint16_t i = threadIdx.x; i < MAX_NUM_LABELS; i += blockDim.x)
    label_total_2d[offset + i] = shared_count[i];
}
