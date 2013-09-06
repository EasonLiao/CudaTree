#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define THREADS_PER_BLOCK %d
#define MAX_NUM_LABELS %d
#define LABEL_DATA_TYPE %s
#define COUNT_DATA_TYPE %s
#define IDX_DATA_TYPE %s

__global__ void count_total(
                        IDX_DATA_TYPE *sorted_indices,
                        LABEL_DATA_TYPE *labels, 
                        COUNT_DATA_TYPE *label_total,
                        int n_samples
                        ){
  
  
  __shared__ COUNT_DATA_TYPE shared_count[MAX_NUM_LABELS];
  __shared__ LABEL_DATA_TYPE shared_labels[THREADS_PER_BLOCK]; 
  IDX_DATA_TYPE stop_pos;
  
  for(int i = threadIdx.x; i < MAX_NUM_LABELS; i += blockDim.x)
    shared_count[i] = 0;
  

  for(int i = threadIdx.x; i < n_samples; i += blockDim.x){
    shared_labels[threadIdx.x] = labels[sorted_indices[i]];
    
    __syncthreads();

    if(threadIdx.x == 0){
      stop_pos = (i + blockDim.x < n_samples)? blockDim.x : n_samples - i;

      for(int t = 0; t < stop_pos; ++t)
        shared_count[shared_labels[t]]++;
    } 

    __syncthreads();
  }
   
  if(threadIdx.x == 0)
    for(int i = 0; i < MAX_NUM_LABELS; i++)
      label_total[i] = shared_count[i];
}
