#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define MAX_NUM_LABELS %d
#define LABEL_DATA_TYPE %s
#define COUNT_DATA_TYPE %s

__global__ void prefix_scan(LABEL_DATA_TYPE *sorted_labels, 
                        COUNT_DATA_TYPE *label_count,
                        int range, 
                        int n_samples){
    int thread_offset = blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x + thread_offset;
    int range_begin = thread_id * range;
    int range_end = (range_begin +  range < n_samples)? range_begin + range : n_samples;
    int n_active_threads = ceil(double(n_samples) / range);

  
    for(int l = 0; l < MAX_NUM_LABELS; ++l)
      label_count[(thread_id + 1) * MAX_NUM_LABELS + l] = 0;
   
    if(thread_id == 0)
      for(int l = 0; l < MAX_NUM_LABELS; ++l)
        label_count[l] = 0;
  
    if(thread_id < n_active_threads){
      LABEL_DATA_TYPE label_val = sorted_labels[range_begin];
      label_count[(thread_id + 1) * MAX_NUM_LABELS + label_val]++;
    }
    
    for(int i = range_begin + 1; i < range_end; ++i){
      LABEL_DATA_TYPE label_val = sorted_labels[i];
      label_count[(thread_id + 1) * MAX_NUM_LABELS + label_val]++;
    }
    
    __syncthreads();

    if(threadIdx.x == 0){
      for(int i = 1; i < blockDim.x; ++i)
        for(int l = 0; l < MAX_NUM_LABELS; ++l)
          label_count[(i + 1 + thread_offset) * MAX_NUM_LABELS + l] += label_count[(i  + thread_offset) * MAX_NUM_LABELS + l]; 
    }
}
