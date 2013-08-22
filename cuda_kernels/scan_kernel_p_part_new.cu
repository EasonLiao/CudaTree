#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define MAX_NUM_SAMPLES %d
#define MAX_NUM_LABELS %d
#define THREADS_PER_BLOCK %d
#define LABEL_DATA_TYPE %s
#define COUNT_DATA_TYPE %s

__global__ void prefix_scan(LABEL_DATA_TYPE *sorted_labels, 
                        COUNT_DATA_TYPE *label_count,
                        int n_features, 
                        int n_samples, 
                        int stride){  
  int range = ceil(double(n_samples) / blockDim.x);            //The range of elements each thread is reponsible for. 
  int n_active_threads = ceil(double(n_samples) / range);     //The number of threads that have the actual work to do.
  
  int count_offset = blockIdx.x * MAX_NUM_LABELS * (blockDim.x + 1);
  int offset = blockIdx.x * stride;
  int range_begin = threadIdx.x * range; 
  int range_end = (range_begin + range < n_samples)? range_begin + range : n_samples;
  
  if(threadIdx.x >= n_active_threads)
    return;

  for(int i = 0; i < MAX_NUM_LABELS; ++i)
    label_count[count_offset + (threadIdx.x + 1) * MAX_NUM_LABELS + i] = 0;

  for(int i = threadIdx.x; i < MAX_NUM_LABELS; i += n_active_threads)
    label_count[count_offset + i] = 0;


  for(int i = range_begin; i < range_end; ++i){
    LABEL_DATA_TYPE cur_label = sorted_labels[offset + i];
    label_count[count_offset + (threadIdx.x + 1) * MAX_NUM_LABELS + cur_label]++; 
  }

  __syncthreads();
  
  for(int i = 2; i < n_active_threads + 1; ++i){
    int last = (i - 1) * MAX_NUM_LABELS;
    int cur = i * MAX_NUM_LABELS;
    
    for(int l = threadIdx.x; l < MAX_NUM_LABELS; l += n_active_threads)
      label_count[count_offset + cur + l] += label_count[count_offset +  last + l];
   
    __syncthreads();
  }
}
