#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define MAX_NUM_LABELS %d
#define THREADS_PER_BLOCK %d
#define LABEL_DATA_TYPE %s
#define COUNT_DATA_TYPE %s
#define IDX_DATA_TYPE %s

__global__ void prefix_scan(LABEL_DATA_TYPE *labels, 
                        COUNT_DATA_TYPE *label_count,
                        IDX_DATA_TYPE *sorted_indices,                        
                        int n_features, 
                        int n_samples, 
                        int range,
                        int n_active_threads,
                        int stride){

  int label_offset = blockIdx.x * MAX_NUM_LABELS * (blockDim.x + 1); //The offset of label_count for each block.
  int indices_offset = blockIdx.x * stride;                //The offset of sorted_targets for each block

  int range_begin =(threadIdx.x * range < n_samples)? threadIdx.x * range : n_samples;  //The begin index of each thread.
  int range_end = (range_begin + range < n_samples)? range_begin + range : n_samples;   //The end index of each thread
  
  //Initialize the first label_count of each thread
  for(int i = 0; i < MAX_NUM_LABELS; ++i)
    label_count[label_offset + i + (threadIdx.x + 1) * MAX_NUM_LABELS] = 0;

  if(threadIdx.x == 0)
    for(int i = 0; i < MAX_NUM_LABELS; ++i)
      label_count[label_offset + i] = 0;
  

  if(threadIdx.x < n_active_threads){  
    LABEL_DATA_TYPE cur_label = labels[sorted_indices[indices_offset + range_begin]];
    label_count[label_offset + cur_label + (threadIdx.x + 1) * MAX_NUM_LABELS]++;    
  }
    
  __syncthreads();
  //Work out a range of label_count of each thread
  for(int i = range_begin + 1; i < range_end; ++i){
    LABEL_DATA_TYPE cur_label = labels[sorted_indices[indices_offset + i]];
    label_count[label_offset + cur_label + (threadIdx.x + 1) * MAX_NUM_LABELS]++; 
  }
  
  
  //Fist thread of the block does prefix sum on last element of label_count each thread
  if(threadIdx.x == 0)
    for(int i = 1; i < n_active_threads + 1; ++i)
    { 
      int last = (i - 1) * MAX_NUM_LABELS;
      int cur = i * MAX_NUM_LABELS;
      
      for(int l = 0; l < MAX_NUM_LABELS; ++l)
        label_count[label_offset + cur + l] += label_count[label_offset + last + l];    
    }
}
