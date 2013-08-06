//Do the simple paralle prefix scan. The label_count is in column major.
#include<stdio.h>
#include<math.h>
#define MAX_NUM_SAMPLES %d
#define MAX_NUM_LABELS %d
#define THREADS_PER_BLOCK %d

__global__ void prefix_scan(int* sorted_targets, 
                        int *label_count,
                        int n_features, 
                        int n_samples, 
                        int leading){

  int label_offset = blockIdx.x * MAX_NUM_LABELS * n_samples; //The offset of label_count for each block.
  int targets_offset = blockIdx.x * n_samples;                //The offset of sorted_targets for each block

  int range = ceil(double(n_samples) / blockDim.x);            //The range of elements each thread is reponsible for. 
  int n_active_threads = ceil(double(n_samples) / range);     //The number of threads that have the actual work to do.
  int range_begin =(threadIdx.x * range < n_samples)? threadIdx.x * range : n_samples;  //The begin index of each thread.
  int range_end = (range_begin + range < n_samples)? range_begin + range : n_samples;   //The end index of each thread

  //Initialize the first label_count of each thread
  if(threadIdx.x < n_active_threads){
    for(int i = 0; i < MAX_NUM_LABELS; ++i)
      label_count[label_offset + i * n_samples + range_begin] = 0;
    
    int cur_label = sorted_targets[targets_offset + range_begin];
    label_count[label_offset + cur_label * n_samples + range_begin]++;    
  }
  
  //Work out a range of label_count of each thread 
  for(int i = range_begin + 1; i < range_end; ++i){
    for(int l = 0; l < MAX_NUM_LABELS; ++l)
      label_count[label_offset + i + l * n_samples] = label_count[label_offset + i - 1 + l * n_samples];

    int cur_label = sorted_targets[targets_offset + i];
    label_count[label_offset + cur_label * n_samples + i]++; 
  }
  
  __syncthreads();
  
  //Fist thread of the block does prefix sum on last element of label_count each thread
  if(threadIdx.x == 0)
    for(int i = 1; i < n_active_threads; ++i)
    { 
      int last = (i * range - 1);
      int cur = (i + 1) * range - 1;
      cur = (cur > n_samples - 1)? (n_samples - 1) : cur;
      
      for(int l = 0; l < MAX_NUM_LABELS; ++l)
        label_count[label_offset + cur + l * n_samples] += label_count[label_offset + last + l * n_samples];
        
    }
 
  __syncthreads();

  //Each thread add the last element of label_count to all the label_count of its range
  if(threadIdx.x > 0)
    for(int i = range_begin; i < range_end - 1; ++i){
      for(int l = 0; l < MAX_NUM_LABELS; ++l)
        label_count[label_offset + l * n_samples + i] += label_count[label_offset + range_begin - 1 + l * n_samples];
    }  
}
