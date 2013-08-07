//Do the simple paralle prefix scan. It's not coalesced memory access.
#include<stdio.h>
#include<math.h>
#define MAX_NUM_SAMPLES %d
#define MAX_NUM_LABELS %d
#define THREADS_PER_BLOCK %d


__global__ void prefix_scan(int* sorted_targets, 
                        int *label_count,
                        int n_features, 
                        int n_samples, 
                        int stride){

  int label_offset = blockIdx.x * MAX_NUM_LABELS * n_samples; //The offset of label_count for each block.
  int targets_offset = blockIdx.x * stride;                //The offset of sorted_targets for each block

  int range = ceil(double(n_samples) / blockDim.x);            //The range of elements each thread is reponsible for. 
  int n_active_threads = ceil(double(n_samples) / range);     //The number of threads that have the actual work to do.
  int range_begin =(threadIdx.x * range < n_samples)? threadIdx.x * range : n_samples;  //The begin index of each thread.
  int range_end = (range_begin + range < n_samples)? range_begin + range : n_samples;   //The end index of each thread

  //Initialize the first label_count of each thread
  if(threadIdx.x < n_active_threads){
    for(int i = 0; i < MAX_NUM_LABELS; ++i)
      label_count[label_offset + i + range_begin * MAX_NUM_LABELS] = 0;
    
    int cur_label = sorted_targets[targets_offset + range_begin];
    label_count[label_offset + cur_label + range_begin * MAX_NUM_LABELS]++;    
  }
  
  //Work out a range of label_count of each thread 
  for(int i = range_begin + 1; i < range_end; ++i){
    for(int l = 0; l < MAX_NUM_LABELS; ++l)
      label_count[label_offset + i * MAX_NUM_LABELS + l] = label_count[label_offset + (i - 1) * MAX_NUM_LABELS + l];

    int cur_label = sorted_targets[targets_offset + i];
    label_count[label_offset + cur_label + i * MAX_NUM_LABELS]++; 
  }
  
  __syncthreads();

  //Fist thread of the block does prefix sum on last element of label_count each thread
  if(threadIdx.x == 0)
    for(int i = 1; i < n_active_threads; ++i)
    { 
      int last = (i * range - 1) * MAX_NUM_LABELS;
      int cur = (i + 1) * range - 1;
      cur = (cur > n_samples - 1)? (n_samples - 1) * MAX_NUM_LABELS : cur * MAX_NUM_LABELS;
      
      for(int l = 0; l < MAX_NUM_LABELS; ++l)
        label_count[label_offset + cur + l] += label_count[label_offset + last + l];
        
    }
 
  __syncthreads();

  //Each thread add the last element of label_count to all the label_count of its range
  if(threadIdx.x > 0)
    for(int i = range_begin; i < range_end - 1; ++i){
      int cur_off =  i * MAX_NUM_LABELS;
      int last_off = (range_begin - 1) * MAX_NUM_LABELS;;

      for(int l = 0; l < MAX_NUM_LABELS; ++l)
        label_count[label_offset + l + cur_off] += label_count[label_offset + last_off + l];
    }  
}
