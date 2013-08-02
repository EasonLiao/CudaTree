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
                        int leading){
  
  //if(blockIdx.x == 0 && threadIdx.x == 0)
  //  printf("scan_P\n");

  int label_offset = blockIdx.x * MAX_NUM_LABELS * n_samples; //The offset of label_count for each block.
  int targets_offset = blockIdx.x * n_samples;                //The offset of sorted_targets for each block

  int range = ceil(double(n_samples) / blockDim.x);            //The range of elements each thread is reponsible for. 
  int n_active_threads = ceil(double(n_samples) / range);     //The number of threads that have the actual work to do.
  int range_begin =(threadIdx.x * range < n_samples)? threadIdx.x * range : n_samples;  //The begin index of each thread.
  int range_end = (range_begin + range < n_samples)? range_begin + range : n_samples;   //The end index of each thread
  __shared__ int label_sum[THREADS_PER_BLOCK][MAX_NUM_LABELS];  //The shared memory that stores the local label_count sum of each thread.

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

  //Copy the last label_count of each thread to the shared memory.
  if(threadIdx.x < n_active_threads)
    for(int l = 0; l < MAX_NUM_LABELS; ++l)
      label_sum[threadIdx.x][l] = label_count[label_offset + l + (range_end - 1) * MAX_NUM_LABELS];

  __syncthreads();

  //Fist thread of the block does prefix sum on label_sum
  if(threadIdx.x == 0)
    for(int i = 1; i < n_active_threads; ++i)
      for(int l = 0; l < MAX_NUM_LABELS; ++l)
        label_sum[i][l] += label_sum[i - 1][l]; 

  __syncthreads();

  //Each thread add the label_sum[threadIdx.x - 1] to all the label_count of its range
  if(threadIdx.x > 0)
    for(int i = range_begin; i < range_end; ++i){
      for(int l = 0; l < MAX_NUM_LABELS; ++l)
        label_count[label_offset + l + i * MAX_NUM_LABELS] += label_sum[threadIdx.x - 1][l];
    }
}
