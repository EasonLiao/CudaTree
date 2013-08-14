#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define MAX_NUM_LABELS %d
#define COUNT_DATA_TYPE %s

__global__ void prefix_scan_3( 
                        COUNT_DATA_TYPE *label_count,
                        int n_active_threads,
                        int range,
                        int n_samples
                        ){
  if(blockIdx.x == 0)
    return;
  
  __shared__ COUNT_DATA_TYPE last_block_count[MAX_NUM_LABELS];
  int thread_offset = blockIdx.x * blockDim.x;  
  int thread_id = threadIdx.x + thread_offset;
  int range_begin = thread_id * range;
  //int range_end = (range_begin + range < n_samples)? range_begin + range : n_samples;
     
  
  if(threadIdx.x == 0){
    int last_block_offset = blockIdx.x * blockDim.x * MAX_NUM_LABELS;
    for(int l = 0; l < MAX_NUM_LABELS; ++l)
      last_block_count[l] = label_count[last_block_offset + l];
    
  }
  __syncthreads();
  
  if(threadIdx.x !=  blockDim.x - 1 && thread_id != n_active_threads - 1)
    for(int l = 0; l < MAX_NUM_LABELS; ++l)
      label_count[(thread_id + 1) * MAX_NUM_LABELS + l] += last_block_count[l];
}
