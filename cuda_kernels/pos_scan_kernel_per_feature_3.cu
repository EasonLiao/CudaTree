#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define IDX_DATA_TYPE %s

__global__ void pos_scan_3( 
                        uint8_t *mark_table,
                        IDX_DATA_TYPE *pos_table,
                        int n_active_threads,
                        int range,
                        int n_samples
                        ){

  if(blockIdx.x == 0)
    return;

  __shared__ IDX_DATA_TYPE last_block_count[2];
  int thread_offset = blockIdx.x * blockDim.x;  
  int thread_id = threadIdx.x + thread_offset;
  int range_begin = thread_id * range;

  if(thread_id >= n_active_threads - 1)
    return;

  //int range_end = (range_begin + range < n_samples)? range_begin + range : n_samples;
      
  if(threadIdx.x == 0){
    int last_block_offset = blockIdx.x * blockDim.x * 2;
    last_block_count[0] = pos_table[last_block_offset];
    last_block_count[1] = pos_table[last_block_offset + 1];
  }
  __syncthreads();
  
  if(threadIdx.x !=  blockDim.x - 1 || blockIdx.x == gridDim.x - 1){
    pos_table[(thread_id + 1) * 2] += last_block_count[0];
    pos_table[(thread_id + 1) * 2 + 1] += last_block_count[1];
  } 
}
