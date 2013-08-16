#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define IDX_DATA_TYPE %s
#define THREADS_PER_BLOCK %s

__global__ void pos_scan_1( 
                        uint8_t *mark_table,
                        IDX_DATA_TYPE *pos_table,     
                        IDX_DATA_TYPE *sorted_indices,
                        int n_active_threads,
                        int range, 
                        int n_samples){
  
    int thread_offset = blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x + thread_offset;
    int range_begin = thread_id * range;
    int range_end = (range_begin +  range < n_samples)? range_begin + range : n_samples;
    __shared__ IDX_DATA_TYPE shared_pos_table[THREADS_PER_BLOCK][2];

    if(thread_id >= n_active_threads - 1)
      return;
    
    if(thread_id == 0){
      pos_table[0] = 0;
      pos_table[1] = 0;
    }

    shared_pos_table[threadIdx.x][0] = 0;
    shared_pos_table[threadIdx.x][1] = 0;

    for(int i = range_begin; i < range_end; ++i){
      if(mark_table[sorted_indices[i]] == 0)
        shared_pos_table[threadIdx.x][0]++;
      else 
        shared_pos_table[threadIdx.x][1]++;
    }
  
    __syncthreads();

    if(threadIdx.x == 0)
      for(int i = 1; i < blockDim.x; ++i){
        shared_pos_table[i][0] += shared_pos_table[i - 1][0];
        shared_pos_table[i][1] += shared_pos_table[i - 1][1];
      }
   
    __syncthreads();
    
    pos_table[(thread_id + 1) * 2] = shared_pos_table[threadIdx.x][0];
    pos_table[(thread_id + 1) * 2 + 1] = shared_pos_table[threadIdx.x][1]; 
}
