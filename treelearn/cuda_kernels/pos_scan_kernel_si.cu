#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define IDX_DATA_TYPE %s
#define THREADS_PER_BLOCK %s

__global__ void pos_scan( 
                        uint8_t *mark_table,
                        IDX_DATA_TYPE *pos_table,     
                        IDX_DATA_TYPE *sorted_indices,
                        int n_active_threads,
                        int range, 
                        int n_samples,
                        int stride){

    int indices_offset = blockIdx.x * stride;
    int pos_offset = THREADS_PER_BLOCK * blockIdx.x;
    int range_begin = threadIdx.x * range;
    int range_end = (range_begin +  range < n_samples)? range_begin + range : n_samples;
    int reg_pos = 0;

    __shared__ IDX_DATA_TYPE shared_pos_table[THREADS_PER_BLOCK];

    if(threadIdx.x >= n_active_threads - 1)
      return;

    for(int i = range_begin; i < range_end; ++i)
      reg_pos += mark_table[sorted_indices[indices_offset + i]];
 
    shared_pos_table[threadIdx.x] = reg_pos;    
    __syncthreads();
    
    if(threadIdx.x == 0){
      pos_table[pos_offset] = 0;
      for(int i = 1; i < n_active_threads - 1; ++i)
        shared_pos_table[i] += shared_pos_table[i-1];
    }

    __syncthreads();

    pos_table[pos_offset + threadIdx.x + 1] = shared_pos_table[threadIdx.x];    
}
