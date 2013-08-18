#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define SAMPLE_DATA_TYPE %s
#define LABEL_DATA_TYPE %s
#define IDX_DATA_TYPE %s
#define THREADS_PER_BLOCK %s

__global__ void reshuffle(uint8_t* mark_table,
                          LABEL_DATA_TYPE* sorted_labels,
                          IDX_DATA_TYPE* sorted_indices,
                          SAMPLE_DATA_TYPE* sorted_samples,
                          LABEL_DATA_TYPE* sorted_labels_out,
                          IDX_DATA_TYPE* sorted_indices_out,
                          SAMPLE_DATA_TYPE* sorted_samples_out,
                          IDX_DATA_TYPE *pos_table,
                          int n_active_threads,
                          int range,
                          int n_samples,
                          int split_idx
                          ){
    int thread_offset = blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x + thread_offset;
    int range_begin = thread_id * range;
    int range_end = (range_begin +  range < n_samples)? range_begin + range : n_samples;
    
    if(thread_id >= n_active_threads)
      return;

    __shared__ IDX_DATA_TYPE shared_pos_table[THREADS_PER_BLOCK][2];
    
    //int pos_0 = pos_table[thread_id * 2];
    //int pos_1 = pos_table[thread_id * 2 + 1];

    int right_start = split_idx + 1;
    int pos;

    shared_pos_table[threadIdx.x][0] = pos_table[thread_id * 2 ];
    shared_pos_table[threadIdx.x][1] = pos_table[thread_id * 2 + 1] + right_start;
    
    for(int i = range_begin; i < range_end; ++i){
      int side = mark_table[sorted_indices[i]];
      pos = shared_pos_table[threadIdx.x][side];

      sorted_indices_out[pos] = sorted_indices[i];
      sorted_samples_out[pos] = sorted_samples[i];
      sorted_labels_out[pos] = sorted_labels[i]; 
      /*
      if(side == 0){
        sorted_indices_out[pos_0] = sorted_indices[i];
        sorted_samples_out[pos_0] = sorted_samples[i];
        sorted_labels_out[pos_0] = sorted_labels[i];
      } 
      else{
        sorted_indices_out[pos_1 + right_start] = sorted_indices[i];  
        sorted_samples_out[pos_1 + right_start] = sorted_samples[i];
        sorted_labels_out[pos_1 + right_start] = sorted_labels[i];
      }
      */
      shared_pos_table[threadIdx.x][side]++;
    }
}






