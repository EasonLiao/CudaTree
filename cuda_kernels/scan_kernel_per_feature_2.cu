#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define MAX_NUM_LABELS %d
#define COUNT_DATA_TYPE %s

__global__ void prefix_scan_2( 
                        COUNT_DATA_TYPE *label_count,
                        int n_active_threads,
                        int threads_per_block,
                        int num_blocks,
                        int n_samples
                        ){
    for(int i = 1; i < num_blocks; ++i){
       int cur_thread_idx = (i + 1) * threads_per_block;
       cur_thread_idx = (cur_thread_idx > n_active_threads)? n_active_threads : cur_thread_idx;
       int cur_label_idx = cur_thread_idx * MAX_NUM_LABELS;
       int last_label_idx = i * threads_per_block * MAX_NUM_LABELS;

       for(int l = 0; l < MAX_NUM_LABELS; ++l)
         label_count[cur_label_idx + l] += label_count[last_label_idx + l];
    }    
}
