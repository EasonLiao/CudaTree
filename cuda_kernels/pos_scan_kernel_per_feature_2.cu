#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define IDX_DATA_TYPE %s

__global__ void pos_scan_2( 
                        IDX_DATA_TYPE* pos_table,
                        int n_active_threads,
                        int threads_per_block,
                        int num_blocks,
                        int n_samples
                        ){

    for(int i = 1; i < num_blocks - 1; ++i){
       int cur_thread_idx = (i + 1) * threads_per_block;
       
       int cur_table_idx = cur_thread_idx * 2;
       int last_label_idx = i * threads_per_block * 2;

       pos_table[cur_table_idx] += pos_table[last_label_idx];
       pos_table[cur_table_idx + 1] += pos_table[last_label_idx + 1];
    }    
}
