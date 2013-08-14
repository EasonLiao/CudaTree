#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define SAMPLE_DATA_TYPE %s
#define LABEL_DATA_TYPE %s
#define IDX_DATA_TYPE %s


__global__ void reshuffle(uint8_t* mark_table,
                          LABEL_DATA_TYPE* sorted_labels,
                          IDX_DATA_TYPE* sorted_indices,
                          SAMPLE_DATA_TYPE* sorted_samples,
                          LABEL_DATA_TYPE* sorted_labels_out,
                          IDX_DATA_TYPE* sorted_indices_out,
                          SAMPLE_DATA_TYPE* sorted_samples_out,
                          int n_samples,
                          int split_idx
                          ){

  int left_start = 0;
  int right_start = split_idx + 1;

  for(int i = 0; i < n_samples; ++i){
    if(mark_table[sorted_indices[i]] == 0){
      sorted_indices_out[left_start] = sorted_indices[i];
      sorted_labels_out[left_start] = sorted_labels[i];
      sorted_samples_out[left_start] = sorted_samples[i];
      left_start++;
    }
    else{
      sorted_indices_out[right_start] = sorted_indices[i];
      sorted_labels_out[right_start] = sorted_labels[i]; 
      sorted_samples_out[right_start] = sorted_samples[i];
      right_start++;
    }
  }
}






