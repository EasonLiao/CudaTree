#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define SAMPLE_DATA_TYPE %s
#define LABEL_DATA_TYPE %s
#define IDX_DATA_TYPE %s


__global__ void reshuffle(IDX_DATA_TYPE* mark_table,
                          LABEL_DATA_TYPE* sorted_labels,
                          IDX_DATA_TYPE* sorted_indices,
                          SAMPLE_DATA_TYPE* sorted_samples,
                          LABEL_DATA_TYPE* sorted_labels_out,
                          IDX_DATA_TYPE* sorted_indices_out,
                          SAMPLE_DATA_TYPE* sorted_samples_out,
                          int n_samples,
                          int split_idx,
                          int stride
                          ){
  int offset = blockIdx.x * stride;
  int left_start = 0;
  int right_start = split_idx + 1;

  for(int i = 0; i < n_samples; ++i){
    if(mark_table[sorted_indices[offset + i]] == 0){
      sorted_indices_out[offset + left_start] = sorted_indices[offset + i];
      sorted_labels_out[offset + left_start] = sorted_labels[offset + i];
      sorted_samples_out[offset + left_start] = sorted_samples[offset + i];
      left_start++;
    }
    else{
      sorted_indices_out[offset + right_start] = sorted_indices[offset + i];
      sorted_labels_out[offset + right_start] = sorted_labels[offset + i]; 
      sorted_samples_out[offset + right_start] = sorted_samples[offset + i];
      right_start++;
    }
  }
}






