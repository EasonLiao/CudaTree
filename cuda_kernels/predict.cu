#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define IDX_DATA_TYPE %s
#define SAMPLE_DATA_TYPE %s
#define LABEL_DATA_TYPE %s

__global__ void predict(IDX_DATA_TYPE *left_child_arr,
                        IDX_DATA_TYPE *right_child_arr,
                        uint16_t *feature_array,
                        float *threshold_array,
                        LABEL_DATA_TYPE *value_array,
                        SAMPLE_DATA_TYPE *predict_array,
                        LABEL_DATA_TYPE *predict_res,
                        int n_features,
                        int n_nodes
                        ){
  int offset = blockIdx.x * n_features;
  int idx = 0; 
  
  while(true){
    IDX_DATA_TYPE left_idx = left_child_arr[idx];
    IDX_DATA_TYPE right_idx = right_child_arr[idx];
    
    if(left_idx == 0 || right_idx == 0){
      //Means it's on leaf.
      predict_res[blockIdx.x] = value_array[idx];
      return;
    }
    
    float threshold = threshold_array[idx]; 
    uint16_t feature_idx = feature_array[idx];
    
    if(predict_array[offset + feature_idx] < threshold)
      idx = left_idx;
    else 
      idx = right_idx;
  }

}









