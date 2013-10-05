#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define IDX_DATA_TYPE %s
#define SAMPLE_DATA_TYPE %s
#define LABEL_DATA_TYPE %s

#define WARP_SIZE 32
#define WARP_MASK 0x1f
#define WARPS_PER_BLOCK 16

__global__ void predict(uint32_t *left_child_arr,
                        uint32_t *right_child_arr,
                        uint16_t *feature_array,
                        float *threshold_array,
                        LABEL_DATA_TYPE *value_array,
                        SAMPLE_DATA_TYPE *predict_array,
                        LABEL_DATA_TYPE *predict_res,
                        int n_features,
                        int n_predict
                        ) {
  /* 
    Predict new labels for previously unseen data points. 
    Inputs: 
      - left_child_arr : what's the index of the left child node? 
      - right_child_arr : what's the index of the right child node? 
      - threshold_array : what's the threshold of a given internal node?
      - value_array : what's the value of a give leaf?
      - predict_array : the input samples need to be predicted.
      - n_features : the number of features the samples have.
      - n_predict : number of samples need to be predicted.
    
    Outputs:
      - predict_res : the result of the prediction
 */

  int lane_id = threadIdx.x & WARP_MASK;
  int warp_id = threadIdx.x / WARP_SIZE;

  if(lane_id != 0)
    return;

  int predict_idx = blockIdx.x * gridDim.y * WARPS_PER_BLOCK + blockIdx.y * WARPS_PER_BLOCK + warp_id;
  int offset = predict_idx * n_features;
  int idx = 0; 
  
  if(predict_idx >= n_predict)
    return;

  while(true){
    uint32_t left_idx = left_child_arr[idx];
    uint32_t right_idx = right_child_arr[idx];
    
    if(left_idx == 0 || right_idx == 0){
      //Means it's on leaf.
      predict_res[predict_idx] = value_array[idx];
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

