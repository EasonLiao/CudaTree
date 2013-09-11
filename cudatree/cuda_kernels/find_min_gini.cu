#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define COUNT_DATA_TYPE %s
#define THREADS_PER_BLOCK %d

__global__ void find_min_imp(
                          float* imp_left,
                          float* imp_right,
                          COUNT_DATA_TYPE *min_split,
                          int max_features
                          ){
  __shared__ uint16_t min_tid;
  __shared__ float shared_imp_total[THREADS_PER_BLOCK];
  float reg_imp_left;
  float reg_imp_right;
  COUNT_DATA_TYPE reg_min_idx = 0;
  int reg_min_feature_idx;

  reg_imp_left = 2.0;
  reg_imp_right = 2.0;

  for(uint16_t i = threadIdx.x; i < max_features; i += blockDim.x){
    float left = imp_left[i]; 
    float right = imp_right[i]; 
    COUNT_DATA_TYPE idx = min_split[i];

    if(left + right < reg_imp_left + reg_imp_right){
      reg_imp_left = left;
      reg_imp_right = right;
      reg_min_idx = idx;
      reg_min_feature_idx = i;
    }
  }
  

  shared_imp_total[threadIdx.x] = reg_imp_left + reg_imp_right;
  
  __syncthreads();
  
  if(threadIdx.x == 0){
    float min_imp = 4.0;
    for(uint16_t i = 0; i < blockDim.x; ++i)
      if(shared_imp_total[i] < min_imp){
        min_tid = i;
        min_imp = shared_imp_total[i];
      }
  }
  __syncthreads();
  
  if(threadIdx.x == min_tid){
    imp_left[0] = reg_imp_left;
    imp_left[1] = reg_imp_right;
    imp_left[2] = reg_min_idx;
    imp_left[3] = reg_min_feature_idx;
  }

}









