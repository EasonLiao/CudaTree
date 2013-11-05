#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define THREADS_PER_BLOCK %d
#define IDX_DATA_TYPE %s
#define DEBUG %d

__global__ void reduce(float *imp_min_2d, 
                        IDX_DATA_TYPE *split_2d,
                        uint16_t *min_feature_idx_2d,
                        float *imp_min,
                        IDX_DATA_TYPE *split,
                        uint16_t *min_feature,
                        int max_features){
  
  uint16_t offset = blockIdx.x * max_features;
  IDX_DATA_TYPE reg_min_split;
  float reg_min_left = 4.0;
  float reg_min_right = 4.0;
  uint16_t reg_min_fidx = 0;
  
  for(int i = 0; i < max_features; ++i){
    float left = imp_min_2d[2 * (offset + i)];
    float right = imp_min_2d[2 * (offset + i) + 1];
    if(reg_min_left + reg_min_right > left + right){
      reg_min_left = left;
      reg_min_right = right;
      reg_min_fidx = min_feature_idx_2d[offset + i];
      reg_min_split = split_2d[offset + i];
    }
  }
  
  split[blockIdx.x] = reg_min_split;
  min_feature[blockIdx.x] = reg_min_fidx;
  imp_min[2 * blockIdx.x] = reg_min_left;
  imp_min[2 * blockIdx.x + 1] = reg_min_right;
}


