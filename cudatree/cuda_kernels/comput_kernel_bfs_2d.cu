#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define THREADS_PER_BLOCK %d
#define MAX_NUM_LABELS %d
#define SAMPLE_DATA_TYPE %s
#define LABEL_DATA_TYPE %s
#define COUNT_DATA_TYPE %s
#define IDX_DATA_TYPE %s
#define DEBUG %d

#include "common_func.cu"

__global__ void compute(
                        SAMPLE_DATA_TYPE *samples, 
                        LABEL_DATA_TYPE *labels,
                        IDX_DATA_TYPE *sorted_indices_1,
                        IDX_DATA_TYPE *sorted_indices_2,
                        uint32_t *begin_stop_idx,
                        uint8_t *si_idx,
                        COUNT_DATA_TYPE *label_total,
                        uint16_t *subset_indices, 
                        float *imp_min, 
                        COUNT_DATA_TYPE *split,
                        uint16_t *min_feature_idx,
                        int max_features,
                        int n_features,
                        int stride){

  IDX_DATA_TYPE* p_sorted_indices;
  IDX_DATA_TYPE reg_start_idx;
  IDX_DATA_TYPE reg_stop_idx;
 
  __shared__ float shared_count_total[MAX_NUM_LABELS];
  __shared__ float shared_label_count[MAX_NUM_LABELS];
  __shared__ LABEL_DATA_TYPE shared_labels[THREADS_PER_BLOCK];
  __shared__ SAMPLE_DATA_TYPE shared_samples[THREADS_PER_BLOCK];
  

  uint16_t bidx = blockIdx.x;
  uint16_t tidx = threadIdx.x;
  
  reg_start_idx = begin_stop_idx[2 * bidx];
  reg_stop_idx = begin_stop_idx[2 * bidx + 1];
  
  uint16_t step = blockDim.x - 1;
  float reg_min_left = 2.0;
  float reg_min_right = 2.0;
  IDX_DATA_TYPE reg_min_split = reg_stop_idx;

  for(uint16_t i = tidx; i < MAX_NUM_LABELS; i += blockDim.x)
    shared_count_total[i] = label_total[bidx * MAX_NUM_LABELS + i];
 
  uint8_t reg_si_idx = si_idx[bidx];
  if(reg_si_idx == 0)
    p_sorted_indices = sorted_indices_1;
  else
    p_sorted_indices = sorted_indices_2;
 

  uint16_t f = blockIdx.y;  
  uint16_t feature_idx = subset_indices[f];
  uint32_t offset = feature_idx * stride;

  //Reset shared_label_count array.
  for(uint16_t t = tidx; t < MAX_NUM_LABELS; t += blockDim.x)
    shared_label_count[t] = 0.0;
  
  __syncthreads();

  for(IDX_DATA_TYPE i = reg_start_idx; i < reg_stop_idx - 1; i += step){
    IDX_DATA_TYPE idx = i + tidx;

    if(idx < reg_stop_idx){
      shared_labels[tidx] = labels[p_sorted_indices[offset + idx]];
      shared_samples[tidx] = samples[offset + p_sorted_indices[offset + idx]];
    }

    __syncthreads();

    if(tidx == 0){
      IDX_DATA_TYPE stop_pos = (i + step < reg_stop_idx - 1)? step : reg_stop_idx - 1 - i;
      
      for(IDX_DATA_TYPE t = 0; t < stop_pos; ++t){
        shared_label_count[shared_labels[t]]++;
        if(shared_samples[t] == shared_samples[t + 1])
          continue;
        
        IDX_DATA_TYPE n_left =  i + t - reg_start_idx + 1;
        IDX_DATA_TYPE n_right = reg_stop_idx - reg_start_idx - n_left; 

        float left = calc_imp_left(shared_label_count, n_left) * n_left / (reg_stop_idx - reg_start_idx);
        float right = calc_imp_right(shared_label_count, shared_count_total, n_right) * 
          n_right / (reg_stop_idx - reg_start_idx);
        
        if(left + right < reg_min_left + reg_min_right){
          reg_min_left = left;
          reg_min_right = right;
          reg_min_split = i + t;
        }
      }  
    }
    __syncthreads();
  }

  if(tidx == 0){
    offset = gridDim.y * blockIdx.x + blockIdx.y; 
    imp_min[offset * 2] = reg_min_left;
    imp_min[offset * 2 + 1] = reg_min_right;
    split[offset] = reg_min_split;
    min_feature_idx[offset] = feature_idx;
  }
}


