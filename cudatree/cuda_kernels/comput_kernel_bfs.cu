#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define THREADS_PER_BLOCK %d
#define MAX_NUM_LABELS %d
#define SAMPLE_DATA_TYPE %s
#define LABEL_DATA_TYPE %s
#define COUNT_DATA_TYPE %s
#define IDX_DATA_TYPE %s

__device__  float calc_imp_right(COUNT_DATA_TYPE* label_previous, COUNT_DATA_TYPE* label_now, COUNT_DATA_TYPE total_size){
  uint32_t sum = 0.0; 
  for(LABEL_DATA_TYPE i = 0; i < MAX_NUM_LABELS; ++i){
    uint32_t count = label_now[i] - label_previous[i];
    sum += count * count;
  }
 
  uint32_t denom =  total_size * total_size;
  return 1.0 - (float(sum) / denom); 
}

__device__  float calc_imp_left(COUNT_DATA_TYPE* label_now, COUNT_DATA_TYPE total_size){
  uint32_t sum = 0.0;
  for(LABEL_DATA_TYPE i = 0; i < MAX_NUM_LABELS; ++i){
    uint32_t count = label_now[i];
    sum += count * count;
  }
  
  uint32_t denom =  total_size * total_size;
  return 1.0 - ((float)sum / denom); 
}


__global__ void compute(
                        SAMPLE_DATA_TYPE *samples, 
                        LABEL_DATA_TYPE *labels,
                        IDX_DATA_TYPE *sorted_indices_1,
                        IDX_DATA_TYPE *sorted_indices_2,
                        uint32_t *begin_stop_idx,
                        uint8_t *si_idx,
                        COUNT_DATA_TYPE *label_total,
                        IDX_DATA_TYPE *subset_indices, 
                        float *imp_min, 
                        COUNT_DATA_TYPE *split,
                        uint16_t *min_feature_idx,
                        int max_features,
                        int stride){

  IDX_DATA_TYPE* p_sorted_indices;
  IDX_DATA_TYPE reg_start_idx;
  IDX_DATA_TYPE reg_stop_idx;
  __shared__ IDX_DATA_TYPE shared_count_total[MAX_NUM_LABELS];
  __shared__ IDX_DATA_TYPE shared_label_count[MAX_NUM_LABELS];
  __shared__ LABEL_DATA_TYPE shared_labels[THREADS_PER_BLOCK];
  __shared__ SAMPLE_DATA_TYPE shared_samples[THREADS_PER_BLOCK + 1];
 
  reg_start_idx = begin_stop_idx[2 * blockIdx.x];
  reg_stop_idx = begin_stop_idx[2 * blockIdx.x + 1];
  
  float reg_min_left = 2.0;
  float reg_min_right = 2.0;
  uint16_t reg_min_fidx = 0;
  IDX_DATA_TYPE reg_min_split = reg_stop_idx;

  for(uint16_t i = threadIdx.x; i < MAX_NUM_LABELS; i += blockDim.x)
    shared_count_total[i] = label_total[blockIdx.x * MAX_NUM_LABELS + i];
 
  
  uint8_t reg_si_idx = si_idx[blockIdx.x];
  if(reg_si_idx == 0)
    p_sorted_indices = sorted_indices_1;
  else
    p_sorted_indices = sorted_indices_2;
 
  for(uint16_t f = 0; f < max_features; ++f){
    //Reset shared_label_count array.
    for(uint16_t t = threadIdx.x; t < MAX_NUM_LABELS; t += blockDim.x)
      shared_label_count[t] = 0;
    
    __syncthreads();

    uint16_t feature_idx = subset_indices[blockIdx.x * max_features + f];
    uint32_t offset = feature_idx * stride;

    for(IDX_DATA_TYPE i = reg_start_idx; i < reg_stop_idx - 1; i += blockDim.x){
      IDX_DATA_TYPE idx = i + threadIdx.x;

      if(idx < reg_stop_idx - 1){
        shared_labels[threadIdx.x] = labels[p_sorted_indices[offset + idx]];
        shared_samples[threadIdx.x] = samples[offset + p_sorted_indices[offset + idx]];
      }

      __syncthreads();

      if(threadIdx.x == 0){
        IDX_DATA_TYPE stop_pos = (i + blockDim.x < reg_stop_idx - 1)? i + blockDim.x : reg_stop_idx - 1;
        shared_samples[stop_pos - i] = samples[offset + p_sorted_indices[offset + stop_pos]];
        
        for(IDX_DATA_TYPE t = 0; t < stop_pos - i; ++t){
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
            reg_min_fidx = feature_idx;
            reg_min_split = i + t;
          }
        }  
      }
      __syncthreads();
    }
  
    __syncthreads();
  } 
  
  if(threadIdx.x == 0){
    imp_min[2 * blockIdx.x] = reg_min_left;
    imp_min[2 * blockIdx.x + 1] = reg_min_right;
    split[blockIdx.x] = reg_min_split;
    min_feature_idx[blockIdx.x] = reg_min_fidx;
  }
}


