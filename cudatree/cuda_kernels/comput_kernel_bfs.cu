#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define THREADS_PER_BLOCK %d
#define MAX_NUM_LABELS %d
#define SAMPLE_DATA_TYPE %s
#define LABEL_DATA_TYPE %s
#define COUNT_DATA_TYPE %s
#define IDX_DATA_TYPE %s

__device__  float calc_imp_right(COUNT_DATA_TYPE label_previous[MAX_NUM_LABELS], COUNT_DATA_TYPE label_now[MAX_NUM_LABELS], int total_size){
  float sum = 0.0; 
  for(int i = 0; i < MAX_NUM_LABELS; ++i){
    float count = label_now[i] - label_previous[i];
    sum += count * count;
  }
 
  //printf("sum: %%f, denom: %%d\n", sum, total_size);
  float denom = ((float) total_size) * total_size;
  return 1.0 - (sum / denom); 
}

__device__  float calc_imp_left(COUNT_DATA_TYPE label_now[MAX_NUM_LABELS], int total_size){
  float sum = 0.0;
  for(int i = 0; i < MAX_NUM_LABELS; ++i){
    float count = label_now[i];
    sum += count * count;
  }
  
  float denom = ((float) total_size) * total_size;
  return 1.0 - (sum / denom); 
}


__global__ void compute(
                        SAMPLE_DATA_TYPE *samples, 
                        LABEL_DATA_TYPE *labels,
                        IDX_DATA_TYPE *sorted_indices_1,
                        IDX_DATA_TYPE *sorted_indices_2,
                        IDX_DATA_TYPE *begin_stop_idx,
                        uint8_t *si_idx,
                        COUNT_DATA_TYPE *label_total,
                        IDX_DATA_TYPE *subset_indices, 
                        float *imp_min, 
                        COUNT_DATA_TYPE *split,
                        uint16_t *min_feature_idx,
                        int max_features,
                        int stride){

  __shared__ IDX_DATA_TYPE* p_sorted_indices;
  __shared__ IDX_DATA_TYPE shared_start_idx;
  __shared__ IDX_DATA_TYPE shared_stop_idx;
  __shared__ IDX_DATA_TYPE shared_count_total[MAX_NUM_LABELS];
  __shared__ IDX_DATA_TYPE shared_label_count[MAX_NUM_LABELS];
  __shared__ LABEL_DATA_TYPE shared_labels[THREADS_PER_BLOCK];
  __shared__ SAMPLE_DATA_TYPE shared_samples[THREADS_PER_BLOCK + 1];

  float reg_min_left = 2.0;
  float reg_min_right = 2.0;
  uint16_t reg_min_fidx = 0;
  IDX_DATA_TYPE reg_min_split = 0;


  for(int i = threadIdx.x; i < MAX_NUM_LABELS; i += blockDim.x)
    shared_count_total[i] = label_total[blockIdx.x * MAX_NUM_LABELS + i];
 

  if(threadIdx.x == 0){
    shared_start_idx = begin_stop_idx[2 * blockIdx.x];
    shared_stop_idx = begin_stop_idx[2 * blockIdx.x + 1];
    
    uint8_t reg_si_idx = si_idx[blockIdx.x];
    if(reg_si_idx == 0)
      p_sorted_indices = sorted_indices_1;
    else
      p_sorted_indices = sorted_indices_2;
  }

  __syncthreads();
  
  for(int f = 0; f < max_features; ++f){
    //Reset shared_label_count array.
    for(int t = threadIdx.x; t < MAX_NUM_LABELS; t += blockDim.x)
      shared_label_count[t] = 0;
    
    uint16_t feature_idx = subset_indices[blockIdx.x * max_features + f];
    int offset = feature_idx * stride;

    for(int i = shared_start_idx + threadIdx.x; i < shared_stop_idx - 1; i += blockDim.x){
      shared_labels[threadIdx.x] = labels[p_sorted_indices[offset + i]];
      shared_samples[threadIdx.x] = samples[offset + p_sorted_indices[offset + i]];
      
      if(threadIdx.x == 0){
        uint16_t stop_pos = (i + blockDim.x < shared_stop_idx - 1)? i + blockDim.x : shared_stop_idx - 1;
        shared_samples[stop_pos - i] = samples[offset + p_sorted_indices[offset + stop_pos]];
        
        for(int t = 0; t < stop_pos - i; ++t){
          shared_label_count[shared_labels[t]]++;
          if(shared_samples[t] == shared_samples[t + 1])
            continue;
          
          int n_left =  i + t - shared_start_idx + 1;
          int n_right = shared_stop_idx - shared_start_idx - n_left; 

          float left = calc_imp_left(shared_label_count, n_left) * n_left / (shared_stop_idx - shared_start_idx);
          float right = calc_imp_right(shared_label_count, shared_count_total, n_right) * 
            n_right / (shared_stop_idx - shared_start_idx);
          
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


