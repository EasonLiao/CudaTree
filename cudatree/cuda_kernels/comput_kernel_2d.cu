#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define THREADS_PER_BLOCK %d
#define MAX_NUM_LABELS %d
#define SAMPLE_DATA_TYPE %s
#define LABEL_DATA_TYPE %s
#define COUNT_DATA_TYPE %s
#define IDX_DATA_TYPE %s
#define MAX_BLOCK_PER_FEATURE %d

__device__  float calc_imp_right(COUNT_DATA_TYPE label_previous[MAX_NUM_LABELS], COUNT_DATA_TYPE label_now[MAX_NUM_LABELS], int total_size){
  float sum = 0.0; 
  for(uint16_t i = 0; i < MAX_NUM_LABELS; ++i){
    float count = label_now[i] - label_previous[i];
    sum += count * count;
  }

  float denom = ((float) total_size) * total_size;
  return 1.0 - (sum / denom); 
}

__device__  float calc_imp_left(COUNT_DATA_TYPE label_now[MAX_NUM_LABELS], int total_size){
  float sum = 0.0;
  for(uint16_t i = 0; i < MAX_NUM_LABELS; ++i){
    float count = label_now[i];
    sum += count * count;
  }
  
  float denom = ((float) total_size) * total_size;
  return 1.0 - (sum / denom); 
}

__global__ void compute(IDX_DATA_TYPE *sorted_indices,
                        SAMPLE_DATA_TYPE *samples, 
                        LABEL_DATA_TYPE *labels,
                        float *impurity, 
                        COUNT_DATA_TYPE *label_total,
                        COUNT_DATA_TYPE *split, 
                        IDX_DATA_TYPE *subset_indices,
                        int n_range,
                        int n_samples, 
                        int stride){
  
  uint32_t offset = subset_indices[blockIdx.x] * stride;
  float reg_imp_right = 2.0;
  float reg_imp_left = 2.0;
  COUNT_DATA_TYPE reg_min_split = 0;

  __shared__ COUNT_DATA_TYPE shared_count[MAX_NUM_LABELS];
  __shared__ LABEL_DATA_TYPE shared_labels[THREADS_PER_BLOCK];
  __shared__ COUNT_DATA_TYPE shared_count_total[MAX_NUM_LABELS];
  __shared__ SAMPLE_DATA_TYPE shared_samples[THREADS_PER_BLOCK];
  
  uint32_t cur_offset = blockIdx.x * (MAX_BLOCK_PER_FEATURE + 1) * MAX_NUM_LABELS + blockIdx.y * MAX_NUM_LABELS;
  uint32_t last_offset = int(ceil(float(n_samples) / n_range)) * MAX_NUM_LABELS;

  for(uint16_t i = threadIdx.x; i < MAX_NUM_LABELS; i += blockDim.x){   
      shared_count[i] = label_total[cur_offset + i];
      shared_count_total[i] = label_total[last_offset + i];
  }
  
  IDX_DATA_TYPE stop_pos = ((blockIdx.y + 1) * n_range < n_samples)? (blockIdx.y + 1) * n_range : n_samples;

  for(IDX_DATA_TYPE i = blockIdx.y * n_range + threadIdx.x; i < stop_pos; i += blockDim.x){ 
    IDX_DATA_TYPE idx = sorted_indices[offset + i];
    shared_labels[threadIdx.x] = labels[idx]; 
    shared_samples[threadIdx.x] = samples[offset + idx];

    __syncthreads();
     
    if(threadIdx.x == 0){
      IDX_DATA_TYPE end_pos = (i + blockDim.x < stop_pos - 1)? blockDim.x : stop_pos - i - 1;
      
        for(IDX_DATA_TYPE t = 0; t < end_pos; ++t){
          shared_count[shared_labels[t]]++;
                    
          if(t != end_pos - 1){
            if(shared_samples[t] == shared_samples[t + 1])
              continue;
          }
          else if(shared_samples[t] == samples[offset + sorted_indices[offset + end_pos + i]])
            continue;
          
          float imp_left = calc_imp_left(shared_count, i + 1 + t) * (i + t + 1) / n_samples;
          float imp_right = calc_imp_right(shared_count, shared_count_total, n_samples - i - 1 - t) *
            (n_samples - i - 1 - t) / n_samples;
          
          if(imp_left + imp_right < reg_imp_right + reg_imp_left){
            reg_imp_left = imp_left;
            reg_imp_right = imp_right;
            reg_min_split = i + t;
          }  
        }
    }    
    __syncthreads();
  }
    
  if(threadIdx.x == 0){
    impurity[blockIdx.x * MAX_BLOCK_PER_FEATURE * 2 + 2 * blockIdx.y] = reg_imp_left;
    impurity[blockIdx.x * MAX_BLOCK_PER_FEATURE * 2 + 2 * blockIdx.y] = reg_imp_right;
    split[blockIdx.x * MAX_BLOCK_PER_FEATURE + blockIdx.y] = reg_min_split;
  }
}
