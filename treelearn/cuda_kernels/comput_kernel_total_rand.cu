//Add parallel reduction to find mininum impurity based on kernel_2.cu
#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define THREADS_PER_BLOCK %d
#define MAX_NUM_LABELS %d
#define SAMPLE_DATA_TYPE %s
#define LABEL_DATA_TYPE %s
#define COUNT_DATA_TYPE %s
#define IDX_DATA_TYPE %s

//texture<COUNT_DATA_TYPE> tex_label_total;

__device__  float calc_imp_right(COUNT_DATA_TYPE label_previous[MAX_NUM_LABELS], COUNT_DATA_TYPE label_now[MAX_NUM_LABELS], int total_size){
  float sum = 0.0; 
  for(int i = 0; i < MAX_NUM_LABELS; ++i){
    float count = label_now[i] - label_previous[i];
    //float count = tex1Dfetch(tex_label_total, i) - label_previous[i];
    sum += count * count;
  }

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

__global__ void compute(IDX_DATA_TYPE *sorted_indices,
                        SAMPLE_DATA_TYPE *samples, 
                        LABEL_DATA_TYPE *labels,
                        float *imp_left, 
                        float *imp_right, 
                        COUNT_DATA_TYPE *label_total,
                        COUNT_DATA_TYPE *split, 
                        IDX_DATA_TYPE *subset_indices,
                        int n_samples, 
                        int stride){
  int offset = subset_indices[blockIdx.x] * stride;
  int stop_pos;
  float reg_imp_right = 2.0;
  float reg_imp_left = 2.0;
  COUNT_DATA_TYPE reg_min_split = 0;

  __shared__ COUNT_DATA_TYPE shared_count[MAX_NUM_LABELS];
  __shared__ LABEL_DATA_TYPE shared_labels[THREADS_PER_BLOCK];
  __shared__ COUNT_DATA_TYPE shared_count_total[MAX_NUM_LABELS];
  __shared__ SAMPLE_DATA_TYPE shared_samples[THREADS_PER_BLOCK];
  

  for(int i = threadIdx.x; i < MAX_NUM_LABELS; i += blockDim.x){   
      shared_count[i] = 0;
      shared_count_total[i] = label_total[i];
  }
  
  
  for(int i = threadIdx.x; i < n_samples; i += blockDim.x){ 
    int idx = sorted_indices[offset + i];
    shared_labels[threadIdx.x] = labels[idx]; 
    shared_samples[threadIdx.x] = samples[offset + idx];

    __syncthreads();
     
    if(threadIdx.x == 0){
      stop_pos = (i + blockDim.x < n_samples - 1)? blockDim.x : n_samples - i - 1;
      
        for(int t = 0; t < stop_pos; ++t){
          shared_count[shared_labels[t]]++;
                    
          if(t != stop_pos - 1){
            if(shared_samples[t] == shared_samples[t + 1])
              continue;
          }
          else if(shared_samples[t] == samples[offset + sorted_indices[offset + stop_pos + i]])
            continue;
          
          float imp_left = (i + t + 1) / float(n_samples) * calc_imp_left(shared_count, i + 1 + t);
          float imp_right = (n_samples - i - 1- t) / float(n_samples) * calc_imp_right(shared_count, shared_count_total, n_samples - i - 1 - t);
          //float imp_right = (n_samples - i - 1- t) / float(n_samples) * calc_imp_right(shared_count, label_total, n_samples - i - 1 - t);
          
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
    split[blockIdx.x] = reg_min_split;
    imp_left[blockIdx.x] = reg_imp_left;
    imp_right[blockIdx.x] = reg_imp_right;
  }
}
