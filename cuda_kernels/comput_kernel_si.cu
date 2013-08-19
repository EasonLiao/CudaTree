//Add parallel reduction to find mininum impurity based on kernel_2.cu
#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define THREADS_PER_BLOCK %s
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


__global__ void compute(SAMPLE_DATA_TYPE *samples, 
                        LABEL_DATA_TYPE *labels,
                        IDX_DATA_TYPE *sorted_indices,
                        float *imp_left, 
                        float *imp_right, 
                        COUNT_DATA_TYPE *label_count,
                        COUNT_DATA_TYPE *split,
                        int range,
                        int n_active_threads,
                        int n_features, 
                        int n_samples, 
                        int stride){
  
  int count_offset = blockIdx.x * MAX_NUM_LABELS * (blockDim.x + 1); 
  int samples_offset = blockIdx.x * stride;
  int indices_offset = blockIdx.x * stride;

  __shared__ int quit;
  __shared__ float shared_imp_left[THREADS_PER_BLOCK];
  __shared__ float shared_imp_right[THREADS_PER_BLOCK];
  __shared__ COUNT_DATA_TYPE shared_split_index[THREADS_PER_BLOCK];
  
  int range_begin = threadIdx.x * range;
  int range_end = (range_begin + range < n_samples)? range_begin + range : n_samples - 1;
  
  float reg_min_imp_left = 2;
  float reg_min_imp_right = 2;
  int   reg_min_index;
  float reg_imp_right;
  float reg_imp_left;
  int cur_index;

  if(threadIdx.x == 0){
    if(samples[samples_offset + sorted_indices[indices_offset]] == samples[samples_offset + sorted_indices[indices_offset + n_samples - 1]]){
      imp_left[blockIdx.x] = 2;
      imp_right[blockIdx.x] = 2; 
      split[blockIdx.x] = 0;
      quit = 1;
    }
    else
      quit = 0;
  }

  __syncthreads();
  
  if(quit == 1)
    return; 
  
  for(int i = range_begin; i < range_end; ++i){
    cur_index = sorted_indices[indices_offset + i];
    label_count[count_offset + threadIdx.x * MAX_NUM_LABELS + labels[cur_index]]++;
    
    if(samples[samples_offset + cur_index] == samples[samples_offset + sorted_indices[indices_offset + i + 1]])
      continue;
    
    reg_imp_left = ((i + 1) / float(n_samples)) * calc_imp_left(&label_count[count_offset + threadIdx.x * MAX_NUM_LABELS], i + 1);
    reg_imp_right = ((n_samples - i - 1) / float(n_samples)) * calc_imp_right(&label_count[count_offset + threadIdx.x * MAX_NUM_LABELS],
                                                                                &label_count[count_offset + n_active_threads * MAX_NUM_LABELS], n_samples - i - 1);
    
    if(reg_imp_left + reg_imp_right < reg_min_imp_left + reg_min_imp_right){
      reg_min_imp_left = reg_imp_left;
      reg_min_imp_right = reg_imp_right;
      reg_min_index = i;
    }  
    
  }
  
  shared_imp_left[threadIdx.x] = reg_min_imp_left;
  shared_imp_right[threadIdx.x] = reg_min_imp_right;
  shared_split_index[threadIdx.x] = reg_min_index;
  
  __syncthreads();


  int n_threads = blockDim.x;
  int next_thread;

  //Parallel tree reduction to find mininum impurity
  while(n_threads > 1){
    int half = (n_threads >> 1);
    if(threadIdx.x < half){
      next_thread = threadIdx.x + half;
      if(shared_imp_left[threadIdx.x] + shared_imp_right[threadIdx.x] > shared_imp_left[next_thread] + shared_imp_right[next_thread]){
        shared_imp_left[threadIdx.x] = shared_imp_left[next_thread];
        shared_imp_right[threadIdx.x] = shared_imp_right[next_thread];
        shared_split_index[threadIdx.x] = shared_split_index[next_thread];
      }
    }

    __syncthreads(); 
    n_threads = half;
  }
 
  __syncthreads();
  
  if(threadIdx.x != 0)
    return;
  
  imp_left[blockIdx.x] = shared_imp_left[0];
  imp_right[blockIdx.x] = shared_imp_right[0];
  split[blockIdx.x] = shared_split_index[0];  
}
