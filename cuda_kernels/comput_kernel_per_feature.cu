//Add parallel reduction to find mininum impurity based on kernel_2.cu
#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define THREADS_PER_BLOCK %d 
#define MAX_NUM_LABELS %d
#define SAMPLE_DATA_TYPE %s
#define LABEL_DATA_TYPE %s
#define COUNT_DATA_TYPE %s

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


__global__ void compute(SAMPLE_DATA_TYPE *sorted_samples, 
                        LABEL_DATA_TYPE *sorted_labels,
                        COUNT_DATA_TYPE *label_count,
                        float *imp_left, 
                        float *imp_right, 
                        COUNT_DATA_TYPE *split, 
                        int range, 
                        int n_active_threads, 
                        int n_samples){
  
  int thread_offset = blockIdx.x * blockDim.x;
  int thread_id = threadIdx.x + thread_offset;
  int range_begin = thread_id * range;
  int range_end = (range_begin + range < n_samples - 1) ? range_begin + range : n_samples - 1;
  __shared__ int quit;
  __shared__ float shared_imp_left[THREADS_PER_BLOCK];
  __shared__ float shared_imp_right[THREADS_PER_BLOCK];
  __shared__ COUNT_DATA_TYPE shared_split_index[THREADS_PER_BLOCK];
  __shared__ COUNT_DATA_TYPE shared_label_count_total[MAX_NUM_LABELS];


  shared_imp_left[threadIdx.x] = 2.0;
  shared_imp_right[threadIdx.x] = 2.0;

  //Check if this block should be skipped.
  if(threadIdx.x == 0){
    int start_sample_idx = thread_offset * range;
    int stop_sample_idx = (blockIdx.x + 1) * blockDim.x * range - 1;
    stop_sample_idx = (stop_sample_idx > n_samples - 1)? n_samples -1 : stop_sample_idx;

    if(sorted_samples[start_sample_idx] == sorted_samples[stop_sample_idx]){
      imp_left[blockIdx.x] = 2;
      imp_right[blockIdx.x] = 2;
      quit = 1;
    }
    else
      quit = 0;
  
    
    for(int l = 0; l < MAX_NUM_LABELS; ++l)  
      shared_label_count_total[l] = label_count[n_active_threads * MAX_NUM_LABELS + l];
  }

  __syncthreads();
  
  if(quit == 1)
    return;
 
  //printf("%%d -- %%d\n", range_begin, range_end);
  for(int i = range_begin; i < range_end; ++i){
    LABEL_DATA_TYPE label_val = sorted_labels[i];
    label_count[thread_id * MAX_NUM_LABELS + label_val]++;
    
    if(sorted_samples[i] == sorted_samples[i + 1])
      continue;
      
    float imp_left = ((i + 1) / float(n_samples)) * calc_imp_left(&label_count[thread_id * MAX_NUM_LABELS], i + 1);
    float imp_right = ((n_samples - i - 1) / float(n_samples)) * calc_imp_right(&label_count[thread_id * MAX_NUM_LABELS],
                                                                                shared_label_count_total, n_samples - i - 1);
    float impurity = imp_left + imp_right;
    if(impurity < shared_imp_left[threadIdx.x] + shared_imp_right[threadIdx.x]){
      shared_imp_left[threadIdx.x] = imp_left;
      shared_imp_right[threadIdx.x] = imp_right;
      shared_split_index[threadIdx.x] = i;  
    }
  } 

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

  if(threadIdx.x != 0)
    return;
  

  imp_left[blockIdx.x] = shared_imp_left[0];
  imp_right[blockIdx.x] = shared_imp_right[0];
  split[blockIdx.x] = shared_split_index[0];  
}
