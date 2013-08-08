//Add parallel reduction to find mininum impurity based on kernel_2.cu
#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define MAX_NUM_SAMPLES %d
#define MAX_NUM_LABELS %d
#define MAX_THREADS_PER_BLOCK 256 
#define SAMPLE_DATA_TYPE %s

__device__  float calc_imp_right(int label_previous[MAX_NUM_LABELS], int label_now[MAX_NUM_LABELS], int total_size){
  float sum = 0.0; 
  for(int i = 0; i < MAX_NUM_LABELS; ++i){
    float count = label_now[i] - label_previous[i];
    sum += count * count;
  }

  float denom = ((float) total_size) * total_size;

  return 1.0 - (sum / denom); 
}

__device__  float calc_imp_left(int label_now[MAX_NUM_LABELS], int total_size){
  float sum = 0.0;
  for(int i = 0; i < MAX_NUM_LABELS; ++i){
    float count = label_now[i];
    sum += count * count;
  }
  
  float denom = ((float) total_size) * total_size;
  return 1.0 - (sum / denom); 
}

__global__ void compute(SAMPLE_DATA_TYPE *sorted_samples, 
                        int *sorted_labels,
                        float *imp_left, 
                        float *imp_right, 
                        int *label_count,
                        int *split, 
                        int n_features, 
                        int n_samples, 
                        int stride){

  int count_offset = blockIdx.x * MAX_NUM_LABELS * (blockDim.x + 1); 
  int samples_offset = blockIdx.x * stride;
  int labels_offset = blockIdx.x * stride;
  
  __shared__ int quit;
  __shared__ float shared_imp_left[MAX_THREADS_PER_BLOCK];
  __shared__ float shared_imp_right[MAX_THREADS_PER_BLOCK];
  __shared__ int shared_split_index[MAX_THREADS_PER_BLOCK];

  int range = ceil(double(n_samples) / blockDim.x);
  int n_active_threads = ceil(double(n_samples) / range);     //The number of threads that have the actual work to do.
  int range_begin =(threadIdx.x * range < n_samples)? threadIdx.x * range : n_samples - 1;
  int range_end = (range_begin + range < n_samples)? range_begin + range : n_samples - 1;
  
  shared_imp_left[threadIdx.x] = 2;
  shared_imp_right[threadIdx.x] = 2;

  if(threadIdx.x == 0){
    if(sorted_samples[samples_offset] == sorted_samples[samples_offset + n_samples - 1]){
      imp_left[blockIdx.x] = 2;
      imp_right[blockIdx.x] = 2; 
      quit = 1;
    }
    else
      quit = 0;
  }

  __syncthreads();

  if(quit == 1)
    return; 

  for(int i = range_begin; i < range_end; ++i){
    int label_val = sorted_labels[labels_offset + i];
    label_count[count_offset + threadIdx.x * MAX_NUM_LABELS + label_val]++;
    
    if(sorted_samples[samples_offset + i] == sorted_samples[samples_offset + i + 1])
      continue;
    
    float imp_left = ((i + 1) / float(n_samples)) * calc_imp_left(&label_count[count_offset + threadIdx.x * MAX_NUM_LABELS], i + 1);
    float imp_right = ((n_samples - i - 1) / float(n_samples)) * calc_imp_right(&label_count[count_offset + threadIdx.x * MAX_NUM_LABELS],
                                                                                &label_count[count_offset + n_active_threads * MAX_NUM_LABELS], n_samples - i - 1);
    
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
 
  __syncthreads();
  imp_left[blockIdx.x] = shared_imp_left[0];
  imp_right[blockIdx.x] = shared_imp_right[0];
  split[blockIdx.x] = shared_split_index[0];  
}
