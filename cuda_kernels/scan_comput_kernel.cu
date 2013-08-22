#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define MAX_NUM_LABELS %d
#define THREADS_PER_BLOCK %d
#define LABEL_DATA_TYPE %s
#define COUNT_DATA_TYPE %s
#define SAMPLE_DATA_TYPE %s

__global__ void scan_compute(
                          LABEL_DATA_TYPE *sorted_labels,
                          SAMPLE_DATA_TYPE *sorted_samples,
                          COUNT_DATA_TYPE *label_count,
                          float *imp_left,
                          float *imp_right,
                          COUNT_DATA_TYPE *split,                    
                          int range,
                          int n_active_threads,
                          int n_samples,
                          int stride    
                            ){
  int count_offset = blockIdx.x * MAX_NUM_LABELS * (blockDim.x + 1);
  int offset = blockIdx.x * stride;
  int range_begin = threadIdx.x * range; 
  int range_end = (range_begin + range < n_samples)? range_begin + range : n_samples;
  
  if(threadIdx.x >= n_active_threads)
    return;

  for(int i = 0; i < MAX_NUM_LABELS; ++i)
    label_count[count_offset + (threadIdx.x + 1) * MAX_NUM_LABELS + i] = 0;

  for(int i = threadIdx.x; i < MAX_NUM_LABELS; i += n_active_threads)
    label_count[count_offset + i] = 0;


  for(int i = range_begin; i < range_end; ++i){
    LABEL_DATA_TYPE cur_label = sorted_labels[offset + i];
    label_count[count_offset + (threadIdx.x + 1) * MAX_NUM_LABELS + cur_label]++; 
  }

  __syncthreads();
  
  for(int i = 2; i < n_active_threads + 1; ++i){
    int last = (i - 1) * MAX_NUM_LABELS;
    int cur = i * MAX_NUM_LABELS;
    
    for(int l = threadIdx.x; l < MAX_NUM_LABELS; l += n_active_threads)
      label_count[count_offset + cur + l] += label_count[count_offset +  last + l];
   
    __syncthreads();
  }

}


/*
__global__ void prefix_scan(LABEL_DATA_TYPE *sorted_targets, 
                        COUNT_DATA_TYPE *label_count,
                        int n_features, 
                        int n_samples, 
                        int stride){  
  int label_offset = blockIdx.x * MAX_NUM_LABELS * (blockDim.x + 1); //The offset of label_count for each block.
  int targets_offset = blockIdx.x * stride;                //The offset of sorted_targets for each block

  int range = ceil(double(n_samples) / blockDim.x);            //The range of elements each thread is reponsible for. 
  int n_active_threads = ceil(double(n_samples) / range);     //The number of threads that have the actual work to do.
  int range_begin =(threadIdx.x * range < n_samples)? threadIdx.x * range : n_samples;  //The begin index of each thread.
  int range_end = (range_begin + range < n_samples)? range_begin + range : n_samples;   //The end index of each thread

  //Initialize the first label_count of each thread
  for(int i = 0; i < MAX_NUM_LABELS; ++i)
    label_count[label_offset + i + threadIdx.x * MAX_NUM_LABELS] = 0;

  if(threadIdx.x == blockDim.x - 1)
    for(int i = 0; i < MAX_NUM_LABELS; ++i)
      label_count[label_offset + i + blockDim.x * MAX_NUM_LABELS] = 0;

  if(threadIdx.x < n_active_threads){  
    LABEL_DATA_TYPE cur_label = sorted_targets[targets_offset + range_begin];
    label_count[label_offset + cur_label + (threadIdx.x + 1) * MAX_NUM_LABELS]++;    
  }
 
  //Work out a range of label_count of each thread
  if(threadIdx.x < n_active_threads)
    for(int i = range_begin + 1; i < range_end; ++i){
      LABEL_DATA_TYPE cur_label = sorted_targets[targets_offset + i];
      label_count[label_offset + cur_label + (threadIdx.x + 1) * MAX_NUM_LABELS]++; 
    }
  
  __syncthreads();
  
  //Fist thread of the block does prefix sum on last element of label_count each thread
  if(threadIdx.x == 0)
    for(int i = 1; i < n_active_threads + 1; ++i)
    { 
      int last = (i - 1) * MAX_NUM_LABELS;
      int cur = i * MAX_NUM_LABELS;
      
      for(int l = 0; l < MAX_NUM_LABELS; ++l)
        label_count[label_offset + cur + l] += label_count[label_offset + last + l];    
    }
}

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
                        float *imp_left, 
                        float *imp_right, 
                        COUNT_DATA_TYPE *label_count,
                        COUNT_DATA_TYPE *split, 
                        int n_features, 
                        int n_samples, 
                        int stride){
  int count_offset = blockIdx.x * MAX_NUM_LABELS * (blockDim.x + 1); 
  int samples_offset = blockIdx.x * stride;
  int labels_offset = blockIdx.x * stride;
  
  __shared__ int quit;
  __shared__ float shared_imp_left[MAX_THREADS_PER_BLOCK];
  __shared__ float shared_imp_right[MAX_THREADS_PER_BLOCK];
  __shared__ COUNT_DATA_TYPE shared_split_index[MAX_THREADS_PER_BLOCK];

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
    LABEL_DATA_TYPE label_val = sorted_labels[labels_offset + i];
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
  
  if(threadIdx.x != 0)
    return;
  
  imp_left[blockIdx.x] = shared_imp_left[0];
  imp_right[blockIdx.x] = shared_imp_right[0];
  split[blockIdx.x] = shared_split_index[0];  
}

*/

