#include<stdio.h>
#include<math.h>
#define MAX_NUM_SAMPLES %d
#define MAX_NUM_LABELS %d
#define MAX_THREADS_PER_BLOCK 64

__device__  float calc_imp(int label_previous[MAX_NUM_LABELS], int label_now[MAX_NUM_LABELS], int total_size){
  float imp = 1.0;
  for(int i = 0; i < MAX_NUM_LABELS; ++i)
    imp -= pow(((label_now[i] - label_previous[i]) / double(total_size)), 2); 

  return imp; 
}

__global__ void compute(int* sorted_targets, 
                        float *sorted_samples, 
                        float *imp_left, 
                        float *imp_right, 
                        int *label_count,
                        int *split, 
                        int n_features, 
                        int n_samples, 
                        int leading){
   
  int offset = blockIdx.x * MAX_NUM_LABELS * n_samples; 
  int label_zeros[MAX_NUM_LABELS] = {0, };
  __shared__ int quit;
  __shared__ float shared_imp_left[MAX_THREADS_PER_BLOCK];
  __shared__ float shared_imp_right[MAX_THREADS_PER_BLOCK];
  __shared__ int shared_split_index[MAX_THREADS_PER_BLOCK];


  int range = ceil(double(n_samples) / blockDim.x);
  int range_begin =(threadIdx.x * range < n_samples)? threadIdx.x * range : n_samples - 1;
  int range_end = (range_begin + range < n_samples)? range_begin + range : n_samples - 1;
  shared_imp_left[threadIdx.x] = 2;
  shared_imp_right[threadIdx.x] = 2;

  if(threadIdx.x == 0){
    
    if(sorted_samples[blockIdx.x * n_samples] == sorted_samples[blockIdx.x * n_samples + n_samples - 1]){
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

  if(threadIdx.x == 0){

    for(int i = 0; i < MAX_NUM_LABELS; ++i)
      label_count[i + offset] = 0;

    split[blockIdx.x] = 0; 
    

    int curr_label = sorted_targets[blockIdx.x * n_samples];
    label_count[curr_label + offset]++;
   
    for(int i = 1; i < n_samples; ++i) {
        for(int l = 0; l < MAX_NUM_LABELS; ++l)
          label_count[i * MAX_NUM_LABELS + l + offset] = label_count[(i-1) * MAX_NUM_LABELS + l + offset];
        
        curr_label = sorted_targets[blockIdx.x * n_samples + i];
        
        label_count[i * MAX_NUM_LABELS + curr_label + offset]++; 
      }
  
  }

  __syncthreads();
  
  for(int i = range_begin; i < range_end; ++i){
    float cur_value = sorted_samples[blockIdx.x * n_samples + i];
    float next_value = sorted_samples[blockIdx.x * n_samples + i + 1];
    if(cur_value == next_value)
      continue;

    float imp_left = ((i + 1) / float(n_samples)) * calc_imp(label_zeros, &label_count[i * MAX_NUM_LABELS + offset], i + 1);
    float imp_right = ((n_samples - i - 1) / float(n_samples)) * calc_imp(&label_count[i * MAX_NUM_LABELS + offset], &label_count[MAX_NUM_LABELS * (n_samples-1) + offset], n_samples - i - 1);
    float impurity = imp_left + imp_right;
    if(impurity < shared_imp_left[threadIdx.x] + shared_imp_right[threadIdx.x]){
      shared_imp_left[threadIdx.x] = imp_left;
      shared_imp_right[threadIdx.x] = imp_right;
      shared_split_index[threadIdx.x] = i;
    }
  }   
  __syncthreads();

  float min_imp = 2;
  int min_index;
  
  if(threadIdx.x == 0){
    for(int i = 0; i < blockDim.x; ++i){
      float total_imp = shared_imp_left[i] + shared_imp_right[i];      
      if(total_imp < min_imp){
        min_imp = total_imp;
        min_index = i;
      }
    }
  }
  else
    return;
    
  imp_left[blockIdx.x] = shared_imp_left[min_index];
  imp_right[blockIdx.x] = shared_imp_right[min_index];
  split[blockIdx.x] = shared_split_index[min_index];
  
}
