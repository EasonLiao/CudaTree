#include<stdio.h>
#include<math.h>
#include<stdint.h>

#define THREADS_PER_BLOCK %d
#define MAX_NUM_LABELS %d
#define SAMPLE_DATA_TYPE %s
#define LABEL_DATA_TYPE %s
#define COUNT_DATA_TYPE %s
#define IDX_DATA_TYPE %s

texture<COUNT_DATA_TYPE, 1> tex_label_total;

__device__ uint32_t d_square(uint32_t d){ return d * d; }

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
  float reg_imp_right;
  float reg_imp_left;
  float reg_min_imp_left = 2.0;
  float reg_min_imp_right = 2.0;
  COUNT_DATA_TYPE reg_min_split = 0;
  LABEL_DATA_TYPE cur_label;
  uint8_t skip;
  int n;
  uint32_t left_count;
  uint32_t right_count;
  uint32_t pos;
  IDX_DATA_TYPE cur_idx;

  __shared__ uint16_t min_thread_index;
  __shared__ COUNT_DATA_TYPE shared_count[MAX_NUM_LABELS];
  //__shared__ COUNT_DATA_TYPE shared_count_total[MAX_NUM_LABELS];
  __shared__ SAMPLE_DATA_TYPE shared_samples[THREADS_PER_BLOCK + 1];
  __shared__ float shared_imp_total[THREADS_PER_BLOCK];  
  __shared__ uint8_t shared_pos[THREADS_PER_BLOCK];

  for(int i = threadIdx.x; i < MAX_NUM_LABELS; i += blockDim.x){   
      shared_count[i] = 0;
      //shared_count_total[i] = label_total[i];
  }
 
  
  shared_imp_total[threadIdx.x] = 4.0;

  for(int i = 0; i < n_samples - 1; i += blockDim.x){
    pos = i + threadIdx.x;
    cur_idx = (pos < n_samples - 1)? sorted_indices[offset + pos] : 0;
    cur_label = labels[cur_idx];
    skip = 0;
    left_count = 0;
    right_count = 0;
    shared_samples[threadIdx.x] = samples[offset + cur_idx];
    
    if(threadIdx.x == blockDim.x - 1){
      int next_pos;
      next_pos = (pos < n_samples - 1)? pos + 1 : n_samples - 1;
      shared_samples[threadIdx.x + next_pos - pos] = samples[offset + sorted_indices[offset + next_pos]];
    } 

    __syncthreads();

    if(pos >= n_samples - 1 || shared_samples[threadIdx.x] == shared_samples[threadIdx.x + 1])
      skip = 1;
  
    for(int l = 0; l < MAX_NUM_LABELS; ++l){
      shared_pos[threadIdx.x] = (l == cur_label)? 1:0; 
      
      //Prefix scan.
      __syncthreads(); 

      for(int s = 1; s < blockDim.x; s*= 2){
        if(threadIdx.x >= s) 
          n = shared_pos[threadIdx.x - s];
        else 
          n = 0;
        
        __syncthreads();
        shared_pos[threadIdx.x] += n;
        __syncthreads();
      
      }
     
      if(skip == 0){
        int total = shared_pos[threadIdx.x] + shared_count[l];
        left_count += d_square(total);
        //right_count += d_square(shared_count_total[l] - total);
        //right_count += d_square(label_total[l] - total);
        right_count += d_square(tex1Dfetch(tex_label_total, l) - total);
      }
      __syncthreads();

      if(threadIdx.x == blockDim.x - 1)
        shared_count[l] += shared_pos[threadIdx.x];
    }
  

    if(skip == 0){
      reg_imp_left = (1 - float(left_count) / d_square(pos + 1)) * (pos + 1) / n_samples;
      reg_imp_right = (1 - float(right_count) / d_square(n_samples - pos - 1)) * (n_samples - pos - 1) / n_samples;
    
      if(reg_imp_left + reg_imp_right < reg_min_imp_left + reg_min_imp_right){
        reg_min_imp_left = reg_imp_left;
        reg_min_imp_right = reg_imp_right;
        reg_min_split = pos;
      }
    } 
  }
  
  shared_imp_total[threadIdx.x] = reg_min_imp_left + reg_min_imp_right;
  
  __syncthreads();

  if(threadIdx.x == 0){
    min_thread_index = 0;
    float imp_min = 4.0;
    for(int i = 0; i < blockDim.x; ++i)
      if(imp_min > shared_imp_total[i]){
        imp_min = shared_imp_total[i];
        min_thread_index = i;
      } 
  }

  __syncthreads();
  
  if(threadIdx.x == min_thread_index){
    split[blockIdx.x] = reg_min_split;
    imp_left[blockIdx.x] = reg_min_imp_left;
    imp_right[blockIdx.x] = reg_min_imp_right;
  }
}
