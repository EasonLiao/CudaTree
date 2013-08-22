//Add parallel reduction to find mininum impurity based on kernel_2.cu
#include<stdio.h>
#include<math.h>
#include<stdint.h>
#include<math.h>
#define THREADS_PER_BLOCK %d
#define MAX_NUM_LABELS %d
#define SAMPLE_DATA_TYPE %s
#define LABEL_DATA_TYPE %s
#define COUNT_DATA_TYPE %s

__device__  double calc_imp_right(COUNT_DATA_TYPE label_previous[MAX_NUM_LABELS], COUNT_DATA_TYPE label_now[MAX_NUM_LABELS], int total_size){
  double sum = 0.0; 
  for(int i = 0; i < MAX_NUM_LABELS; ++i){
    double count = label_now[i] - label_previous[i];
    sum += count * count;
  }

  double denom = ((double) total_size) * total_size;
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
                        COUNT_DATA_TYPE *label_total,
                        COUNT_DATA_TYPE *split, 
                        int n_samples, 
                        int stride){
  int offset = blockIdx.x * stride;
  int stop_pos;
  double reg_imp_right = 2.0;
  double reg_imp_left = 2.0;
  COUNT_DATA_TYPE reg_min_split;
  double imp_right_before;
  double imp_left_before;

  __shared__ COUNT_DATA_TYPE shared_count[MAX_NUM_LABELS];
  __shared__ LABEL_DATA_TYPE shared_labels[THREADS_PER_BLOCK];
  __shared__ COUNT_DATA_TYPE shared_count_total[MAX_NUM_LABELS];
  __shared__ SAMPLE_DATA_TYPE shared_samples[THREADS_PER_BLOCK];
  

  for(int i = threadIdx.x; i < MAX_NUM_LABELS; i += blockDim.x){   
      shared_count[i] = 0;
      shared_count_total[i] = label_total[i];
  }


  for(int i = threadIdx.x; i < n_samples; i += blockDim.x){ 
    shared_labels[threadIdx.x] = sorted_labels[offset + i]; 
    shared_samples[threadIdx.x] = sorted_samples[offset + i];

    __syncthreads();
     
    if(threadIdx.x == 0){
      stop_pos = (i + blockDim.x < n_samples - 1)? blockDim.x : n_samples - i - 1;
      
        for(int t = 0; t < stop_pos; ++t){
          COUNT_DATA_TYPE lcount = ++shared_count[shared_labels[t]];
          
          int now_n = t + i + 1;
          int pre_n = now_n - 1;
          int pre_r_n = n_samples - pre_n;
          int now_r_n = n_samples - now_n;
          COUNT_DATA_TYPE rlcount = shared_count_total[shared_labels[t]] - lcount;

          if(pre_n != 0)
          {  
            imp_left_before = (1 - ((1 - imp_left_before - pow((lcount - 1)/double(pre_n),2)) * pow(double(pre_n) / now_n, 2) + pow(lcount / double(now_n), 2)));
            imp_right_before = (1 - ((1 - imp_right_before - pow((rlcount + 1)/double(pre_r_n),2)) * pow(double(pre_r_n) / now_r_n, 2) + pow(rlcount / double(now_r_n), 2)));
          }
          else{ 
            imp_left_before = 0.0;
            imp_right_before  = calc_imp_right(shared_count, shared_count_total, n_samples - i - 1 - t);
          }

         
          if(t != stop_pos - 1){
            if(shared_samples[t] == shared_samples[t + 1]){
              continue;
            }
          }
          else if(shared_samples[t] == sorted_samples[offset + stop_pos + i]){
            continue;
          }
            
          double imp_left = (i + 1 + t) / double(n_samples) * imp_left_before;
          double imp_right = (n_samples - i - 1 - t) / double(n_samples) * imp_right_before; 
          
          /*
          float imp_l = (i + 1 + t) / float(n_samples) * calc_imp_left(shared_count, i + 1 + t); 
          float imp_r = (n_samples - i - 1 - t) / float(n_samples) * calc_imp_right(shared_count, shared_count_total,  n_samples - i - 1 - t);

          if(blockIdx.x == 1)
            printf("pos: %%d,   %%f %%f ; %%f %%f\n", t + i,  imp_left, imp_l, imp_right, imp_r);
          */

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
