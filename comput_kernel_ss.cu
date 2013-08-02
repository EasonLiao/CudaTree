//One thread per feature
#include<stdio.h>
#include<math.h>
#define MAX_NUM_SAMPLES %d
#define MAX_NUM_LABELS %d

__device__  float calc_imp(int label_previous[MAX_NUM_LABELS], int label_now[MAX_NUM_LABELS], int total_size){
  float imp = 1.0;
  for(int i = 0; i < MAX_NUM_LABELS; ++i)
    imp -= pow(((label_now[i] - label_previous[i]) / double(total_size)), 2); 

  return imp; 
}

__global__ void compute(float *sorted_samples, 
                        float *imp_left, 
                        float *imp_right, 
                        int *label_count,
                        int *split, 
                        int n_features, 
                        int n_samples, 
                        int leading) {

  int offset = blockIdx.x * n_samples * MAX_NUM_LABELS;
  int label_zeros[MAX_NUM_LABELS] = {0, }; 
  float min_imp = 1000;
  float min_imp_left, min_imp_right;
  int min_split;
 
  split[blockIdx.x] = 0; 
  
  //If the first value of the feature equals the last value of the feature, then it means all 
  //the values of this feature are same. Ignore it.
  if(sorted_samples[blockIdx.x * n_samples] == sorted_samples[blockIdx.x * n_samples + n_samples - 1]){
    imp_left[blockIdx.x] = 2;
    imp_right[blockIdx.x] = 2;
    return;
  }

  for(int i = 0; i < n_samples - 1; ++i) {

    float curr_value = sorted_samples[i + blockIdx.x * n_samples];
    float next_value = sorted_samples[i + 1 + blockIdx.x * n_samples];
    
    
    if (curr_value == next_value) continue;

    float imp_left = ((i + 1) / float(n_samples)) * calc_imp(label_zeros, &label_count[offset + i * MAX_NUM_LABELS], i + 1);
    float imp_right = ((n_samples - i - 1) / float(n_samples)) * calc_imp(&label_count[offset + i * MAX_NUM_LABELS],&label_count[offset + (n_samples-1) * MAX_NUM_LABELS], n_samples - i - 1);
    float impurity = imp_left + imp_right;
    if(min_imp > impurity) {
      min_imp = impurity;
      min_split = i;
      min_imp_left = imp_left;
      min_imp_right = imp_right;
    }
  }
  
  split[blockIdx.x] = min_split;
  imp_left[blockIdx.x] = min_imp_left;
  imp_right[blockIdx.x] = min_imp_right;  
}
