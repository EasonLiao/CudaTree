#define DIFF_THRESHOLD 0.00001

__device__ inline  float calc_imp_right(float* label_previous, float* label_now, COUNT_DATA_TYPE total_size){
  float sum = 0.0;
//#pragma unroll
  for(LABEL_DATA_TYPE i = 0; i < MAX_NUM_LABELS; ++i){
    float count = label_now[i] - label_previous[i];
    sum += count * count;
  }
 
  float denom =  ((float)total_size) * total_size;
  return 1.0 - (sum / denom); 
}

__device__ inline float calc_imp_left(float* label_now, COUNT_DATA_TYPE total_size){
  float sum = 0.0;
//#pragma unroll
  for(LABEL_DATA_TYPE i = 0; i < MAX_NUM_LABELS; ++i){
    float count = label_now[i];
    sum += count * count;
  }
  
  float denom =  ((float)total_size) * total_size;
  return 1.0 - (sum / denom); 
}

__device__ __inline__ void calc_impurity(float *label_now, float *label_total, float* left_imp,
    float *right_imp, COUNT_DATA_TYPE left_total, COUNT_DATA_TYPE right_total){
  //float left_sum = 0.0;
  //float right_sum = 0.0;

  *left_imp = 0;
  *right_imp = 0;

  for(LABEL_DATA_TYPE i = 0; i < MAX_NUM_LABELS; ++i){
    float left_count = label_now[i];
    *left_imp += left_count * left_count;
    float right_count = label_total[i] - left_count;
    *right_imp += right_count * right_count; 
  }
  
  *left_imp = (1.0 - *left_imp / (left_total * left_total)) * left_total / (left_total + right_total);
  *right_imp = (1.0 - *right_imp / (right_total * right_total)) * right_total / (left_total + right_total);
}
