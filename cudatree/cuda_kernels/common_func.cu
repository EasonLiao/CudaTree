
__device__ inline  float calc_imp_right(float* label_previous, float* label_now, COUNT_DATA_TYPE total_size){
  float sum = 0.0;
#pragma unroll
  for(LABEL_DATA_TYPE i = 0; i < MAX_NUM_LABELS; ++i){
    float count = label_now[i] - label_previous[i];
    sum += count * count;
  }
 
  float denom =  ((float)total_size) * total_size;
  return 1.0 - (sum / denom); 
}

__device__ inline  float calc_imp_left(float* label_now, COUNT_DATA_TYPE total_size){
  float sum = 0.0;
#pragma unroll
  for(LABEL_DATA_TYPE i = 0; i < MAX_NUM_LABELS; ++i){
    float count = label_now[i];
    sum += count * count;
  }
  
  float denom =  ((float)total_size) * total_size;
  return 1.0 - (sum / denom); 
}
