#include<stdio.h>
#include<math.h>
#include<stdint.h>
#define IDX_DATA_TYPE %s
#define THREADS_PER_BLOCK %s

#define WARP_SIZE 32
#define WARP_MASK 0x1f

texture<char, 1> tex_mark;

__global__ void scan_reshuffle(uint8_t* mark_table,
                          IDX_DATA_TYPE* sorted_indices,
                          IDX_DATA_TYPE* sorted_indices_out,
                          int n_samples,
                          int split_idx,
                          int stride
                          ){  
  uint32_t indices_offset = blockIdx.x * stride;
  IDX_DATA_TYPE reg_pos = 0;
  uint32_t out_pos;
  uint32_t right_pos = indices_offset + split_idx + 1;
  uint8_t side;
  IDX_DATA_TYPE n;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  uint16_t lane_id = threadIdx.x & WARP_MASK;
  uint16_t warp_id = threadIdx.x / WARP_SIZE;
  __shared__ IDX_DATA_TYPE shared_pos_table[THREADS_PER_BLOCK / WARP_SIZE];
#else
  __shared__ IDX_DATA_TYPE shared_pos_table[THREADS_PER_BLOCK];
#endif
  
  __shared__ IDX_DATA_TYPE last_sum;
  
  if(threadIdx.x == 0)
    last_sum = 0;
  
  for(IDX_DATA_TYPE i = threadIdx.x; i < n_samples; i += blockDim.x){
    side = tex1Dfetch(tex_mark, sorted_indices[indices_offset + i]);
    reg_pos = side;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  
    for(uint16_t s = 1; s < WARP_SIZE; s *= 2){
      n = __shfl_up(reg_pos, s);
      if(lane_id >= s)
        reg_pos += n;
    }

    if(lane_id == WARP_SIZE - 1)
      shared_pos_table[warp_id] = reg_pos;
   
    __syncthreads();
   
    if(threadIdx.x == 0)
      for(uint16_t l = 1; l < blockDim.x / WARP_SIZE - 1; ++l)
        shared_pos_table[l] += shared_pos_table[l-1];

    __syncthreads();
    
    if(warp_id > 0)
      reg_pos += shared_pos_table[warp_id - 1];
    
    reg_pos += last_sum; 

#else
    
    shared_pos_table[threadIdx.x] = reg_pos;
    __syncthreads();
     
    for(uint16_t s = 1; s < blockDim.x; s *= 2){
      if(threadIdx.x >= s){
        n = shared_pos_table[threadIdx.x - s];
      }
      else 
        n = 0;

      __syncthreads();
      shared_pos_table[threadIdx.x] += n;
      __syncthreads();
    }

    reg_pos = shared_pos_table[threadIdx.x] + last_sum;  
#endif

    out_pos = (side == 1)? indices_offset + reg_pos - 1 : right_pos + i - reg_pos ;
    sorted_indices_out[out_pos] = sorted_indices[indices_offset + i];  
    
    __syncthreads();
    
    if(threadIdx.x == blockDim.x - 1)
      last_sum = reg_pos; 
  }

}

