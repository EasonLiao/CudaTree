#include<stdio.h>
#include<math.h>

__global__ void fill_table(int* sorted_indices,
                          int n_samples,
                          int split_idx,
                          int *mark_table,
                          int stride
                          ){

    for(int i = threadIdx.x; i < n_samples; i += blockDim.x)
      if(i <= split_idx)
        mark_table[sorted_indices[i]] = 0;
      else 
        mark_table[sorted_indices[i]] = 1;
}









