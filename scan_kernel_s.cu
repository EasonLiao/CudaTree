//Serialized prefix scan
#include<stdio.h>
#include<math.h>
#define MAX_NUM_SAMPLES %d
#define MAX_NUM_LABELS %d
#define THREADS_PER_BLOCK %d


__global__ void prefix_scan(int* sorted_targets, 
                        int *label_count,
                        int n_features, 
                        int n_samples, 
                        int leading){
  //if(blockIdx.x == 0 && threadIdx.x == 0)
  //  printf("scan_S\n");

  int offset = blockIdx.x * MAX_NUM_LABELS * n_samples; 
  if(threadIdx.x == 0){
    for(int i = 0; i < MAX_NUM_LABELS; ++i)
      label_count[i + offset] = 0;

    int curr_label = sorted_targets[blockIdx.x * n_samples];
    label_count[curr_label + offset]++;
   
    for(int i = 1; i < n_samples; ++i) {
        for(int l = 0; l < MAX_NUM_LABELS; ++l)
          label_count[i * MAX_NUM_LABELS + l + offset] = label_count[(i-1) * MAX_NUM_LABELS + l + offset];
        
        curr_label = sorted_targets[blockIdx.x * n_samples + i];
        
        label_count[i * MAX_NUM_LABELS + curr_label + offset]++; 
      } 
  }

}
