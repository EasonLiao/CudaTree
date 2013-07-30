import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from sklearn import tree
from sklearn.datasets import load_iris
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import cProfile

mod = SourceModule(r'''
    #include<stdio.h>
    #include<math.h>
    #define MAX_LABELS 3
    #define MAX_COLS 256

    __device__ float calc_imp(int label_previous[MAX_LABELS], int label_now[MAX_LABELS], int total_size){
      float imp = 1.0;
      for(int i = 0; i < MAX_LABELS; ++i)
        imp -= pow(((label_now[i] - label_previous[i]) / double(total_size)), 2); 

      return imp; 
    }

    __global__ void compute(int* sorted_targets, float *sorted_samples, float *imp_left, float *imp_right, int *split, int row, int col, int leading){
      int label_count[MAX_COLS][MAX_LABELS] = {0, };  
      int label_zeros[MAX_LABELS] = {0, };
      
      float min_imp = -1;
      float min_imp_left, min_imp_right;
      int min_split;

      for(int i = 0; i < col; ++i)
        if(i == 0)
          label_count[i][sorted_targets[i + threadIdx.x + threadIdx.y * leading]]++;
        else{
          for(int l = 0; l < MAX_LABELS; ++l)
            label_count[i][l] = label_count[i-1][l];
          
          label_count[i][sorted_targets[i + threadIdx.x + threadIdx.y * leading]]++; 
         
        }
        
      for(int i = 0; i < col - 1; ++i){
        if(sorted_samples[i + threadIdx.x + threadIdx.y * leading] == sorted_samples[i + 1 + threadIdx.x + threadIdx.y * leading])
          continue;

        float imp_left = calc_imp(label_zeros, label_count[i], i + 1);
        float imp_right = calc_imp(label_count[i], label_count[col-1], col - i - 1);

        if(min_imp == -1 || min_imp > imp_left + imp_right){
          min_imp = imp_left + imp_right;
          min_split = i;
          min_imp_left = imp_left;
          min_imp_right = imp_right;
        }
      }

        split[threadIdx.y] = min_split;
        imp_left[threadIdx.y] = min_imp_left;
        imp_right[threadIdx.y] = min_imp_right;
        __syncthreads();
    }
    '''
    )

class Node(object):
  def __init__(self):
    self.value = None 
    self.error = None
    self.samples = None
    self.feature_threshold = None
    self.feature_index = None
    self.left_child = None
    self.right_child = None
    self.height = None

class SplitInfo(object):
  def __init__(self):
    self.min_impurity = None
    self.impurity_left = None
    self.impurity_right = None
    self.min_threshold = None
    self.boolean_mask_left = None
    self.boolean_mask_right = None

class DecisionTree(object): 
  def __init__(self):
    self.root = None
    self.num_features = None
    self.num_examples = None
    self.kernel = mod.get_function("compute")

  def fit(self, Samples, Target):
    self.num_features = Samples[0].size
    self.num_examples = Samples.size
    assert isinstance(Samples, np.ndarray)
    assert isinstance(Target, np.ndarray)
    assert Samples.size / Samples[0].size == Target.size

    Samples = np.require(np.transpose(Samples), dtype = np.float32, requirements = 'C')
    Target = np.require(np.transpose(Target), dtype = np.int32, requirements = 'C') 
    self.root = self.construct(Samples, Target, 1, 1.0) 


  def construct(self, Samples, Target, Height, ErrRate):
    def check_terminate():
      if ErrRate == 0:
        return True
      else:
        return False
    
    ret_node = Node()
    ret_node.error = ErrRate
    ret_node.samples = Target.size
    ret_node.height = Height

    if check_terminate():
      ret_node.value = Target[0]
      return ret_node

    SortedExamples = np.empty_like(Samples)
    SortedTargets = np.empty_like(Samples).astype(np.int32)
    SortedTargetsGPU = None 
    ImpurityRes = None

    for i,f in enumerate(Samples):
      sorted_index = np.argsort(f)
      SortedExamples[i] = Samples[i][sorted_index]
      SortedTargets[i] = Target[sorted_index]
   
    SortedTargetsGPU = gpuarray.to_gpu(SortedTargets)
    SortedExamplesGPU = gpuarray.to_gpu(SortedExamples)
    ImpurityLeft = gpuarray.empty(SortedTargetsGPU.shape[0], dtype = np.float32)
    ImpurityRight = gpuarray.empty(SortedTargetsGPU.shape[0], dtype = np.float32)
    MinSplit = gpuarray.empty(SortedTargetsGPU.shape[0], dtype = np.int32)
    
    self.kernel(SortedTargetsGPU, 
                SortedExamplesGPU,
                ImpurityLeft,
                ImpurityRight,
                MinSplit,
                np.int32(SortedTargetsGPU.shape[0]), 
                np.int32(SortedTargetsGPU.shape[1]), 
                np.int32(SortedTargetsGPU.strides[0] / Target.itemsize), #leading
                block = (1, SortedTargetsGPU.shape[0], 1), #launch number of features threads
                grid = (1, 1)
                )

    imp_left = ImpurityLeft.get()
    imp_right = ImpurityRight.get()
    imp_total = imp_left + imp_right
    
    ret_node.feature_index =  imp_total.argmin()
    row = ret_node.feature_index
    col = MinSplit.get()[row]
    ret_node.feature_threshold = (SortedExamples[row][col] + SortedExamples[row][col + 1]) / 2.0 

    boolean_mask_left = (Samples[ret_node.feature_index] < ret_node.feature_threshold)
    boolean_mask_right = (Samples[ret_node.feature_index] >= ret_node.feature_threshold)
 
    ret_node.left_child = self.construct(Samples[:, boolean_mask_left].copy(), Target[boolean_mask_left].copy(), Height + 1, imp_left[ret_node.feature_index])
    ret_node.right_child = self.construct(Samples[:, boolean_mask_right].copy(), Target[boolean_mask_right].copy(), Height + 1, imp_right[ret_node.feature_index])
    return ret_node 

   
  def _predict(self, val):
    temp = self.root
     
    while True:
      if temp.left_child and temp.right_child:
        if val[temp.feature_index] < temp.feature_threshold:
          temp = temp.left_child
        else:
          temp = temp.right_child
      else: 
          return temp.value

  def predict(self, inputs):
    res = []
    for val in inputs:
      res.append(self._predict(val))
    return np.array(res)

  def print_tree(self):
    def recursive_print(node):
      if node.left_child and node.right_child:
        print "Height : %s,  Feature Index : %s,  Threshold : %s Samples: %s" % (node.height, node.feature_index, node.feature_threshold, node.samples)  
        recursive_print(node.left_child)
        recursive_print(node.right_child)
      else:
        print "Leaf Height : %s,  Samples : %s" % (node.height, node.samples)  
    assert self.root is not None
    recursive_print(self.root)

if __name__ == "__main__":
  d = DecisionTree()
  iris = load_iris()
  cProfile.run("d.fit(iris.data, iris.target)")
  d.print_tree() 
  res = d.predict(iris.data)
  print np.allclose(iris.target, res)
  

