import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from sklearn import tree
import sklearn.datasets
from sklearn.datasets import load_iris
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import cProfile
import sys

def mk_kernel(n_samples, n_labels, _cache = {}):
  key = (n_samples, n_labels)
  if key in _cache:
    return _cache[key]
  mod = SourceModule(r'''
    #include<stdio.h>
    #include<math.h>
    #define MAX_NUM_SAMPLES %d
    #define MAX_NUM_LABELS %d

    __device__ float calc_imp(int label_previous[MAX_NUM_LABELS], int label_now[MAX_NUM_LABELS], int total_size){
      float imp = 1.0;
      for(int i = 0; i < MAX_NUM_LABELS; ++i)
        imp -= pow(((label_now[i] - label_previous[i]) / double(total_size)), 2); 

      return imp; 
    }

    __global__ void compute(int* sorted_targets, 
                            float *sorted_samples, 
                            float *imp_left, 
                            float *imp_right, 
                            int *split, 
                            int n_features, 
                            int n_samples, 
                            int leading) {
      int label_count[MAX_NUM_SAMPLES][MAX_NUM_LABELS] = {0, };  
      int label_zeros[MAX_NUM_LABELS] = {0, };
     
      float min_imp = 1000;
      float min_imp_left, min_imp_right;
      int min_split;
     
      split[blockIdx.x] = 0; 
      int curr_label = sorted_targets[blockIdx.x * n_samples];
      label_count[0][curr_label]++;
      for(int i = 1; i < n_samples; ++i) {
          for(int l = 0; l < MAX_NUM_LABELS; ++l)
            label_count[i][l] = label_count[i-1][l];
          
          curr_label = sorted_targets[blockIdx.x * n_samples + i];
          label_count[i][curr_label]++; 
        }
      
      //If the first value of the feature equals the last value of the feature, then it means all the values of this feature are same.
      //Igonore it.
      if(sorted_samples[blockIdx.x * n_samples] == sorted_samples[blockIdx.x * n_samples + n_samples - 1]){
        imp_left[blockIdx.x] = 2;
        imp_right[blockIdx.x] = 2;
        return;
      }

      for(int i = 0; i < n_samples - 1; ++i) {

        float curr_value = sorted_samples[i + blockIdx.x * n_samples];
        float next_value = sorted_samples[i + 1 + blockIdx.x * n_samples];
        
        
        if (curr_value == next_value) continue;

        float imp_left = calc_imp(label_zeros, label_count[i], i + 1);
        float imp_right = calc_imp(label_count[i], label_count[n_samples-1], n_samples - i - 1);
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
    ''' % (n_samples, n_labels)
    )
  fn = mod.get_function("compute")
  _cache[key] = fn
  return fn

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
  def __init__(self, num_examples, num_labels, num_features):
    self.root = None
    self.num_examples = num_examples
    self.num_labels = num_labels
    self.num_features = num_features
        
    self.kernel = mk_kernel(num_examples, num_labels)


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
    n_features = SortedTargetsGPU.shape[0]
    ImpurityLeft = gpuarray.empty(n_features, dtype = np.float32)
    ImpurityRight = gpuarray.empty(n_features, dtype = np.float32)
    MinSplit = gpuarray.empty(n_features, dtype = np.int32)


    self.kernel(SortedTargetsGPU, 
                SortedExamplesGPU,
                ImpurityLeft,
                ImpurityRight,
                MinSplit,
                np.int32(SortedTargetsGPU.shape[0]), 
                np.int32(SortedTargetsGPU.shape[1]), 
                np.int32(SortedTargetsGPU.strides[0] / Target.itemsize), #leading
                block = (1, 1, 1), #launch number of features threads
                grid = (SortedTargetsGPU.shape[0], 1)
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
    data_left =  Samples[:, boolean_mask_left].copy()
    target_left = Target[boolean_mask_left].copy()
    assert len(target_left) > 0
    ret_node.left_child = self.construct(data_left, target_left, Height + 1, imp_left[ret_node.feature_index])

    data_right = Samples[:, boolean_mask_right].copy()
    target_right = Target[boolean_mask_right].copy()
    assert len(target_right) > 0 
    ret_node.right_child = self.construct(data_right, target_right, Height + 1, imp_right[ret_node.feature_index])
    
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


def main():
  d = DecisionTree()
  iris = load_iris()
  d.fit(iris.data, iris.target)
  d.print_tree() 


if __name__ == "__main__":
  dataset = sklearn.datasets.load_digits()
  num_examples, num_features = dataset.data.shape
  num_labels = len(dataset.target_names)
 
  d = DecisionTree(num_examples, num_labels, num_features)
  d.fit(dataset.data, dataset.target)
  
  #d.print_tree()
  res = d.predict(dataset.data)
  print np.allclose(dataset.target, res)
  

