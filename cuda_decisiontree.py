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

        float imp_left = ((i + 1) / float(n_samples)) * calc_imp(label_zeros, label_count[i], i + 1);
        float imp_right = ((n_samples - i - 1) / float(n_samples)) * calc_imp(label_count[i], label_count[n_samples-1], n_samples - i - 1);
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
  def __init__(self):
    self.root = None


  def fit(self, samples, target, num_labels):
    assert isinstance(samples, np.ndarray)
    assert isinstance(target, np.ndarray)
    assert samples.size / samples[0].size == target.size

    self.kernel = mk_kernel(target.size, num_labels)

    samples = np.require(np.transpose(samples), dtype = np.float32, requirements = 'C')
    target = np.require(np.transpose(target), dtype = np.int32, requirements = 'C') 
    self.root = self.construct(samples, target, 1, 1.0) 


  def construct(self, samples, target, Height, error_rate):
    def check_terminate():
      if error_rate == 0:
        return True
      else:
        return False
    
    ret_node = Node()
    ret_node.error = error_rate
    ret_node.samples = target.size
    ret_node.height = Height

    if check_terminate():
      ret_node.value = target[0]
      return ret_node

    sorted_examples = np.empty_like(samples)
    sorted_targets = np.empty_like(samples).astype(np.int32)
    sorted_targetsGPU = None 

    for i,f in enumerate(samples):
      sorted_index = np.argsort(f)
      sorted_examples[i] = samples[i][sorted_index]
      sorted_targets[i] = target[sorted_index]
   
    sorted_targetsGPU = gpuarray.to_gpu(sorted_targets)
    sorted_examplesGPU = gpuarray.to_gpu(sorted_examples)
    n_features = sorted_targetsGPU.shape[0]
    impurity_left = gpuarray.empty(n_features, dtype = np.float32)
    impurity_right = gpuarray.empty(n_features, dtype = np.float32)
    min_split = gpuarray.empty(n_features, dtype = np.int32)


    self.kernel(sorted_targetsGPU, 
                sorted_examplesGPU,
                impurity_left,
                impurity_right,
                min_split,
                np.int32(sorted_targetsGPU.shape[0]), 
                np.int32(sorted_targetsGPU.shape[1]), 
                np.int32(sorted_targetsGPU.strides[0] / target.itemsize), #leading
                block = (1, 1, 1), #launch number of features threads
                grid = (sorted_targetsGPU.shape[0], 1)
                )
    
    imp_left = impurity_left.get()
    imp_right = impurity_right.get()
    imp_total = imp_left + imp_right
    
    ret_node.feature_index =  imp_total.argmin()
    row = ret_node.feature_index
    col = min_split.get()[row]
    ret_node.feature_threshold = (sorted_examples[row][col] + sorted_examples[row][col + 1]) / 2.0 

    boolean_mask_left = (samples[ret_node.feature_index] < ret_node.feature_threshold)
    boolean_mask_right = (samples[ret_node.feature_index] >= ret_node.feature_threshold)
    data_left =  samples[:, boolean_mask_left].copy()
    target_left = target[boolean_mask_left].copy()
    assert len(target_left) > 0
    ret_node.left_child = self.construct(data_left, target_left, Height + 1, imp_left[ret_node.feature_index])

    data_right = samples[:, boolean_mask_right].copy()
    target_right = target[boolean_mask_right].copy()
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


if __name__ == "__main__":
  d = DecisionTree()  
  dataset = sklearn.datasets.load_digits()
  num_labels = len(dataset.target_names) 
  d.fit(dataset.data, dataset.target, num_labels)
  #d.print_tree()
  res = d.predict(dataset.data)
  print np.allclose(dataset.target, res)
  

