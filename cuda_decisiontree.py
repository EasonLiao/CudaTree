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
import math
import sys

def mk_kernel(n_samples, n_labels, kernel_file,  _cache = {}):
  key = (n_samples, n_labels)
  if key in _cache:
    return _cache[key]
  
  with open(kernel_file) as code_file:
    code = code_file.read()  
    mod = SourceModule(code % (n_samples, n_labels))
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


class DecisionTree(object): 
  KERNEL_1 = "kernel_1.cu"  #One thread per feature.
  KERNEL_2 = "kernel_2.cu"  #One block per feature.
  KERNEL_3 = "kernel_3.cu"  #One block per feature, add parallel reduction to find minimum impurity.
  
  THREADS_PER_BLOCK = 64  

  def __init__(self):
    self.root = None
    self.kernel_type = None
    self.num_labels = None

  def fit(self, samples, target,  kernel_type):
    assert isinstance(samples, np.ndarray)
    assert isinstance(target, np.ndarray)
    assert samples.size / samples[0].size == target.size
    
    self.num_labels = np.unique(target).size
    self.kernel = mk_kernel(target.size,self.num_labels, kernel_type)
    self.kernel_type = kernel_type
    samples = np.require(np.transpose(samples), dtype = np.float32, requirements = 'C')
    target = np.require(np.transpose(target), dtype = np.int32, requirements = 'C') 
    self.root = self.__construct(samples, target, 1, 1.0) 


  def __construct(self, samples, target, Height, error_rate):
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

    
    n_features = sorted_targetsGPU.shape[0]
    n_samples = sorted_targetsGPU.shape[1]
    leading = sorted_targetsGPU.strides[0] / target.itemsize

    assert n_samples == leading #Just curious about if they can be different.
    grid = (n_features, 1)
    
    if self.kernel_type !=  self.KERNEL_1:
      #Launch 64 threads per thread block
      #Launch number of features blocks
      block = (self.THREADS_PER_BLOCK, 1, 1)

      #Create a global label_count array, and pass it to kernel function.
      label_count = gpuarray.empty(ret_node.samples * self.num_labels * sorted_targetsGPU.shape[0], dtype = np.int32)
      self.kernel(sorted_targetsGPU, 
                  sorted_examplesGPU,
                  impurity_left,
                  impurity_right,
                  label_count,
                  min_split,
                  np.int32(n_features), 
                  np.int32(n_samples), 
                  np.int32(leading),
                  block = block,
                  grid = grid
                  )
    else:
      block = (1, 1, 1)
      self.kernel(sorted_targetsGPU, 
                  sorted_examplesGPU,
                  impurity_left,
                  impurity_right,
                  min_split,
                  np.int32(n_features), 
                  np.int32(n_samples), 
                  np.int32(leading),
                  block = block,
                  grid = grid
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
    ret_node.left_child = self.__construct(data_left, target_left, Height + 1, imp_left[ret_node.feature_index])

    data_right = samples[:, boolean_mask_right].copy()
    target_right = target[boolean_mask_right].copy()
    assert len(target_right) > 0 
    ret_node.right_child = self.__construct(data_right, target_right, Height + 1, imp_right[ret_node.feature_index])
    
    return ret_node 

   
  def __predict(self, val):
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
      res.append(self.__predict(val))
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
  dataset = sklearn.datasets.load_iris()
  num_labels = len(dataset.target_names) 
  d.fit(dataset.data, dataset.target, DecisionTree.KERNEL_2)
  #d.print_tree()
  res = d.predict(dataset.data)
  print np.allclose(dataset.target, res)
  

