import numpy as np
from util import get_best_dtype
from pycuda import gpuarray

class BaseTree(object):
  def __init__(self):
    self.root = None
    self.max_depth = None

  def print_tree(self):
    def recursive_print(node):
      if node.left_child and node.right_child:
        print node
        recursive_print(node.left_child)
        recursive_print(node.right_child)
      else:
        print node
    assert self.root is not None
    recursive_print(self.root)

  def __predict(self, val):
    temp = self.root
    while True:
      if temp.left_child and temp.right_child:
        if val[temp.feature_index] < temp.feature_threshold:
          temp = temp.left_child
        else:
          temp = temp.right_child
      else:
          if hasattr(self, "compt_table"):
            return self.compt_table[temp.value]
          else:
            return temp.value

  def predict(self, inputs):
    res = []
    for val in inputs:
      res.append(self.__predict(val))
    return np.array(res)

  def __gpu_predict(self, val):
    idx = 0
    while True:
      threshold = self.threshold_array[idx]
      threshold_idx = self.feature_array[idx]
      left_idx = self.left_child_array[idx]
      right_idx = self.right_child_array[idx]

      if left_idx != 0 and right_idx != 0: #Means it has children
        if val[threshold_idx] < threshold:
          idx = left_idx
        else:
          idx = right_idx
      else:
        return self.value_array[idx]

  def gpu_predict(self, inputs):
    inputs = np.require(inputs.copy(), requirements = "C")
    n_predict = inputs.shape[0]    
    predict_gpu = gpuarray.to_gpu(inputs)
    left_child_gpu = gpuarray.to_gpu(self.left_child_array)
    right_child_gpu = gpuarray.to_gpu(self.right_child_array)
    threshold_gpu = gpuarray.to_gpu(self.threshold_array)
    value_gpu = gpuarray.to_gpu(self.value_array)
    feature_gpu = gpuarray.to_gpu(self.feature_array)
    predict_res_gpu = gpuarray.zeros(n_predict, dtype=self.dtype_labels)

    self.predict_kernel.prepared_call(
                  (n_predict, 1),
                  (1, 1, 1),
                  left_child_gpu.ptr,
                  right_child_gpu.ptr,
                  feature_gpu.ptr,
                  threshold_gpu.ptr,
                  value_gpu.ptr,
                  predict_gpu.ptr,
                  predict_res_gpu.ptr,
                  self.n_features,
                  self.n_nodes)
    
    if hasattr(self, "compt_table"):
      return np.array([self.compt_table[i] for i in predict_res_gpu.get()], dtype = self.compt_table.dtype)
    else:
      return predict_res_gpu.get()

  def decorate_nodes(self, samples, labels):
    """ In __construct function, the node just store the indices of the actual data, now decorate it with the actual data."""
    def recursive_decorate(node):
      if node.left_child and node.right_child:
        idx_tuple = node.feature_threshold
        node.feature_threshold = (float(samples[idx_tuple[0], idx_tuple[1]]) + samples[idx_tuple[0], idx_tuple[2]]) / 2
        recursive_decorate(node.left_child)
        recursive_decorate(node.right_child)
      else:
        idx = node.value
        node.value = labels[idx]
        
    assert self.root != None
    recursive_decorate(self.root)
  
  def __find_most_common_label(self, x):
    return np.argmax(np.bincount(x))

  def gpu_decorate_nodes(self, samples, labels):
    """ Decorate the nodes for GPU predicting """
    def recursive_decorate(node):
      if node.left_child and node.right_child:
        idx_tuple = node.feature_threshold
        node.feature_threshold = (float(samples[idx_tuple[0], idx_tuple[1]]) + samples[idx_tuple[0], idx_tuple[2]]) / 2
        
        self.left_child_array[node.nid] = node.left_child.nid
        self.right_child_array[node.nid] = node.right_child.nid
        self.threshold_array[node.nid] = node.feature_threshold
        self.feature_array[node.nid] = node.feature_index

        recursive_decorate(node.left_child)
        recursive_decorate(node.right_child)
        node.left_child = None
        node.right_child = None
      else:
        if isinstance(node.value, np.ndarray):
          res = labels[node.value]
          self.value_array[node.nid] = self.__find_most_common_label(res)
        else:
          idx = node.value
          node.value = labels[idx]
          self.value_array[node.nid] = node.value
    
    """
    self.left_child_array = np.zeros(self.n_nodes, self.dtype_indices)
    self.right_child_array = np.zeros(self.n_nodes, self.dtype_indices)
    self.threshold_array = np.zeros(self.n_nodes, np.float32)
    self.value_array = np.zeros(self.n_nodes, self.dtype_labels) 
    self.feature_array = np.zeros(self.n_nodes, np.uint16)
    """

    """
    assert self.root != None
    recursive_decorate(self.root)
    """
    """
    print np.all(self.left_children[0:self.n_nodes] == self.left_child_array)
    print np.all(self.right_children[0:self.n_nodes] == self.right_child_array)
    print np.all(self.feature_idx_array[0:self.n_nodes] == self.feature_array)
    print np.all(self.feature_threshold_array[0:self.n_nodes] == self.threshold_array)
    print np.all(self.values_array[0:self.n_nodes] == self.value_array)
    """
    

    self.left_child_array = self.left_children
    self.right_child_array = self.right_children
    self.feature_array = self.feature_idx_array
    self.value_array = self.values_array
    self.threshold_array = self.feature_threshold_array
    self.root = None
