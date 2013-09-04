import numpy as np
from util import get_best_dtype

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
    res = []
    for val in inputs:
      res.append(self.__gpu_predict(val))
    return np.array(res)

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

  def gpu_decorate_nodes(self, samples, labels):
    """ Decorate the nodes for GPU predicting """
    def recursive_decorate(node):
      if node.left_child and node.right_child:
        idx_tuple = node.feature_threshold
        node.feature_threshold = (float(samples[idx_tuple[0], idx_tuple[1]]) + samples[idx_tuple[0], idx_tuple[2]]) / 2
        
        self.left_child_array[node.nid] = node.left_nid
        self.right_child_array[node.nid] = node.right_nid
        self.threshold_array[node.nid] = node.feature_threshold
        self.feature_array[node.nid] = node.feature_index

        recursive_decorate(node.left_child)
        recursive_decorate(node.right_child)
      else:
        idx = node.value
        node.value = labels[idx]
        self.value_array[node.nid] = node.value

    dtype_idx = get_best_dtype(self.n_nodes)
    
    self.left_child_array = np.zeros(self.n_nodes, dtype_idx)
    self.right_child_array = np.zeros(self.n_nodes, dtype_idx)
    self.threshold_array = np.zeros(self.n_nodes, np.float32)
    self.value_array = np.zeros(self.n_nodes, self.dtype_labels) 
    self.feature_array = np.zeros(self.n_nodes, np.uint16)

    assert self.root != None
    recursive_decorate(self.root)
    print self.left_child_array
    print self.right_child_array
    print self.threshold_array
    print self.value_array
    print self.feature_array
