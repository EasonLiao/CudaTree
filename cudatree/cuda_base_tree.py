import numpy as np
from util import get_best_dtype
from pycuda import gpuarray

class BaseTree(object):
  def __init__(self):
    self.root = None
    self.max_depth = None

  def print_tree(self):
    assert False

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
    left_child_gpu = gpuarray.to_gpu(self.left_children)
    right_child_gpu = gpuarray.to_gpu(self.right_children)
    threshold_gpu = gpuarray.to_gpu(self.feature_threshold_array)
    value_gpu = gpuarray.to_gpu(self.values_array)
    feature_gpu = gpuarray.to_gpu(self.feature_idx_array)
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

  def _find_most_common_label(self, x):
    return np.argmax(np.bincount(x))

  def gpu_decorate_nodes(self, samples, labels):
    self.left_children = self.left_children[0 : self.n_nodes]
    self.right_children = self.right_children[0 : self.n_nodes]
    self.feature_threshold_array = self.feature_threshold_array[0 : self.n_nodes]
    self.feature_idx_array = self.feature_idx_array[0 : self.n_nodes]
