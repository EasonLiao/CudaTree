import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from sklearn import tree
import numpy as np
import math
from time import sleep
import datasource 
from util import timer, dtype_to_ctype, mk_kernel, get_best_dtype
from node import Node
from cuda_base_tree import BaseTree

class DecisionTree(BaseTree): 
  COMPT_THREADS_PER_BLOCK = 32  #The number of threads do computation per block
  RESHUFFLE_THREADS_PER_BLOCK = 32

  def __init__(self):
    self.root = None
    self.compt_kernel_type = None
    self.num_labels = None
    self.label_count = None
    self.max_depth = None
    self.stride = None
    self.dtype_labels = None
    self.dtype_samples = None
    self.dtype_indices = None
    self.dtype_counts = None

  def fit(self, samples, target, max_depth = None): 
    def compile_kernels():
      ctype_indices = dtype_to_ctype(self.dtype_indices)
      ctype_labels = dtype_to_ctype(self.dtype_labels)
      ctype_counts = dtype_to_ctype(self.dtype_counts)
      ctype_samples = dtype_to_ctype(self.dtype_samples)
      n_labels = self.num_labels
      n_threads = self.COMPT_THREADS_PER_BLOCK
      n_shf_threads = self.RESHUFFLE_THREADS_PER_BLOCK

      self.kernel = mk_kernel((n_threads, n_labels, ctype_samples, ctype_labels, ctype_counts), "compute", "comput_kernel_pp_part.cu")
      self.scan_kernel = mk_kernel((n_labels, n_threads, ctype_labels, ctype_counts), "prefix_scan", "scan_kernel_p_part.cu")
      self.fill_kernel = mk_kernel((ctype_indices,), "fill_table", "fill_table.cu")
      self.shuffle_kernel = mk_kernel((ctype_indices, ctype_samples, ctype_labels, n_shf_threads), "scan_reshuffle", "pos_scan_reshuffle_c.cu")
      self.count_total_kernel = mk_kernel((n_threads, n_labels,  ctype_labels, ctype_counts), "count_total", "scan_kernel_total.cu")
      self.comput_total_kernel = mk_kernel((n_threads, n_labels,  ctype_samples, ctype_labels, ctype_counts), "compute", "comput_kernel_total.cu")
      self.comput_label_loop_kernel = mk_kernel((n_threads, n_labels, ctype_samples, ctype_labels, ctype_counts), "compute", "comput_kernel_label_loop.cu") 
      """ Use prepare to improve speed """
      self.kernel.prepare("PPPPPPiii")
      self.scan_kernel.prepare("PPiii") 
      self.fill_kernel.prepare("PiiPi")
      self.shuffle_kernel.prepare("PPPPPPPiii")
      #self.shuffle_kernel.prepare("PPPPPPPiii")
      #self.scan_comput_kernel.prepare("PPPPPPiiii") 
      self.count_total_kernel.prepare("PPi")
      self.comput_total_kernel.prepare("PPPPPPii") 
      self.comput_label_loop_kernel.prepare("PPPPPPii") 
      
    def allocate_gpuarrays():
      """ Pre-allocate the GPU memory, don't allocate everytime in __construct"""
      self.impurity_left = gpuarray.empty(self.n_features, dtype = np.float32)
      self.impurity_right = gpuarray.empty(self.n_features, dtype = np.float32)
      self.min_split = gpuarray.empty(self.n_features, dtype = self.dtype_counts)
      self.mark_table = gpuarray.empty(target.size, dtype = np.uint8)   
      self.label_count = gpuarray.empty((self.COMPT_THREADS_PER_BLOCK + 1) * self.num_labels * samples.shape[0], dtype = self.dtype_counts)  
      self.label_total = gpuarray.empty(self.num_labels, dtype = self.dtype_counts)
    
    def sort_samples():
      sorted_indices = np.empty((self.n_features, target.size), dtype = self.dtype_indices)
      sorted_labels = np.empty((self.n_features, target.size), dtype = self.dtype_labels)
      sorted_samples = np.empty((self.n_features, target.size), dtype = self.dtype_samples)
      
      with timer("argsort"):
        for i,f in enumerate(samples):
          sort_idx = np.argsort(f)
          sorted_indices[i] = sort_idx  
          sorted_labels[i] = target[sort_idx]
          sorted_samples[i] = samples[i][sort_idx]
    
      self.sorted_samples_gpu = gpuarray.to_gpu(sorted_samples)
      self.sorted_indices_gpu = gpuarray.to_gpu(sorted_indices)
      self.sorted_labels_gpu = gpuarray.to_gpu(sorted_labels)    
      self.sorted_samples_gpu_ = self.sorted_samples_gpu.copy() 
      self.sorted_indices_gpu_ = self.sorted_indices_gpu.copy()
      self.sorted_labels_gpu_ = self.sorted_labels_gpu.copy()  

    assert isinstance(samples, np.ndarray)
    assert isinstance(target, np.ndarray)
    assert samples.size / samples[0].size == target.size
    
    self.max_depth = max_depth
    self.num_labels = int(np.max(target)) + 1    
    self.stride = target.size
  
    self.dtype_indices = get_best_dtype(target.size)
    self.dtype_counts = self.dtype_indices
    self.dtype_labels = get_best_dtype(self.num_labels)
    self.dtype_samples = samples.dtype
      
    samples = np.require(np.transpose(samples), requirements = 'C')
    target = np.require(np.transpose(target), dtype = self.dtype_labels, requirements = 'C') 
    
    self.samples_itemsize = self.dtype_samples.itemsize
    self.labels_itemsize = self.dtype_labels.itemsize
    self.target_value = np.zeros(1, self.dtype_labels) #Retrive the label from gpu array at the leaf node.
    self.threshold_values = np.zeros(2, self.dtype_samples) #Retrive the sample threshold values from gpu array.
    self.n_features = samples.shape[0] 

    compile_kernels() 
    allocate_gpuarrays()
    sort_samples() 
  
    assert self.sorted_indices_gpu.strides[0] == target.size * self.sorted_indices_gpu.dtype.itemsize 
    assert self.sorted_labels_gpu.strides[0] == target.size * self.sorted_labels_gpu.dtype.itemsize 
    assert self.sorted_samples_gpu.strides[0] == target.size * self.sorted_samples_gpu.dtype.itemsize  
    self.root = self.__construct(1, 1.0, 0, target.size, (self.sorted_indices_gpu, self.sorted_labels_gpu, self.sorted_samples_gpu),
                                  (self.sorted_indices_gpu_, self.sorted_labels_gpu_, self.sorted_samples_gpu_)) 
    

  def __construct(self, depth, error_rate, start_idx, stop_idx, gpuarrays_in, gpuarrays_out):
    def check_terminate():
      if error_rate == 0 or (self.max_depth is not None and depth > self.max_depth):
        return True
      else:
        return False 
    
    n_samples = stop_idx - start_idx
    labels_offset = start_idx * self.labels_itemsize
    samples_offset = start_idx * self.samples_itemsize
    indices_offset =  start_idx * self.dtype_indices.itemsize
 
    ret_node = Node()
    ret_node.error = error_rate
    ret_node.samples = n_samples 
    ret_node.height = depth 

    if check_terminate():
      cuda.memcpy_dtoh(self.target_value, gpuarrays_in[1].ptr + int(self.labels_itemsize * start_idx))
      ret_node.value = self.target_value[0]
      return ret_node
  
    block = (self.COMPT_THREADS_PER_BLOCK, 1, 1)
    grid = (self.n_features, 1)
    range_size = int(math.ceil(float(n_samples) / self.COMPT_THREADS_PER_BLOCK))
    n_active_threads = int(math.ceil(float(n_samples) / range_size))
   
    """
    self.scan_kernel.prepared_call(
                grid,
                block,
                gpuarrays_in[1].ptr + labels_offset, 
                self.label_count.ptr,
                self.n_features, 
                n_samples, 
                self.stride) 
    """
    self.count_total_kernel.prepared_call(
                (1, 1),
                block,
                gpuarrays_in[1].ptr + labels_offset,
                self.label_total.ptr,
                n_samples)
    
    self.comput_label_loop_kernel.prepared_call(
              grid,
              block,
              gpuarrays_in[2].ptr + samples_offset,
              gpuarrays_in[1].ptr + labels_offset,
              self.impurity_left.ptr,
              self.impurity_right.ptr,
              self.label_total.ptr,
              self.min_split.ptr,
              n_samples,
              self.stride) 
    """
    self.comput_total_kernel.prepared_call(
              grid,
              block,
              gpuarrays_in[2].ptr + samples_offset,
              gpuarrays_in[1].ptr + labels_offset,
              self.impurity_left.ptr,
              self.impurity_right.ptr,
              self.label_total.ptr,
              self.min_split.ptr,
              n_samples,
              self.stride)
    """
    #  self.min_split.get() 
    
    """ 
    self.kernel.prepared_call(
              grid,
              block,
              gpuarrays_in[2].ptr + samples_offset,
              gpuarrays_in[1].ptr + labels_offset,
              self.impurity_left.ptr,
              self.impurity_right.ptr,
              self.label_count.ptr,
              self.min_split.ptr,
              self.n_features, 
              n_samples, 
              self.stride)
    self.scan_comput_kernel.prepared_call(
              grid,
              block,
              gpuarrays_in[1].ptr + labels_offset,
              gpuarrays_in[2].ptr + samples_offset,
              self.label_count.ptr,
              self.impurity_left.ptr,
              self.impurity_right.ptr,
              self.min_split.ptr,
              range_size,
              n_active_threads,
              n_samples,
              self.stride)
    """
    imp_right = self.impurity_right.get()  
    imp_left = self.impurity_left.get()
    imp_total = imp_left + imp_right
    ret_node.feature_index =  imp_total.argmin()
  

    if imp_total[ret_node.feature_index] == 4:
      return ret_node

    row = ret_node.feature_index
    col = self.min_split.get()[row]
    
    #Record the feature threshold, only transfer a small portion of gpu memory to host memory.
    cuda.memcpy_dtoh(self.threshold_values, gpuarrays_in[2].ptr + int(samples_offset) + int(self.stride * row * self.samples_itemsize + col * self.samples_itemsize))
    ret_node.feature_threshold = (float(self.threshold_values[0]) + float(self.threshold_values[1])) / 2 

    block = (self.RESHUFFLE_THREADS_PER_BLOCK, 1, 1)

    self.fill_kernel.prepared_call(
                      (1, 1),
                      (1024, 1, 1),
                      gpuarrays_in[0].ptr + row * self.stride * self.dtype_indices.itemsize + indices_offset, 
                      n_samples, 
                      col, 
                      self.mark_table.ptr, 
                      self.stride
                      )
    self.shuffle_kernel.prepared_call(
                      (self.n_features, 1),
                      block,
                      self.mark_table.ptr,
                      gpuarrays_in[0].ptr + indices_offset,
                      gpuarrays_in[2].ptr + samples_offset,
                      gpuarrays_in[1].ptr + labels_offset,
                      gpuarrays_out[0].ptr + indices_offset,
                      gpuarrays_out[2].ptr + samples_offset,
                      gpuarrays_out[1].ptr + labels_offset,
                      n_samples,
                      col,
                      self.stride)
    ret_node.left_child = self.__construct(depth + 1, imp_left[ret_node.feature_index], start_idx, start_idx + col + 1, gpuarrays_out, gpuarrays_in)
    ret_node.right_child = self.__construct(depth + 1, imp_right[ret_node.feature_index], start_idx + col + 1, stop_idx, gpuarrays_out, gpuarrays_in)
    return ret_node 



if __name__ == "__main__": 
  x_train, y_train = datasource.load_data("iris") 
  
  """
  with timer("Scikit-learn"):
    clf = tree.DecisionTreeClassifier()    
    clf = clf.fit(x_train, y_train)  
  """
  with timer("Cuda"):
    d = DecisionTree()  
    d.fit(x_train, y_train)
    d.print_tree()
    print np.allclose(d.predict(x_train), y_train)
