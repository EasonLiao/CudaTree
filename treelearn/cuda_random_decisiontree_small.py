import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from sklearn import tree
import numpy as np
import math
from time import sleep
import datasource
from util import mk_kernel, mk_tex_kernel, timer, dtype_to_ctype, get_best_dtype
from node import Node
from cuda_random_base_tree import RandomBaseTree
from collections import deque

class RandomDecisionTreeSmall(RandomBaseTree): 
  def __init__(self, samples_gpu, labels_gpu, sorted_indices, compt_table, dtype_labels, dtype_samples, 
      dtype_indices, dtype_counts, n_features, n_samples, n_labels, n_threads, n_shf_threads, max_features = None,
      max_depth = None):
    self.root = None
    self.n_labels = n_labels
    self.max_depth = None
    self.stride = n_samples
    self.dtype_labels = dtype_labels
    self.dtype_samples = dtype_samples
    self.dtype_indices = dtype_indices
    self.dtype_counts = dtype_counts
    self.n_features = n_features
    self.COMPT_THREADS_PER_BLOCK = n_threads
    self.RESHUFFLE_THREADS_PER_BLOCK = n_shf_threads
    self.samples_gpu = samples_gpu
    self.labels_gpu = labels_gpu
    self.sorted_indices = sorted_indices
    self.compt_table = compt_table
    self.max_depth = max_depth
    self.max_features = max_features

  def __compile_kernels(self):
    ctype_indices = dtype_to_ctype(self.dtype_indices)
    ctype_labels = dtype_to_ctype(self.dtype_labels)
    ctype_counts = dtype_to_ctype(self.dtype_counts)
    ctype_samples = dtype_to_ctype(self.dtype_samples)
    n_labels = self.n_labels
    n_threads = self.COMPT_THREADS_PER_BLOCK
    n_shf_threads = self.RESHUFFLE_THREADS_PER_BLOCK
    
    self.fill_kernel = mk_kernel((ctype_indices,), "fill_table", "fill_table_si.cu") 
    self.scan_total_kernel = mk_kernel((n_threads, n_labels, ctype_labels, ctype_counts, ctype_indices), 
        "count_total", "scan_kernel_total_si.cu") 
    
    self.comput_total_kernel = mk_kernel((n_threads, n_labels, ctype_samples, ctype_labels, 
      ctype_counts, ctype_indices), "compute", "comput_kernel_total_rand.cu")
     
    self.scan_reshuffle_tex, tex_ref = mk_tex_kernel((ctype_indices, n_shf_threads), 
        "scan_reshuffle", "tex_mark", "pos_scan_reshuffle_si_c_tex.cu")   
    self.mark_table.bind_to_texref_ext(tex_ref) 
    
    self.comput_label_loop_kernel = mk_kernel((n_threads, n_labels, ctype_samples, 
      ctype_labels, ctype_counts, ctype_indices), "compute",  "comput_kernel_label_loop_si.cu") 
    
    self.comput_label_loop_rand_kernel = mk_kernel((n_threads, n_labels, ctype_samples, 
      ctype_labels, ctype_counts, ctype_indices), "compute",  "comput_kernel_label_loop_rand.cu") 
    
    self.find_min_kernel = mk_kernel((ctype_counts, 32), "find_min_imp", "find_min_gini.cu")
      
    self.predict_kernel = mk_kernel((ctype_indices, ctype_samples, ctype_labels), "predict", "predict.cu")
  
    self.scan_total_bfs = mk_kernel((n_threads, n_labels, ctype_labels, ctype_counts, ctype_indices), "count_total", "scan_kernel_total_bfs.cu")

    if hasattr(self.fill_kernel, "is_prepared"):
      return
    
    self.fill_kernel.is_prepared = True


    self.fill_kernel.prepare("PiiPi")
    self.scan_reshuffle_tex.prepare("PPPiii") 
    self.scan_total_kernel.prepare("PPPi")
    self.comput_total_kernel.prepare("PPPPPPPPii")
    self.comput_label_loop_kernel.prepare("PPPPPPPii")
    self.comput_label_loop_rand_kernel.prepare("PPPPPPPPii")
    self.find_min_kernel.prepare("PPPi")
    self.predict_kernel.prepare("PPPPPPPii")
    self.scan_total_bfs.prepare("PPPPPPPi")

  def __allocate_gpuarrays(self):
    self.impurity_left = gpuarray.empty(self.max_features, dtype = np.float32)
    self.impurity_right = gpuarray.empty(self.max_features, dtype = np.float32)
    self.min_split = gpuarray.empty(self.max_features, dtype = self.dtype_counts)
    self.mark_table = gpuarray.empty(self.stride, dtype = np.uint8)
    self.label_total = gpuarray.empty(self.n_labels, self.dtype_indices)  
    self.subset_indices = gpuarray.empty(self.max_features, dtype = self.dtype_indices)
  
  def __release_gpuarrays(self):
    self.impurity_left = None
    self.impurity_right = None
    self.min_split = None
    self.mark_table = None
    self.label_total = None
    self.subset_indices = None
    self.sorted_indices_gpu = None
    self.sorted_indices_gpu_ = None
   

  def __dfs(self):
    idx_array = np.zeros(len(self.queue) * 2, dtype = self.dtype_indices)
    si_idx_array = np.zeros(len(self.queue), dtype = np.uint8) 
    subset_indices_array = np.zeros(len(self.queue) * self.max_features, dtype = self.dtype_indices)
    queue_size = len(self.queue)
    
    label_total = np.zeros(queue_size * self.n_labels, dtype = self.dtype_counts)

    for i, node in enumerate(self.queue):
      idx_array[i * 2] = node.start_idx
      idx_array[i * 2 + 1] = node.stop_idx
      si_idx_array[i] = node.si_idx
      subset_indices_array[i * self.max_features : (i + 1) * self.max_features] = node.subset_indices  
      label_total[i * self.n_labels : (i + 1) * self.n_labels] = node.label_total 
    
    """
    print si_idx_array
    print idx_array
    print subset_indices_array.shape, len(self.queue)
    """

    idx_array_gpu = gpuarray.to_gpu(idx_array)
    si_idx_array_gpu = gpuarray.to_gpu(si_idx_array)
    subset_indices_array_gpu = gpuarray.to_gpu(subset_indices_array)
    self.label_total = gpuarray.empty(queue_size * self.n_labels, dtype = self.dtype_counts)  
  
    self.scan_total_bfs.prepared_call(
            (queue_size, 1),
            (32, 1, 1),
            self.sorted_indices_gpu.ptr,
            self.sorted_indices_gpu_.ptr,
            self.labels_gpu.ptr,
            self.label_total.ptr,
            si_idx_array_gpu.ptr,
            idx_array_gpu.ptr,
            subset_indices_array_gpu.ptr,
            self.max_features) 
    
    print queue_size     
    print np.all(self.label_total.get() == label_total)
    print "end"

  def fit(self, samples, target): 
    self.samples_itemsize = self.dtype_samples.itemsize
    self.labels_itemsize = self.dtype_labels.itemsize
    self.target_value_idx = np.zeros(1, self.dtype_indices)
    self.threshold_value_idx = np.zeros(2, self.dtype_indices)
    self.min_imp_info = np.zeros(4, dtype = np.float32)  
    self.queue = deque()

    if self.max_features is None:
      self.max_features = int(math.ceil(math.log(self.n_features, 2)))
    
    assert self.max_features > 0 and self.max_features <= self.n_features, "max_features must be between 0 and n_features." 
    self.__allocate_gpuarrays()
    self.__compile_kernels() 
    self.sorted_indices_gpu = gpuarray.to_gpu(self.sorted_indices)
    self.sorted_indices_gpu_ = self.sorted_indices_gpu.copy()
    
    self.sorted_indices_gpu.idx = 0
    self.sorted_indices_gpu_.idx = 1

    assert self.sorted_indices_gpu.strides[0] == target.size * self.sorted_indices_gpu.dtype.itemsize 
    assert self.samples_gpu.strides[0] == target.size * self.samples_gpu.dtype.itemsize   
    
    self.n_nodes = 0
    
    self.root = self.__construct(1, 1.0, 0, target.size, self.sorted_indices_gpu, self.sorted_indices_gpu_, self.get_indices()) 
    self.__dfs()
    self.__release_gpuarrays()
    #self.gpu_decorate_nodes(samples, target)

  def __construct(self, depth, error_rate, start_idx, stop_idx, si_gpu_in, si_gpu_out, subset_indices):
    def check_terminate():
      if error_rate == 0 or (self.max_depth is not None and depth > self.max_depth):
        return True
      else:
        return False 

    n_samples = stop_idx - start_idx
    indices_offset =  start_idx * self.dtype_indices.itemsize
 
    ret_node = Node()
    ret_node.error = error_rate
    ret_node.samples = n_samples 
    ret_node.height = depth 
    ret_node.nid = self.n_nodes
    
    self.n_nodes += 1

    if check_terminate():
      cuda.memcpy_dtoh(self.target_value_idx, si_gpu_in.ptr + int(start_idx * self.dtype_indices.itemsize))
      ret_node.value = self.target_value_idx[0] 
      return ret_node
   

    cuda.memcpy_htod(self.subset_indices.ptr, subset_indices)
    grid = (self.max_features, 1) 
    
    if n_samples >= self.COMPT_THREADS_PER_BLOCK:
      block = (self.COMPT_THREADS_PER_BLOCK, 1, 1)
    else:
      block = (32, 1, 1)

    self.scan_total_kernel.prepared_call(
                (1, 1),
                block,
                si_gpu_in.ptr + indices_offset,
                self.labels_gpu.ptr,
                self.label_total.ptr,
                n_samples)
    
    if n_samples <= 32:
      ret_node.start_idx = start_idx
      ret_node.stop_idx = stop_idx
      ret_node.si_idx = si_gpu_in.idx
      ret_node.subset_indices = subset_indices
      ret_node.label_total = self.label_total.get()
      self.queue.append(ret_node)
      return ret_node
    
    self.comput_label_loop_rand_kernel.prepared_call(
                grid,
                block,
                si_gpu_in.ptr + indices_offset,
                self.samples_gpu.ptr,
                self.labels_gpu.ptr,
                self.impurity_left.ptr,
                self.impurity_right.ptr,
                self.label_total.ptr,
                self.min_split.ptr,
                self.subset_indices.ptr,
                n_samples,
                self.stride)
    
    #self.comput_total_kernel.prepared_call(
    #            grid,
    #            block,
    #            si_gpu_in.ptr + indices_offset,
    #            self.samples_gpu.ptr,
    #            self.labels_gpu.ptr,
    #            self.impurity_left.ptr,
    #            self.impurity_right.ptr,
    #            self.label_total.ptr,
    #            self.min_split.ptr,
    #            self.subset_indices.ptr,
    #            n_samples,
    #            self.stride)

    subset_indices_left = self.get_indices()
    subset_indices_right = self.get_indices()
    
    self.find_min_kernel.prepared_call(
                (1, 1),
                (32, 1, 1),
                self.impurity_left.ptr,
                self.impurity_right.ptr,
                self.min_split.ptr,
                self.max_features)
    
    cuda.memcpy_dtoh(self.min_imp_info, self.impurity_left.ptr)
    min_right = self.min_imp_info[1] 
    min_left = self.min_imp_info[0] 
    
    if min_left + min_right == 4:
      print "imp min == 4, n_samples : %s" % (n_samples, )
      cuda.memcpy_dtoh(self.target_value_idx, si_gpu_in.ptr + int(start_idx * self.dtype_indices.itemsize))
      ret_node.value = self.target_value_idx[0] 
      return ret_node
     
    col = int(self.min_imp_info[2]) 
    row = int(self.min_imp_info[3])
    row = subset_indices[row]
    ret_node.feature_index = row
    
    cuda.memcpy_dtoh(self.threshold_value_idx, si_gpu_in.ptr + int(indices_offset) + 
        int(row * self.stride + col) * int(self.dtype_indices.itemsize))
    ret_node.feature_threshold = (row, self.threshold_value_idx[0], self.threshold_value_idx[1])
   

    self.fill_kernel.prepared_call(
                      (1, 1),
                      (1024, 1, 1),
                      si_gpu_in.ptr + row * self.stride * self.dtype_indices.itemsize + indices_offset, 
                      n_samples, 
                      col, 
                      self.mark_table.gpudata, 
                      self.stride
                      )
      
    
    if n_samples >= self.RESHUFFLE_THREADS_PER_BLOCK:
      block = (self.RESHUFFLE_THREADS_PER_BLOCK, 1, 1)
    else:
      block = (32, 1, 1)

    self.scan_reshuffle_tex.prepared_call(
                      (self.n_features, 1),
                      block,
                      self.mark_table.ptr,
                      si_gpu_in.ptr + indices_offset,
                      si_gpu_out.ptr + indices_offset,
                      n_samples,
                      col,
                      self.stride) 

    ret_node.left_nid = self.n_nodes
    ret_node.left_child = self.__construct(depth + 1, min_left, 
        start_idx, start_idx + col + 1, si_gpu_out, si_gpu_in, subset_indices_left)
    
    ret_node.right_nid = self.n_nodes
    ret_node.right_child = self.__construct(depth + 1, min_right, 
        start_idx + col + 1, stop_idx, si_gpu_out, si_gpu_in, subset_indices_right)
    return ret_node 

