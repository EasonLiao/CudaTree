import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from sklearn import tree
import numpy as np
import math
from time import sleep
import datasource
from util import mk_kernel, mk_tex_kernel, timer, dtype_to_ctype, get_best_dtype, start_timer, end_timer
from node import Node
from cuda_random_base_tree import RandomBaseTree
from collections import deque

class RandomDecisionTreeSmall(RandomBaseTree): 
  def __init__(self, samples_gpu, labels_gpu, sorted_indices, compt_table, dtype_labels, dtype_samples, 
      dtype_indices, dtype_counts, n_features, n_samples, n_labels, n_threads, n_shf_threads, max_features = None,
      max_depth = None, min_samples_split = None):
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
    self.min_samples_split =  min_samples_split

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
    
    #self.comput_total_kernel = mk_kernel((n_threads, n_labels, ctype_samples, ctype_labels, 
    #  ctype_counts, ctype_indices), "compute", "comput_kernel_total_rand.cu")
     
    self.scan_reshuffle_tex, tex_ref = mk_tex_kernel((ctype_indices, n_shf_threads), 
        "scan_reshuffle", "tex_mark", "pos_scan_reshuffle_si_c_tex.cu")   
    self.mark_table.bind_to_texref_ext(tex_ref) 
    
    #self.comput_label_loop_kernel = mk_kernel((n_threads, n_labels, ctype_samples, 
    #  ctype_labels, ctype_counts, ctype_indices), "compute",  "comput_kernel_label_loop_si.cu") 
    
    self.comput_label_loop_rand_kernel = mk_kernel((n_threads, n_labels, ctype_samples, 
      ctype_labels, ctype_counts, ctype_indices), "compute",  "comput_kernel_label_loop_rand.cu") 
    
    self.find_min_kernel = mk_kernel((ctype_counts, 32), "find_min_imp", "find_min_gini.cu")
      
    self.predict_kernel = mk_kernel((ctype_indices, ctype_samples, ctype_labels), "predict", "predict.cu")
  
    self.scan_total_bfs = mk_kernel((32, n_labels, ctype_labels, ctype_counts, ctype_indices), "count_total", "scan_kernel_total_bfs.cu")
  
    self.comput_bfs = mk_kernel((32, n_labels, ctype_samples, ctype_labels, ctype_counts, ctype_indices), "compute", "comput_kernel_bfs.cu")
    
    self.fill_bfs = mk_kernel((ctype_indices,), "fill_table", "fill_table_bfs.cu")
    
    self.reshuffle_bfs = mk_kernel((ctype_indices, 32), "scan_reshuffle", "pos_scan_reshuffle_bfs.cu")
    
    if hasattr(self.fill_kernel, "is_prepared"):
      return
    
    self.fill_kernel.is_prepared = True
    self.fill_kernel.prepare("PiiPi")
    self.scan_reshuffle_tex.prepare("PPPiii") 
    self.scan_total_kernel.prepare("PPPi")
    #self.comput_total_kernel.prepare("PPPPPPPPii")
    #self.comput_label_loop_kernel.prepare("PPPPPPPii")
    self.comput_label_loop_rand_kernel.prepare("PPPPPPPPii")
    self.find_min_kernel.prepare("PPPi")
    self.predict_kernel.prepare("PPPPPPPii")
    self.scan_total_bfs.prepare("PPPPPPPi")
    self.comput_bfs.prepare("PPPPPPPPPPPii")
    self.fill_bfs.prepare("PPPPPPPi")
    self.reshuffle_bfs.prepare("PPPPPPPii")

  def __allocate_gpuarrays(self):
    if self.max_features < 4:
      imp_size = 4
    else:
      imp_size = self.max_features
    self.impurity_left = gpuarray.empty(imp_size, dtype = np.float32)
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
    self.fill_kernel = None
    self.scan_reshuffle_tex = None 
    self.scan_total_kernel = None
    self.comput_label_loop_rand_kernel = None
    self.find_min_kernel = None
    self.scan_total_bfs = None
    self.comput_bfs = None
    self.fill_bfs = None
    self.reshuffle_bfs = None
  
  def __bfs_construct(self):
    while len(self.queue):
      self.__bfs()

  def __bfs(self):
    idx_array = np.zeros(len(self.queue) * 2, dtype = self.dtype_indices)
    si_idx_array = np.zeros(len(self.queue), dtype = np.uint8) 
    subset_indices_array = np.zeros(len(self.queue) * self.max_features, dtype = self.dtype_indices)
    
    queue_size = len(self.queue)
    
    for i, node in enumerate(self.queue):
      idx_array[i * 2] = node.start_idx
      idx_array[i * 2 + 1] = node.stop_idx
      si_idx_array[i] = node.si_idx
      subset_indices_array[i * self.max_features : (i + 1) * self.max_features] = node.subset_indices  
      node.subset_indices = None

    idx_array_gpu = gpuarray.to_gpu(idx_array)
    si_idx_array_gpu = gpuarray.to_gpu(si_idx_array)
    subset_indices_array_gpu = gpuarray.to_gpu(subset_indices_array)
    min_feature_idx_gpu = gpuarray.empty(queue_size, dtype = np.uint16)
    
    self.label_total = gpuarray.empty(queue_size * self.n_labels, dtype = self.dtype_counts)  
    impurity_gpu = gpuarray.empty(queue_size * 2, dtype = np.float32)
    self.min_split = gpuarray.empty(queue_size, dtype = self.dtype_indices)

    if len(self.mark_table.shape) == 1:
      self.mark_table = gpuarray.zeros((self.n_features, self.stride), dtype=np.uint8)
    
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
    
    self.comput_bfs.prepared_call(
          (queue_size, 1),
          (32, 1, 1),
          self.samples_gpu.ptr,
          self.labels_gpu.ptr,
          self.sorted_indices_gpu.ptr,
          self.sorted_indices_gpu_.ptr,
          idx_array_gpu.ptr,
          si_idx_array_gpu.ptr,
          self.label_total.ptr,
          subset_indices_array_gpu.ptr,
          impurity_gpu.ptr,
          self.min_split.ptr,
          min_feature_idx_gpu.ptr,
          self.max_features,
          self.stride)
    
    min_split = self.min_split.get()
    imp_min = impurity_gpu.get()
    feature_idx = min_feature_idx_gpu.get()
    begin_end_idx = idx_array

    self.fill_bfs.prepared_call(
          (queue_size, 1),
          (32, 1, 1),
          self.sorted_indices_gpu.ptr,
          self.sorted_indices_gpu_.ptr,
          si_idx_array_gpu.ptr,
          min_feature_idx_gpu.ptr,
          idx_array_gpu.ptr,
          self.min_split.ptr,
          self.mark_table.ptr,
          self.stride)
     
    self.reshuffle_bfs.prepared_call(
          (queue_size, 128),
          (32, 1, 1),
          self.mark_table.ptr,
          si_idx_array_gpu.ptr,
          self.sorted_indices_gpu.ptr,
          self.sorted_indices_gpu_.ptr,
          idx_array_gpu.ptr,
          self.min_split.ptr,
          min_feature_idx_gpu.ptr,
          self.n_features,
          self.stride)
    
    """ While the GPU is being utilized, run some CPU intensive code on CPU"""
    for i, node in enumerate(self.queue):
      si_id = 1 - node.si_idx 
      node.left_child = Node()
      node.right_child = Node()
      node.left_child.depth = node.depth + 1
      node.right_child.depth = node.depth + 1
      node.left_child.si_idx = si_id
      node.right_child.si_idx = si_id
      if imp_min[2 * i] != 0.0:
        node.left_child.subset_indices = self.get_indices()
      if imp_min[2 * i + 1] != 0.0:
        node.right_child.subset_indices = self.get_indices()

    for i in xrange(queue_size):
      node = self.queue.popleft()
      node.feature_index = feature_idx[i]
      
      if node.si_idx == 1:
        si = self.sorted_indices_gpu
      else:
        si = self.sorted_indices_gpu_

      row = node.feature_index
      col = min_split[i]
      left_imp = imp_min[2 * i]
      right_imp = imp_min[2 * i + 1]

      cuda.memcpy_dtoh(self.threshold_value_idx, si.ptr +  
          int(row * self.stride + col) * int(self.dtype_indices.itemsize))
      node.feature_threshold = (row, self.threshold_value_idx[0], self.threshold_value_idx[1])
     
      if left_imp + right_imp == 4.0:
        node.left_child = None
        node.right_child = None
        self.__record_leaf(node, node.start_idx, node.stop_idx - node.start_idx, si)
        continue
       
      left_node =  node.left_child
      right_node = node.right_child
      
      left_node.nid = self.n_nodes 
      self.n_nodes += 1  
      
      right_node.nid = self.n_nodes
      self.n_nodes += 1
       
      if left_imp != 0.0:
        n_samples_left = col + 1 - node.start_idx 
        if n_samples_left < self.min_samples_split or (self.max_depth is not None and left_node.depth >= self.max_depth):
          left_node.subset_indices = None
          self.__record_leaf(left_node, node.start_idx, n_samples_left, si)
        else:
          left_node.start_idx = node.start_idx
          left_node.stop_idx = col + 1
          self.queue.append(left_node)
      else:
        cuda.memcpy_dtoh(self.target_value_idx, si.ptr + int(node.start_idx * self.dtype_indices.itemsize))
        left_node.subset_indices = None
        left_node.value = self.target_value_idx[0]

      if right_imp != 0.0:
        n_samples_right = node.stop_idx - col - 1
        if n_samples_right < self.min_samples_split or (self.max_depth is not None and right_node.depth >= self.max_depth):
          right_node.subset_indices = None
          self.__record_leaf(right_node, col + 1, n_samples_right, si)
        else:
          right_node.start_idx = col + 1
          right_node.stop_idx = node.stop_idx
          self.queue.append(right_node)
      else:
        cuda.memcpy_dtoh(self.target_value_idx, si.ptr + int((col + 1) * self.dtype_indices.itemsize)) 
        right_node.subset_indices = None
        right_node.value = self.target_value_idx[0]   

      node.left_child = left_node
      node.right_child = right_node

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
    
    self.samples = samples
    self.target = target
    self.left_children = np.zeros(self.stride * 2, dtype = self.dtype_indices)
    self.right_children = np.zeros(self.stride * 2, dtype = self.dtype_indices)
    self.feature_idx_array = np.zeros(2 *self.stride, dtype = np.uint16)
    self.feature_threshold_array = np.zeros(2 * self.stride, dtype = np.float32)
    self.values_array = np.zeros(2 * self.stride, dtype = self.dtype_labels)

    self.n_nodes = 0 
    self.root = self.__dfs_construct(1, 1.0, 0, target.size, self.sorted_indices_gpu, self.sorted_indices_gpu_, self.get_indices())  
    self.__bfs_construct() 
    self.__release_gpuarrays()
    self.gpu_decorate_nodes(samples, target)

  def __record_leaf(self, ret_node, start_idx, n_samples, si):
      """ Pick the indices to record on the leaf node. In decoration step, we'll choose the most common label """
      if n_samples < 3:
        cuda.memcpy_dtoh(self.target_value_idx, si.ptr + int(start_idx * self.dtype_indices.itemsize))
        ret_node.value = self.target_value_idx[0] 
      else:
        si_labels = np.empty(n_samples, dtype=self.dtype_indices)
        cuda.memcpy_dtoh(si_labels, si.ptr + int(start_idx * self.dtype_indices.itemsize))
        ret_node.value = si_labels
  
  def __turn_to_leaf(self, nid, start_idx, n_samples, si):
      """ Pick the indices to record on the leaf node. In decoration step, we'll choose the most common label """
      if n_samples < 3:
        cuda.memcpy_dtoh(self.target_value_idx, si.ptr + int(start_idx * self.dtype_indices.itemsize))
        self.values_array[nid] = self.target[self.target_value_idx[0]]
      else:
        si_labels = np.empty(n_samples, dtype=self.dtype_indices)
        cuda.memcpy_dtoh(si_labels, si.ptr + int(start_idx * self.dtype_indices.itemsize))
        self.values_array[nid]  = self.__find_most_common(self.targetp[si_labels])


  def __dfs_construct(self, depth, error_rate, start_idx, stop_idx, si_gpu_in, si_gpu_out, subset_indices):
    def check_terminate():
      if error_rate == 0:
        return True
      else:
        return False 

    n_samples = stop_idx - start_idx
    indices_offset =  start_idx * self.dtype_indices.itemsize
    

    nid = self.n_nodes

    self.n_nodes += 1

    if check_terminate():
      cuda.memcpy_dtoh(self.target_value_idx, si_gpu_in.ptr + int(start_idx * self.dtype_indices.itemsize))
      #ret_node.value = self.target_value_idx[0] 
      self.values_array[nid] = self.target[self.target_value_idx[0]]
      return
    
    if n_samples < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
      return
      self.__record_leaf(ret_node, start_idx, n_samples, si_gpu_in)
      return ret_node
    
    if n_samples <= 1:
      pass
      """
      print "!"
      ret_node.start_idx = start_idx
      ret_node.stop_idx = stop_idx
      ret_node.si_idx = si_gpu_in.idx
      ret_node.subset_indices = subset_indices
      self.queue.append(ret_node)
      return ret_node 
      """

    block = (self.COMPT_THREADS_PER_BLOCK, 1, 1)
    cuda.memcpy_htod(self.subset_indices.ptr, subset_indices)
    grid = (self.max_features, 1) 
    
    self.scan_total_kernel.prepared_call(
                (1, 1),
                block,
                si_gpu_in.ptr + indices_offset,
                self.labels_gpu.ptr,
                self.label_total.ptr,
                n_samples)

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
      self.__turn_to_leaf(nid, start_idx, n_samples, si_gpu_in) 
      return
     
    col = int(self.min_imp_info[2]) 
    row = int(self.min_imp_info[3])
    row = subset_indices[row]
    #ret_node.feature_index = row
   

    cuda.memcpy_dtoh(self.threshold_value_idx, si_gpu_in.ptr + int(indices_offset) + 
        int(row * self.stride + col) * int(self.dtype_indices.itemsize))
    #ret_node.feature_threshold = (row, self.threshold_value_idx[0], self.threshold_value_idx[1])
    
    self.feature_idx_array[nid] = row
    self.feature_threshold_array[nid] = (float(self.samples[row, self.threshold_value_idx[0]]) + self.samples[row, self.threshold_value_idx[1]]) / 2
   
    self.fill_kernel.prepared_call(
                      (1, 1),
                      (1024, 1, 1),
                      si_gpu_in.ptr + row * self.stride * self.dtype_indices.itemsize + indices_offset, 
                      n_samples, 
                      col, 
                      self.mark_table.ptr, 
                      self.stride)
       
    block = (self.RESHUFFLE_THREADS_PER_BLOCK, 1, 1)

    self.scan_reshuffle_tex.prepared_call(
                      (self.n_features, 1),
                      block,
                      self.mark_table.ptr,
                      si_gpu_in.ptr + indices_offset,
                      si_gpu_out.ptr + indices_offset,
                      n_samples,
                      col,
                      self.stride) 

    self.left_children[nid] = self.n_nodes
    self.__dfs_construct(depth + 1, min_left, 
        start_idx, start_idx + col + 1, si_gpu_out, si_gpu_in, subset_indices_left)
    
    self.right_children[nid] = self.n_nodes
    self.__dfs_construct(depth + 1, min_right, 
        start_idx + col + 1, stop_idx, si_gpu_out, si_gpu_in, subset_indices_right)

