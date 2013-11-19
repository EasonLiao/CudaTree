import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
import numpy as np
import math
from util import total_times, compile_module, mk_kernel, mk_tex_kernel, timer
from util import  dtype_to_ctype, get_best_dtype, start_timer, end_timer
from base_tree import BaseTree
from pycuda import driver
import random
from parakeet import jit
from util import start_timer, end_timer, show_timings
import sys
import util

def sync():
  if False:
    driver.Context.synchronize()

@jit
def  _shuffle(x, r):
  for i in xrange(1, len(x)):
    j = np.fmod(r[i], i)
    old_xj = x[j]
    x[j] = x[i]
    x[i] = old_xj

def shuffle(x):
    r = np.random.randint(0, len(x), len(x))
    _shuffle(x, r)

@jit
def decorate(target, 
              si_0, 
              si_1, 
              values_idx_array, 
              values_si_idx_array, 
              values_array, 
              n_nodes):
  
  for i in range(n_nodes):
    if values_si_idx_array[i] == 0:
      values_array[i] = target[si_0[values_idx_array[i]]] 
    else:
      values_array[i] = target[si_1[values_idx_array[i]]] 

@jit
def turn_to_leaf(nid, start_idx, idx, values_idx_array, values_si_idx_array):
  values_idx_array[nid] = start_idx
  values_si_idx_array[nid] = idx

#bfs loop called in bfs construction.
@jit
def bfs_loop(queue_size, 
            n_nodes, 
            max_features, 
            new_idx_array, 
            idx_array, 
            new_si_idx_array, 
            new_nid_array, 
            left_children, 
            right_children, 
            feature_idx_array, 
            feature_threshold_array, 
            nid_array, 
            imp_min, 
            min_split, 
            feature_idx, 
            si_idx_array, 
            threshold, 
            min_samples_split,
            values_idx_array, 
            values_si_idx_array):
  
  new_queue_size = 0
  for i in range(queue_size):
    if si_idx_array[i] == 1:
      si_idx = 0
      si_idx_ = 1
    else:
      si_idx = 1
      si_idx_ = 0
    
    nid = nid_array[i]
    row = feature_idx[i]
    col = min_split[i]     
    left_imp = imp_min[2 * i]
    right_imp = imp_min[2 * i + 1]

    start_idx = idx_array[2 * i]
    stop_idx = idx_array[2 * i + 1] 
    feature_idx_array[nid] = row
    feature_threshold_array[nid] = threshold[i] 
  
    if left_imp + right_imp == 4.0:
      turn_to_leaf(nid, start_idx, si_idx_, values_idx_array, values_si_idx_array)
    else:
      left_nid = n_nodes
      n_nodes += 1
      right_nid = n_nodes
      n_nodes += 1
      right_children[nid] = right_nid
      left_children[nid] = left_nid

      if left_imp != 0.0:
        n_samples_left = col + 1 - start_idx 
        if n_samples_left < min_samples_split:
          turn_to_leaf(left_nid, 
                        start_idx, 
                        si_idx, 
                        values_idx_array, 
                        values_si_idx_array)
        else:
          new_idx_array[2 * new_queue_size] = start_idx
          new_idx_array[2 * new_queue_size + 1] = col + 1
          new_si_idx_array[new_queue_size] = si_idx
          new_nid_array[new_queue_size] = left_nid
          new_queue_size += 1
      else:
        turn_to_leaf(left_nid, 
                      start_idx, 
                      si_idx, 
                      values_idx_array, 
                      values_si_idx_array)

      if right_imp != 0.0:
        n_samples_right = stop_idx - col - 1
        if n_samples_right < min_samples_split:
          turn_to_leaf(right_nid, 
                        col + 1, 
                        si_idx, 
                        values_idx_array, 
                        values_si_idx_array)
        else:
          new_idx_array[2 * new_queue_size] = col + 1
          new_idx_array[2 * new_queue_size + 1] = stop_idx
          new_si_idx_array[new_queue_size] = si_idx
          new_nid_array[new_queue_size] = right_nid
          new_queue_size += 1
      else:
        turn_to_leaf(right_nid, 
                      col + 1, 
                      si_idx, 
                      values_idx_array, 
                      values_si_idx_array)   
  
  return n_nodes, new_queue_size, new_idx_array, new_si_idx_array, new_nid_array


class RandomClassifierTree(BaseTree): 
  def __init__(self, 
              forest):
    self.n_labels = forest.n_labels
    self.stride = forest.stride
    self.dtype_labels = forest.dtype_labels
    self.dtype_samples = forest.dtype_samples
    self.dtype_indices = forest.dtype_indices
    self.dtype_counts = forest.dtype_counts
    self.n_features = forest.n_features
    self.COMPUTE_THREADS_PER_BLOCK = forest.COMPUTE_THREADS_PER_BLOCK
    self.RESHUFFLE_THREADS_PER_BLOCK = forest.RESHUFFLE_THREADS_PER_BLOCK
    self.samples_gpu = forest.samples_gpu
    self.labels_gpu = forest.labels_gpu
    self.compt_table = forest.compt_table
    self.max_features = forest.max_features
    self.min_samples_split =  forest.min_samples_split
    self.bfs_threshold = forest.bfs_threshold
    self.forest = forest
    self.BFS_THREADS = self.forest.BFS_THREADS
    self.MAX_BLOCK_PER_FEATURE = self.forest.MAX_BLOCK_PER_FEATURE
    self.MAX_BLOCK_BFS = self.forest.MAX_BLOCK_BFS
    if forest.debug == False:
      self.debug = 0
    else:
      self.debug = 1
     
  def __shuffle_feature_indices(self):
    if self.debug == 0:
      shuffle(self.features_array)

  def __compile_kernels(self):
    """ DFS module """
    f = self.forest
    self.find_min_kernel = f.find_min_kernel  
    self.fill_kernel = f.fill_kernel 
    self.scan_reshuffle_tex = f.scan_reshuffle_tex 
    self.comput_total_2d = f.comput_total_2d 
    self.reduce_2d = f.reduce_2d
    self.scan_total_2d = f.scan_total_2d 
    self.scan_reduce = f.scan_reduce 
    
    """ BFS module """
    self.scan_total_bfs = f.scan_total_bfs
    self.comput_bfs_2d = f.comput_bfs_2d
    self.fill_bfs = f.fill_bfs 
    self.reshuffle_bfs = f.reshuffle_bfs 
    self.reduce_bfs_2d = f.reduce_bfs_2d 
    self.get_thresholds = f.get_thresholds 

    """ Other """
    self.predict_kernel = f.predict_kernel 
    self.mark_table = f.mark_table
    const_sorted_indices = f.bfs_module.get_global("sorted_indices_1")[0]
    const_sorted_indices_ = f.bfs_module.get_global("sorted_indices_2")[0]
    cuda.memcpy_htod(const_sorted_indices, np.uint64(self.sorted_indices_gpu.ptr)) 
    cuda.memcpy_htod(const_sorted_indices_, np.uint64(self.sorted_indices_gpu_.ptr)) 

  def __allocate_gpuarrays(self):
    f = self.forest
    self.impurity_left = f.impurity_left 
    self.impurity_right = f.impurity_right 
    self.min_split = f.min_split 
    self.label_total = f.label_total  
    self.label_total_2d = f.label_total_2d
    self.impurity_2d = f.impurity_2d 
    self.min_split_2d = f.min_split_2d 
    self.features_array_gpu = f.features_array_gpu 

  def __release_gpuarrays(self):
    self.impurity_left = None
    self.impurity_right = None
    self.min_split = None
    self.label_total = None
    self.sorted_indices_gpu = None
    self.sorted_indices_gpu_ = None
    self.label_total_2d = None
    self.min_split_2d = None
    self.impurity_2d = None
    self.feature_mask = None
    self.features_array_gpu = None
    
    #Release kernels
    self.fill_kernel = None
    self.scan_reshuffle_tex = None 
    self.scan_total_kernel = None
    self.comput_label_loop_rand_kernel = None
    self.find_min_kernel = None
    self.scan_total_bfs = None
    self.comput_bfs = None
    self.fill_bfs = None
    self.reshuffle_bfs = None
    self.reduce_bfs_2d = None
    self.comput_bfs_2d = None
    #self.predict_kernel = None
    self.get_thresholds = None
    self.scan_reduce = None
    self.mark_table = None

  def __allocate_numpyarrays(self):
    f = self.forest
    self.left_children = np.zeros(self.n_samples * 2, dtype = np.uint32)
    self.right_children = np.zeros(self.n_samples * 2, dtype = np.uint32) 
    self.feature_idx_array = np.zeros(2 * self.n_samples, dtype = np.uint16)
    self.feature_threshold_array = np.zeros(2 * self.n_samples, dtype = np.float32)
    self.idx_array = f.idx_array 
    self.si_idx_array = f.si_idx_array 
    self.nid_array = f.nid_array 
    self.values_idx_array = f.values_idx_array 
    self.values_si_idx_array = f.values_si_idx_array 
    self.threshold_value_idx = f.threshold_value_idx 
    self.min_imp_info = f.min_imp_info  
    self.features_array = f.features_array  

  def __release_numpyarrays(self):
    self.features_array = None
    self.nid_array = None
    self.idx_array = None
    self.si_idx_array = None
    self.threshold_value_idx = None
    self.min_imp_info = None
    self.samples = None
    self.target = None

  def __bfs_construct(self):
    while self.queue_size > 0:
      self.__bfs()
  
  def __bfs(self):
    block_per_split = int(math.ceil(float(self.MAX_BLOCK_BFS) / self.queue_size))
    
    if block_per_split > self.max_features:
      n_blocks = self.max_features
    else:
      n_blocks = block_per_split

    idx_array_gpu = gpuarray.to_gpu(
                    self.idx_array[0 : self.queue_size * 2])
    
    si_idx_array_gpu = gpuarray.to_gpu(
                    self.si_idx_array[0 : self.queue_size])
    
    self.label_total = gpuarray.empty(self.queue_size * self.n_labels, 
                                      dtype = self.dtype_counts)  
    
    threshold_value = gpuarray.empty(self.queue_size, dtype = np.float32)
    
    impurity_gpu = gpuarray.empty(self.queue_size * 2, dtype = np.float32)
    self.min_split = gpuarray.empty(self.queue_size, dtype = self.dtype_indices) 
    min_feature_idx_gpu = gpuarray.empty(self.queue_size, dtype = np.uint16)
    
    impurity_gpu_2d = gpuarray.empty(self.queue_size * 2 * n_blocks, 
                                      dtype = np.float32)
    
    min_split_2d = gpuarray.empty(self.queue_size * n_blocks, 
                                      dtype = self.dtype_indices) 

    min_feature_idx_gpu_2d = gpuarray.empty(self.queue_size * n_blocks, 
                                      dtype = np.uint16)
    
    cuda.memcpy_htod(self.features_array_gpu.ptr, self.features_array) 
    
      
    self.scan_total_bfs.prepared_call(
            (self.queue_size, 1),
            (self.BFS_THREADS, 1, 1),
            self.labels_gpu.ptr,
            self.label_total.ptr,
            si_idx_array_gpu.ptr,
            idx_array_gpu.ptr)
    

    self.comput_bfs_2d.prepared_call(
          (self.queue_size, n_blocks),
          (self.BFS_THREADS, 1, 1),
          self.samples_gpu.ptr,
          self.labels_gpu.ptr,
          idx_array_gpu.ptr,
          si_idx_array_gpu.ptr,
          self.label_total.ptr,
          self.features_array_gpu.ptr,
          impurity_gpu_2d.ptr,
          min_split_2d.ptr,
          min_feature_idx_gpu_2d.ptr)
    
    self.reduce_bfs_2d.prepared_call(
          (self.queue_size, 1),
          (1, 1, 1),
          impurity_gpu_2d.ptr,
          min_split_2d.ptr,
          min_feature_idx_gpu_2d.ptr,
          impurity_gpu.ptr,
          self.min_split.ptr,
          min_feature_idx_gpu.ptr,
          n_blocks)

    self.fill_bfs.prepared_call(
          (self.queue_size, 1),
          (self.BFS_THREADS, 1, 1),
          si_idx_array_gpu.ptr,
          min_feature_idx_gpu.ptr,
          idx_array_gpu.ptr,
          self.min_split.ptr,
          self.mark_table.ptr)


    if block_per_split > self.n_features:
      n_blocks = self.n_features
    else:
      n_blocks = block_per_split
      
    self.reshuffle_bfs.prepared_call(
          (self.queue_size, n_blocks),
          (self.BFS_THREADS, 1, 1),
          si_idx_array_gpu.ptr,
          idx_array_gpu.ptr,
          self.min_split.ptr)
    
    self.__shuffle_feature_indices()
    
    self.get_thresholds.prepared_call(
          (self.queue_size, 1),
          (1, 1, 1),
          si_idx_array_gpu.ptr,
          self.samples_gpu.ptr,
          threshold_value.ptr,
          min_feature_idx_gpu.ptr,
          self.min_split.ptr) 
    
    new_idx_array = np.empty(self.queue_size * 2 * 2, dtype = np.uint32)
    idx_array = self.idx_array
    new_si_idx_array = np.empty(self.queue_size * 2, dtype = np.uint8)
    new_nid_array = np.empty(self.queue_size * 2, dtype = np.uint32)
    left_children = self.left_children
    right_children = self.right_children
    feature_idx_array = self.feature_idx_array
    feature_threshold_array = self.feature_threshold_array
    nid_array = self.nid_array
    
    imp_min = np.empty(self.queue_size * 2, np.float32)
    min_split = np.empty(self.queue_size, self.dtype_indices)
    feature_idx = np.empty(self.queue_size, np.uint16)
    threshold = np.empty(self.queue_size, np.float32) 
    cuda.memcpy_dtoh(imp_min, impurity_gpu.ptr)
    cuda.memcpy_dtoh(min_split, self.min_split.ptr)
    cuda.memcpy_dtoh(feature_idx, min_feature_idx_gpu.ptr)
    cuda.memcpy_dtoh(threshold, threshold_value.ptr) 
    
    si_idx_array = self.si_idx_array 

    self.n_nodes, self.queue_size, self.idx_array, self.si_idx_array, self.nid_array =\
        bfs_loop(self.queue_size, 
                  self.n_nodes, 
                  self.max_features, 
                  new_idx_array, 
                  idx_array, 
                  new_si_idx_array, 
                  new_nid_array, 
                  left_children, 
                  right_children, 
                  feature_idx_array, 
                  feature_threshold_array, 
                  nid_array, 
                  imp_min, 
                  min_split, 
                  feature_idx, 
                  si_idx_array, 
                  threshold, 
                  self.min_samples_split, 
                  self.values_idx_array, 
                  self.values_si_idx_array)

    self.n_nodes = int(self.n_nodes)
    self.queue_size = int(self.queue_size)
 

  def fit(self, samples, target, sorted_indices, n_samples): 
    self.samples_itemsize = self.dtype_samples.itemsize
    self.labels_itemsize = self.dtype_labels.itemsize
    
    self.__allocate_gpuarrays()
    self.sorted_indices_gpu = sorted_indices 
    self.sorted_indices_gpu_ = self.sorted_indices_gpu.copy()
    self.__compile_kernels() 
    self.n_samples = n_samples    

    self.sorted_indices_gpu.idx = 0
    self.sorted_indices_gpu_.idx = 1

    self.samples = samples
    self.target = target
    self.queue_size = 0

    self.__allocate_numpyarrays()
    self.n_nodes = 0 
    
    self.__shuffle_feature_indices()
    self.__dfs_construct(1, 
                        1.0, 
                        0, 
                        self.n_samples, self.sorted_indices_gpu, 
                        self.sorted_indices_gpu_)  
    
    self.__bfs_construct() 
    self.__gpu_decorate_nodes(samples, target)
    self.__release_gpuarrays() 
    self.__release_numpyarrays()

  def __gpu_decorate_nodes(self, samples, labels):
    si_0 = driver.pagelocked_empty(self.n_samples, dtype = self.dtype_indices)
    si_1 = driver.pagelocked_empty(self.n_samples, dtype = self.dtype_indices)
    self.values_array = np.empty(self.n_nodes, dtype = self.dtype_labels)
    cuda.memcpy_dtoh(si_0, self.sorted_indices_gpu.ptr)
    cuda.memcpy_dtoh(si_1, self.sorted_indices_gpu_.ptr)
    
    decorate(self.target, 
              si_0, 
              si_1, 
              self.values_idx_array, 
              self.values_si_idx_array, 
              self.values_array, 
              self.n_nodes)

    self.values_idx_array = None
    self.values_si_idx_array = None
    self.left_children.resize(self.n_nodes, refcheck = False)
    self.right_children.resize(self.n_nodes, refcheck = False) 
    self.feature_threshold_array.resize(self.n_nodes, refcheck = False) 
    self.feature_idx_array.resize(self.n_nodes, refcheck = False)

  def __get_block_size(self, n_samples):
    n_block = int(math.ceil(float(n_samples) / 2000))
    if n_block > self.MAX_BLOCK_PER_FEATURE:
      n_block = self.MAX_BLOCK_PER_FEATURE
    return n_block, int(math.ceil(float(n_samples) / n_block))

  def __gini(self, n_samples, indices_offset, si_gpu_in):
    n_block, n_range = self.__get_block_size(n_samples)
    
    self.scan_total_2d.prepared_call(
          (self.max_features, n_block),
          (self.COMPUTE_THREADS_PER_BLOCK, 1, 1),
          si_gpu_in.ptr + indices_offset,
          self.labels_gpu.ptr,
          self.label_total_2d.ptr,
          self.features_array_gpu.ptr,
          n_range,
          n_samples)

    self.scan_reduce.prepared_call(
          (self.max_features, 1),
          (32, 1, 1),
          self.label_total_2d.ptr,
          n_block)  
    
    self.comput_total_2d.prepared_call(
         (self.max_features, n_block),
         (self.COMPUTE_THREADS_PER_BLOCK, 1, 1),
         si_gpu_in.ptr + indices_offset,
         self.samples_gpu.ptr,
         self.labels_gpu.ptr,
         self.impurity_2d.ptr,
         self.label_total_2d.ptr,
         self.min_split_2d.ptr,
         self.features_array_gpu.ptr,
         n_range,
         n_samples)

    self.reduce_2d.prepared_call(
         (self.max_features, 1),
         (32, 1, 1),
         self.impurity_2d.ptr,
         self.impurity_left.ptr,
         self.impurity_right.ptr,
         self.min_split_2d.ptr,
         self.min_split.ptr,
         n_block)    
    
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
    col = int(self.min_imp_info[2]) 
    row = int(self.min_imp_info[3])
    row = self.features_array[row]  
    return min_left, min_right, row, col

  def  __dfs_construct(self, 
                        depth, 
                        error_rate, 
                        start_idx, 
                        stop_idx, 
                        si_gpu_in, 
                        si_gpu_out):
    def check_terminate():
      if error_rate == 0.0:
        return True
      else:
        return False     

    n_samples = stop_idx - start_idx 
    indices_offset =  start_idx * self.dtype_indices.itemsize    
    nid = self.n_nodes
    self.n_nodes += 1

    if check_terminate():
      turn_to_leaf(nid, 
                    start_idx, 
                    si_gpu_in.idx, 
                    self.values_idx_array, 
                    self.values_si_idx_array
                    )
      return
    
    if n_samples < self.min_samples_split:
      turn_to_leaf(nid, 
                    start_idx, 
                    si_gpu_in.idx, 
                    self.values_idx_array, 
                    self.values_si_idx_array
                    )
      return
    
    if n_samples <= self.bfs_threshold:
      self.idx_array[self.queue_size * 2] = start_idx
      self.idx_array[self.queue_size * 2 + 1] = stop_idx
      self.si_idx_array[self.queue_size] = si_gpu_in.idx
      self.nid_array[self.queue_size] = nid
      self.queue_size += 1
      return
    
    cuda.memcpy_htod(self.features_array_gpu.ptr, self.features_array)
    min_left, min_right, row, col = self.__gini(n_samples, 
                                                indices_offset, 
                                                si_gpu_in) 
    if min_left + min_right == 4:
      turn_to_leaf(nid, 
                  start_idx, 
                  si_gpu_in.idx, 
                  self.values_idx_array, 
                  self.values_si_idx_array) 
      return
    
    cuda.memcpy_dtoh(self.threshold_value_idx, 
                    si_gpu_in.ptr + int(indices_offset) + \
                    int(row * self.stride + col) * \
                    int(self.dtype_indices.itemsize)) 

    self.feature_idx_array[nid] = row
    self.feature_threshold_array[nid] = (float(self.samples[row, \
        self.threshold_value_idx[0]]) + self.samples[row, \
        self.threshold_value_idx[1]]) / 2
    

    self.fill_kernel.prepared_call(
                      (1, 1),
                      (512, 1, 1),
                      si_gpu_in.ptr + row * self.stride * \
                          self.dtype_indices.itemsize + \
                          indices_offset, 
                      n_samples, 
                      col, 
                      self.mark_table.ptr) 


    block = (self.RESHUFFLE_THREADS_PER_BLOCK, 1, 1)
    
    self.scan_reshuffle_tex.prepared_call(
                      (self.n_features, 1),
                      block,
                      si_gpu_in.ptr + indices_offset,
                      si_gpu_out.ptr + indices_offset,
                      n_samples,
                      col)

    self.__shuffle_feature_indices() 

    self.left_children[nid] = self.n_nodes
    self.__dfs_construct(depth + 1, min_left, 
        start_idx, start_idx + col + 1, si_gpu_out, si_gpu_in)
    
    self.right_children[nid] = self.n_nodes
    self.__dfs_construct(depth + 1, min_right, 
        start_idx + col + 1, stop_idx, si_gpu_out, si_gpu_in)
