import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
import numpy as np
import math
from util import total_times, mk_kernel, mk_tex_kernel, timer, dtype_to_ctype, get_best_dtype, start_timer, end_timer
from cuda_random_base_tree import RandomBaseTree
from pycuda import driver
import random
from parakeet import jit
from util import start_timer, end_timer, show_timings

def sync():
  if True:
    driver.Context.synchronize()

@jit
def decorate(target, si_0, si_1, values_idx_array, values_si_idx_array, values_array, n_nodes):
  for i in range(n_nodes):
    if values_si_idx_array[i] == 0:
      values_array[i] = target[si_0[values_idx_array[i]]] 
    else:
      values_array[i] = target[si_1[values_idx_array[i]]] 

@jit
def turn_to_leaf(nid, start_idx, n_samples, idx, values_idx_array, values_si_idx_array):
  values_idx_array[nid] = start_idx
  values_si_idx_array[nid] = idx

@jit
def bfs_loop(queue_size, n_nodes, max_features, new_idx_array, idx_array, new_si_idx_array, new_nid_array, left_children, right_children,
    feature_idx_array, feature_threshold_array, nid_array, imp_min, min_split, feature_idx, si_idx_array, threshold, min_samples_split,
    values_idx_array, values_si_idx_array):
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
      turn_to_leaf(nid, start_idx, stop_idx - start_idx, si_idx_, values_idx_array, values_si_idx_array)
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
          turn_to_leaf(left_nid, start_idx, n_samples_left, si_idx, values_idx_array, values_si_idx_array)
        else:
          new_idx_array[2 * new_queue_size] = start_idx
          new_idx_array[2 * new_queue_size + 1] = col + 1
          new_si_idx_array[new_queue_size] = si_idx
          new_nid_array[new_queue_size] = left_nid
          new_queue_size += 1
      else:
        turn_to_leaf(left_nid, start_idx, 1, si_idx, values_idx_array, values_si_idx_array)

      if right_imp != 0.0:
        n_samples_right = stop_idx - col - 1
        if n_samples_right < min_samples_split:
          turn_to_leaf(right_nid, col + 1, n_samples_right, si_idx, values_idx_array, values_si_idx_array)
        else:
          new_idx_array[2 * new_queue_size] = col + 1
          new_idx_array[2 * new_queue_size + 1] = stop_idx
          new_si_idx_array[new_queue_size] = si_idx
          new_nid_array[new_queue_size] = right_nid
          new_queue_size += 1
      else:
        turn_to_leaf(right_nid, col + 1, 1, si_idx, values_idx_array, values_si_idx_array)   
  return n_nodes , new_queue_size, new_idx_array, new_si_idx_array, new_nid_array


class RandomDecisionTreeSmall(RandomBaseTree): 
  BFS_THREADS = 64
  MAX_BLOCK_PER_FEATURE = 50

  def __init__(self, samples_gpu, labels_gpu, compt_table, dtype_labels, dtype_samples, 
      dtype_indices, dtype_counts, n_features, stride, n_labels, n_threads, n_shf_threads, max_features = None,
      min_samples_split = None, bfs_threshold = 64, debug = False):
    self.root = None
    self.n_labels = n_labels
    self.stride = stride
    self.dtype_labels = dtype_labels
    self.dtype_samples = dtype_samples
    self.dtype_indices = dtype_indices
    self.dtype_counts = dtype_counts
    self.n_features = n_features
    self.COMPT_THREADS_PER_BLOCK = n_threads
    self.RESHUFFLE_THREADS_PER_BLOCK = n_shf_threads
    self.samples_gpu = samples_gpu
    self.labels_gpu = labels_gpu
    self.compt_table = compt_table
    self.max_features = max_features
    self.min_samples_split =  min_samples_split
    self.bfs_threshold = bfs_threshold
    if debug == False:
      self.debug = 0
    else:
      self.debug = 1
     
  def get_indices(self):
    if self.debug:
      return np.arange(self.max_features, dtype = self.dtype_indices)

    return np.array(random.sample(xrange(self.n_features), self.max_features), dtype=self.dtype_indices)
  
  def __shuffle_features(self):
    if self.debug == False:
      np.random.shuffle(self.features_array)

  def __compile_kernels(self):
    ctype_indices = dtype_to_ctype(self.dtype_indices)
    ctype_labels = dtype_to_ctype(self.dtype_labels)
    ctype_counts = dtype_to_ctype(self.dtype_counts)
    ctype_samples = dtype_to_ctype(self.dtype_samples)
    n_labels = self.n_labels
    n_threads = self.COMPT_THREADS_PER_BLOCK
    n_shf_threads = self.RESHUFFLE_THREADS_PER_BLOCK

    self.fill_kernel = mk_kernel(
      params = (ctype_indices,), 
      func_name = "fill_table", 
      kernel_file = "fill_table_si.cu", 
      prepare_args = "PiiPi")

      
    self.scan_total_kernel = mk_kernel(
        params = (n_threads, n_labels, ctype_labels, ctype_counts, ctype_indices), 
        func_name = "count_total", 
        kernel_file = "scan_kernel_total_si.cu", 
        prepare_args = "PPPi") 
    
    self.comput_total_kernel = mk_kernel(
      params = (n_threads, n_labels, ctype_samples, 
                ctype_labels, ctype_counts, ctype_indices), 
      func_name = "compute", 
      kernel_file = "comput_kernel_total_rand.cu", 
      prepare_args = "PPPPPPPPii")
         
    self.scan_reshuffle_tex, tex_ref = mk_tex_kernel(
      params = (ctype_indices, n_shf_threads), 
      func_name = "scan_reshuffle", 
      tex_name = "tex_mark", 
      kernel_file = "pos_scan_reshuffle_si_c_tex.cu", 
      prepare_args = "PPPiii")   
    self.mark_table.bind_to_texref_ext(tex_ref) 
    
    
    self.find_min_kernel = mk_kernel(
      params = (ctype_counts, 32), 
      func_name = "find_min_imp", 
      kernel_file = "find_min_gini.cu", 
      prepare_args = "PPPi")
    
    self.predict_kernel = mk_kernel(
        params = (ctype_indices, ctype_samples, ctype_labels), 
        func_name = "predict", 
        kernel_file = "predict.cu", 
        prepare_args = "PPPPPPPii")
  
    self.scan_total_bfs = mk_kernel(
      params = (self.BFS_THREADS, n_labels, ctype_labels, ctype_counts, ctype_indices), 
      func_name = "count_total", 
      kernel_file = "scan_kernel_total_bfs.cu", 
      prepare_args = "PPPPPP")
  
    self.comput_bfs = mk_kernel(
      params = (self.BFS_THREADS, n_labels, ctype_samples, 
                ctype_labels, ctype_counts, ctype_indices, self.debug), 
      func_name = "compute", 
      kernel_file = "comput_kernel_bfs.cu", 
      prepare_args = "PPPPPPPPPPPiii")
    
    self.fill_bfs = mk_kernel(
      params = (ctype_indices,), 
      func_name = "fill_table", 
      kernel_file = "fill_table_bfs.cu", 
      prepare_args = "PPPPPPPi")
    
    self.reshuffle_bfs, tex_ref = mk_tex_kernel(
      params = (ctype_indices, self.BFS_THREADS), 
      func_name = "scan_reshuffle", 
      tex_name= "tex_mark", 
      kernel_file = "pos_scan_reshuffle_bfs.cu", 
      prepare_args = "PPPPPPii")
    self.mark_table.bind_to_texref_ext(tex_ref) 
    
    self.comput_total_2d = mk_kernel(
      params = (n_threads, n_labels, ctype_samples, ctype_labels, ctype_counts, 
                ctype_indices, self.MAX_BLOCK_PER_FEATURE, self.debug), 
      func_name = "compute", 
      kernel_file = "comput_kernel_2d.cu", 
      prepare_args = "PPPPPPPiii")

    self.reduce_2d = mk_kernel(
      params = (ctype_indices, self.MAX_BLOCK_PER_FEATURE, self.debug), 
      func_name = "reduce", 
      kernel_file = "reduce_2d.cu", 
      prepare_args = "PPPPPi")
    
    self.scan_total_2d = mk_kernel(
      params = (n_threads, n_labels, ctype_labels, ctype_counts, 
                ctype_indices, self.MAX_BLOCK_PER_FEATURE, self.debug),
      func_name = "count_total", 
      kernel_file = "scan_kernel_2d.cu", 
      prepare_args = "PPPPiii")
    
    self.scan_reduce = mk_kernel(
      params = (n_labels, ctype_indices, self.MAX_BLOCK_PER_FEATURE), 
      func_name = "scan_reduce", 
      kernel_file = "scan_reduce.cu", 
      prepare_args = "Pi")
    
    self.get_thresholds = mk_kernel(
      params = (ctype_indices, ctype_samples), 
      func_name = "get_thresholds", 
      kernel_file = "get_thresholds.cu", 
      prepare_args = "PPPPPPPi")
    
    self.feature_selector = mk_kernel(
      params = (ctype_indices, ctype_samples), 
      func_name = "feature_selector", 
      kernel_file = "feature_selector.cu", 
      prepare_args = "PPPii")

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
    self.label_total_2d = gpuarray.zeros(self.max_features * (self.MAX_BLOCK_PER_FEATURE + 1) * self.n_labels, self.dtype_indices)
    self.impurity_2d = gpuarray.empty(self.max_features * self.MAX_BLOCK_PER_FEATURE * 2, np.float32)
    self.min_split_2d = gpuarray.empty(self.max_features * self.MAX_BLOCK_PER_FEATURE, self.dtype_counts)
    self.feature_mask = gpuarray.empty(self.n_features, np.uint8)

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
    self.label_total_2d = None
    self.min_split_2d = None
    self.impurity_2d = None
    self.feature_mask = None
  
  def __allocate_numpyarrays(self):
    self.left_children = np.zeros(self.n_samples * 2, dtype = np.uint32)
    self.right_children = np.zeros(self.n_samples * 2, dtype = np.uint32) 
    self.feature_idx_array = np.zeros(2 * self.n_samples, dtype = np.uint16)
    self.feature_threshold_array = np.zeros(2 * self.n_samples, dtype = np.float32)
    self.idx_array = np.zeros(2 * self.n_samples, dtype = np.uint32)
    self.si_idx_array = np.zeros(self.n_samples, dtype = np.uint8)
    self.nid_array = np.zeros(self.n_samples, dtype = np.uint32)
    self.values_idx_array = np.zeros(2 * self.n_samples, dtype = self.dtype_indices)
    self.values_si_idx_array = np.zeros(2 * self.n_samples, dtype = np.uint8)
    self.features_array = np.arange(self.n_features, dtype = self.dtype_indices)
    self.threshold_value_idx = np.zeros(2, self.dtype_indices)
    self.min_imp_info = np.zeros(4, dtype = np.float32)  
    

  def __release_numpyarrays(self):
    self.feature_array = None
    self.nid_array = None
    self.idx_array = None
    self.si_idx_array = None
    self.threshold_value_idx = None
    self.min_imp_info = None


  def __bfs_construct(self):
    while self.queue_size > 0:
      self.__bfs()
  
  def __bfs(self):
    idx_array_gpu = gpuarray.to_gpu(self.idx_array[0 : self.queue_size * 2])
    si_idx_array_gpu = gpuarray.to_gpu(self.si_idx_array[0 : self.queue_size])
    subset_indices_array_gpu = gpuarray.empty(self.n_features, dtype = self.dtype_indices)
    
    self.label_total = gpuarray.empty(self.queue_size * self.n_labels, dtype = self.dtype_counts)  
    impurity_gpu = gpuarray.empty(self.queue_size * 2 * self.max_features, dtype = np.float32)
    self.min_split = gpuarray.empty(self.queue_size * self.max_features, dtype = self.dtype_indices) 
    threshold_value = gpuarray.empty(self.queue_size * self.max_features, dtype = np.float32)
    min_feature_idx_gpu = gpuarray.empty(self.queue_size * self.max_features, dtype = np.uint16)

    cuda.memcpy_htod(subset_indices_array_gpu.ptr, self.features_array) 
    
    start_timer("gini bfs scan")
    self.scan_total_bfs.prepared_call(
            (self.queue_size, 1),
            (self.BFS_THREADS, 1, 1),
            self.sorted_indices_gpu.ptr,
            self.sorted_indices_gpu_.ptr,
            self.labels_gpu.ptr,
            self.label_total.ptr,
            si_idx_array_gpu.ptr,
            idx_array_gpu.ptr)
    
    sync()
    end_timer("gini bfs scan")

    start_timer("gini bfs comput")
    self.comput_bfs.prepared_call(
          (self.queue_size, 1),
          (self.BFS_THREADS, 1, 1),
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
          self.n_features,
          self.stride)
    
    self.fill_bfs.prepared_call(
          (self.queue_size, 1),
          (self.BFS_THREADS, 1, 1),
          self.sorted_indices_gpu.ptr,
          self.sorted_indices_gpu_.ptr,
          si_idx_array_gpu.ptr,
          min_feature_idx_gpu.ptr,
          idx_array_gpu.ptr,
          self.min_split.ptr,
          self.mark_table.ptr,
          self.stride)

    sync()
    end_timer("gini bfs comput")

    block_per_split = int(math.ceil(float(2000) / self.queue_size))
    if block_per_split > self.n_features:
      block_per_split = self.n_features
    
    
    start_timer("bfs reshuffle")
    self.reshuffle_bfs.prepared_call(
          (self.queue_size, block_per_split),
          (self.BFS_THREADS, 1, 1),
          self.mark_table.ptr,
          si_idx_array_gpu.ptr,
          self.sorted_indices_gpu.ptr,
          self.sorted_indices_gpu_.ptr,
          idx_array_gpu.ptr,
          self.min_split.ptr,
          self.n_features,
          self.stride) 
    
    sync()
    end_timer("bfs reshuffle")
    self.__shuffle_features()

    self.get_thresholds.prepared_call(
          (self.queue_size, 1),
          (1, 1, 1),
          si_idx_array_gpu.ptr,
          self.sorted_indices_gpu.ptr,
          self.sorted_indices_gpu_.ptr,
          self.samples_gpu.ptr,
          threshold_value.ptr,
          min_feature_idx_gpu.ptr,
          self.min_split.ptr,
          self.stride)
    
    new_idx_array = np.empty(self.queue_size * 2 * 2, dtype = np.uint32)
    idx_array = self.idx_array
    new_si_idx_array = np.empty(self.queue_size * 2, dtype = np.uint8)
    new_nid_array = np.empty(self.queue_size * 2, dtype = np.uint32)
    left_children = self.left_children
    right_children = self.right_children
    feature_idx_array = self.feature_idx_array
    feature_threshold_array = self.feature_threshold_array
    nid_array = self.nid_array
    imp_min = impurity_gpu.get()
    min_split = self.min_split.get()
    feature_idx = min_feature_idx_gpu.get()
    si_idx_array = self.si_idx_array 
    threshold = threshold_value.get()
    
    self.n_nodes, self.queue_size, self.idx_array, self.si_idx_array, self.nid_array = bfs_loop(self.queue_size, self.n_nodes, 
        self.max_features, new_idx_array, idx_array, new_si_idx_array, new_nid_array, left_children, right_children,
        feature_idx_array, feature_threshold_array, nid_array, imp_min, min_split, feature_idx, si_idx_array, threshold,
        self.min_samples_split, self.values_idx_array, self.values_si_idx_array)
    

    self.n_nodes = int(self.n_nodes)
    self.queue_size = int(self.queue_size)
 

  def fit(self, samples, target, sorted_indices, n_samples): 
    self.samples_itemsize = self.dtype_samples.itemsize
    self.labels_itemsize = self.dtype_labels.itemsize
    
    self.__allocate_gpuarrays()
    self.__compile_kernels() 
    
    self.sorted_indices_gpu = sorted_indices 
    self.sorted_indices_gpu_ = self.sorted_indices_gpu.copy()
    self.n_samples = n_samples    

    self.sorted_indices_gpu.idx = 0
    self.sorted_indices_gpu_.idx = 1

    assert self.sorted_indices_gpu.strides[0] == target.size * self.sorted_indices_gpu.dtype.itemsize 
    assert self.samples_gpu.strides[0] == target.size * self.samples_gpu.dtype.itemsize   
    
    self.samples = samples
    self.target = target
    self.queue_size = 0

    self.__allocate_numpyarrays()
    self.n_nodes = 0 

    self.root = self.__dfs_construct(1, 1.0, 0, self.n_samples, self.sorted_indices_gpu, self.sorted_indices_gpu_)  
    self.__bfs_construct() 
    self.__gpu_decorate_nodes(samples, target)
    self.__release_gpuarrays() 
    self.__release_numpyarrays()
    show_timings()
    print "n_nodes : ", self.n_nodes

  def __gpu_decorate_nodes(self, samples, labels):
    si_0 = np.empty(self.n_samples, dtype = self.dtype_indices)
    si_1 = np.empty(self.n_samples, dtype = self.dtype_indices)
    self.values_array = np.empty(self.n_nodes, dtype = self.dtype_labels)
    cuda.memcpy_dtoh(si_0, self.sorted_indices_gpu.ptr)
    cuda.memcpy_dtoh(si_1, self.sorted_indices_gpu_.ptr)
    
    decorate(self.target, si_0, si_1, self.values_idx_array, self.values_si_idx_array, self.values_array, self.n_nodes)

    self.values_idx_array = None
    self.values_si_idx_array = None
    """
    self.left_children = None
    self.right_children = None
    self.feature_threshold_array = None
    self.feature_idx_array = None
    """
    self.left_children = self.left_children[0 : self.n_nodes]
    self.right_children = self.right_children[0 : self.n_nodes]
    self.feature_threshold_array = self.feature_threshold_array[0 : self.n_nodes]
    self.feature_idx_array = self.feature_idx_array[0 : self.n_nodes]

  def turn_to_leaf(self, nid, start_idx, n_samples, idx):
    """ Pick the indices to record on the leaf node. We'll choose the most common label """ 
    self.values_idx_array[nid] = start_idx
    self.values_si_idx_array[nid] = idx

  def __gini_small(self, n_samples, indices_offset, subset_indices, si_gpu_in):
    block = (self.COMPT_THREADS_PER_BLOCK, 1, 1)
    grid = (self.max_features, 1) 
    
    start_timer("gini small")
    self.scan_total_kernel.prepared_call(
                (1, 1),
                block,
                si_gpu_in.ptr + indices_offset,
                self.labels_gpu.ptr,
                self.label_total.ptr,
                n_samples)
    
    self.comput_total_kernel.prepared_call(
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
    
    self.find_min_kernel.prepared_call(
                (1, 1),
                (32, 1, 1),
                self.impurity_left.ptr,
                self.impurity_right.ptr,
                self.min_split.ptr,
                self.max_features)
    
    sync()
    end_timer("gini small")

    cuda.memcpy_dtoh(self.min_imp_info, self.impurity_left.ptr)
    min_right = self.min_imp_info[1] 
    min_left = self.min_imp_info[0] 
    col = int(self.min_imp_info[2]) 
    row = int(self.min_imp_info[3])
    row = subset_indices[row] 
    return min_left, min_right, row, col


  def __get_block_size(self, n_samples):
    n_block = int(math.ceil(float(n_samples) / 2000))
    if n_block > self.MAX_BLOCK_PER_FEATURE:
      n_block = self.MAX_BLOCK_PER_FEATURE
    return n_block, int(math.ceil(float(n_samples) / n_block))


  def __gini_large(self, n_samples, indices_offset, subset_indices, si_gpu_in):
    n_block, n_range = self.__get_block_size(n_samples)
    
    start_timer("gini dfs scan")
    self.scan_total_2d.prepared_call(
          (self.max_features, n_block),
          (self.COMPT_THREADS_PER_BLOCK, 1, 1),
          si_gpu_in.ptr + indices_offset,
          self.labels_gpu.ptr,
          self.label_total_2d.ptr,
          self.subset_indices.ptr,
          n_range,
          n_samples,
          self.stride)
     
    self.scan_reduce.prepared_call(
          (self.max_features, 1),
          (32, 1, 1),
          self.label_total_2d.ptr,
          n_block) 
    
    sync()
    end_timer("gini dfs scan")
    
    start_timer("gini dfs comput")
    self.comput_total_2d.prepared_call(
         (self.max_features, n_block),
         (self.COMPT_THREADS_PER_BLOCK, 1, 1),
         si_gpu_in.ptr + indices_offset,
         self.samples_gpu.ptr,
         self.labels_gpu.ptr,
         self.impurity_2d.ptr,
         self.label_total_2d.ptr,
         self.min_split_2d.ptr,
         self.subset_indices.ptr,
         n_range,
         n_samples,
         self.stride)
    
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
    
    sync()
    end_timer("gini dfs comput")
    
    cuda.memcpy_dtoh(self.min_imp_info, self.impurity_left.ptr)
    min_right = self.min_imp_info[1] 
    min_left = self.min_imp_info[0] 
    col = int(self.min_imp_info[2]) 
    row = int(self.min_imp_info[3])
    row = subset_indices[row]  
    return min_left, min_right, row, col


  def  __dfs_construct(self, depth, error_rate, start_idx, stop_idx, si_gpu_in, si_gpu_out):
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
      turn_to_leaf(nid, start_idx, 1, si_gpu_in.idx, self.values_idx_array, self.values_si_idx_array)
      return
    
    if n_samples < self.min_samples_split:
      turn_to_leaf(nid, start_idx, n_samples, si_gpu_in.idx, self.values_idx_array, self.values_si_idx_array)
      return
    
    if n_samples <= self.bfs_threshold:
      self.idx_array[self.queue_size * 2] = start_idx
      self.idx_array[self.queue_size * 2 + 1] = stop_idx
      self.si_idx_array[self.queue_size] = si_gpu_in.idx
      self.nid_array[self.queue_size] = nid
      self.queue_size += 1
      return

    if self.debug: 
      subset_indices = self.get_indices()
    else:
      self.feature_selector.prepared_call(
                (self.n_features, 1),
                (1, 1, 1),
                si_gpu_in.ptr + indices_offset,
                self.samples_gpu.ptr,
                self.feature_mask.ptr,
                n_samples,
                self.stride)
      
      selected_features = np.where(self.feature_mask.get())[0] 
      feature_num = selected_features.size
     
      if feature_num == self.n_features:
        subset_indices = self.get_indices()
      else:
        subset_indices = np.zeros(self.max_features, self.dtype_indices)
        if feature_num < self.max_features:
          max_features = feature_num
        else:
          max_features = self.max_features

        indices = np.array(random.sample(xrange(feature_num), max_features))
        subset_indices[0 : max_features] = selected_features[indices].astype(self.dtype_indices)

    cuda.memcpy_htod(self.subset_indices.ptr, subset_indices)
    
    if n_samples > 2000:
      min_left, min_right, row, col = self.__gini_large(n_samples, indices_offset, subset_indices, si_gpu_in) 
    else:
      min_left, min_right, row, col = self.__gini_small(n_samples, indices_offset, subset_indices, si_gpu_in) 

    if min_left + min_right == 4:
      turn_to_leaf(nid, start_idx, n_samples, si_gpu_in.idx, self.values_idx_array, self.values_si_idx_array) 
      return
  
    cuda.memcpy_dtoh(self.threshold_value_idx, si_gpu_in.ptr + int(indices_offset) + 
        int(row * self.stride + col) * int(self.dtype_indices.itemsize)) 
    self.feature_idx_array[nid] = row
    self.feature_threshold_array[nid] = (float(self.samples[row, self.threshold_value_idx[0]]) + self.samples[row, self.threshold_value_idx[1]]) / 2
    

    start_timer("dfs reshuffle")
    self.fill_kernel.prepared_call(
                      (1, 1),
                      (512, 1, 1),
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

    sync()
    end_timer("dfs reshuffle")

    self.left_children[nid] = self.n_nodes
    self.__dfs_construct(depth + 1, min_left, 
        start_idx, start_idx + col + 1, si_gpu_out, si_gpu_in)
    
    self.right_children[nid] = self.n_nodes
    self.__dfs_construct(depth + 1, min_right, 
        start_idx + col + 1, stop_idx, si_gpu_out, si_gpu_in)
