import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
import numpy as np
import math
from util import total_times, mk_kernel, mk_tex_kernel, timer, dtype_to_ctype, get_best_dtype, start_timer, end_timer
from cuda_random_base_tree import RandomBaseTree
cimport numpy as np

ctypedef np.uint8_t DTYPE_8
ctypedef np.uint32_t DTYPE_32
ctypedef np.float32_t FLOAT_32

class RandomDecisionTreeSmall(RandomBaseTree): 
  BFS_THREADS = 64

  def __init__(self, samples_gpu, labels_gpu, compt_table, dtype_labels, dtype_samples, 
      dtype_indices, dtype_counts, n_features, stride, n_labels, n_threads, n_shf_threads, max_features = None,
      min_samples_split = None, bfs_threshold = 64):
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
    
    self.find_min_kernel = mk_kernel((ctype_counts, 32), 
        "find_min_imp", "find_min_gini.cu")
      
    self.predict_kernel = mk_kernel((ctype_indices, ctype_samples, ctype_labels), 
        "predict", "predict.cu")
  
    self.scan_total_bfs = mk_kernel((self.BFS_THREADS, n_labels, ctype_labels, ctype_counts, ctype_indices), 
        "count_total", "scan_kernel_total_bfs.cu")
  
    self.comput_bfs = mk_kernel((self.BFS_THREADS, n_labels, ctype_samples, ctype_labels, ctype_counts, 
      ctype_indices), "compute", "comput_kernel_bfs.cu")
    
    self.fill_bfs = mk_kernel((ctype_indices,), "fill_table", "fill_table_bfs.cu")
    
    self.reshuffle_bfs, tex_ref = mk_tex_kernel((ctype_indices, self.BFS_THREADS), 
        "scan_reshuffle", "tex_mark", "pos_scan_reshuffle_bfs.cu")
    self.mark_table.bind_to_texref_ext(tex_ref) 
    
    if hasattr(self.fill_kernel, "is_prepared"):
      return
    
    self.fill_kernel.is_prepared = True
    self.fill_kernel.prepare("PiiPi")
    self.scan_reshuffle_tex.prepare("PPPiii") 
    self.scan_total_kernel.prepare("PPPi")
    self.comput_label_loop_rand_kernel.prepare("PPPPPPPPii")
    self.find_min_kernel.prepare("PPPi")
    self.predict_kernel.prepare("PPPPPPPii")
    self.scan_total_bfs.prepare("PPPPPPPi")
    self.comput_bfs.prepare("PPPPPPPPPPPii")
    self.fill_bfs.prepare("PPPPPPPi")
    self.reshuffle_bfs.prepare("PPPPPPii")
    self.target_value_idx = np.zeros(1, self.dtype_indices)

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
    while self.queue_size > 0:
      self.__bfs()

  def __bfs(self):
    idx_array_gpu = gpuarray.to_gpu(self.idx_array[0 : self.queue_size * 2])
    si_idx_array_gpu = gpuarray.to_gpu(self.si_idx_array[0 : self.queue_size])
    subset_indices_array_gpu = gpuarray.empty(self.queue_size * self.max_features, dtype = self.dtype_indices)
    min_feature_idx_gpu = gpuarray.empty(self.queue_size, dtype = np.uint16)
    
    self.label_total = gpuarray.empty(self.queue_size * self.n_labels, dtype = self.dtype_counts)  
    impurity_gpu = gpuarray.empty(self.queue_size * 2, dtype = np.float32)
    self.min_split = gpuarray.empty(self.queue_size, dtype = self.dtype_indices)
    
    cuda.memcpy_htod(subset_indices_array_gpu.ptr, self.subset_indices_array[0:self.max_features * self.queue_size]) 

    self.scan_total_bfs.prepared_call(
            (self.queue_size, 1),
            (self.BFS_THREADS, 1, 1),
            self.sorted_indices_gpu.ptr,
            self.sorted_indices_gpu_.ptr,
            self.labels_gpu.ptr,
            self.label_total.ptr,
            si_idx_array_gpu.ptr,
            idx_array_gpu.ptr,
            subset_indices_array_gpu.ptr,
            self.max_features)
    
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
          self.stride)
    
    min_split = self.min_split.get()
    cdef np.ndarray[FLOAT_32, ndim=1] imp_min = impurity_gpu.get()

    feature_idx = min_feature_idx_gpu.get()
    
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
    
    block_per_split = int(math.ceil(float(5000) / self.queue_size))
    if block_per_split > self.n_features:
      block_per_split = self.n_features

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
    
    cdef int left_idx
    cdef int right_idx
    cdef float left_imp
    cdef float right_imp
    cdef int col
    cdef int start_idx
    cdef int stop_idx
    cdef int n_samples_left
    cdef int n_samples_right
    cdef int i
    cdef int queue_size
    cdef int new_queue_size
    cdef int nid
    cdef int row
    cdef int left_nid
    cdef int right_nid

    new_queue_size = 0

    """ While the GPU is being utilized, run some CPU intensive code on CPU"""
    for i in xrange(self.queue_size):
      left_idx = 2 * i
      right_idx = left_idx + 1
      left_imp = imp_min[left_idx]
      right_imp = imp_min[right_idx]
      col = min_split[i]
      start_idx = self.idx_array[left_idx]
      stop_idx = self.idx_array[right_idx]
      n_samples_left = col + 1 - start_idx 
      n_samples_right = stop_idx - col - 1
      
      if left_imp + right_imp == 4.0:
        continue
      
      if left_imp != 0.0 and n_samples_left >= self.min_samples_split:
        self.subset_indices_array[new_queue_size * self.max_features : 
            (new_queue_size + 1) * self.max_features] = self.get_indices()
        new_queue_size += 1
      
      if right_imp != 0.0 and n_samples_right >= self.min_samples_split:
        self.subset_indices_array[new_queue_size * self.max_features : 
            (new_queue_size + 1) * self.max_features] = self.get_indices()
        new_queue_size += 1
    
    queue_size = 0 
    new_idx_array = np.empty(new_queue_size * 2, dtype = self.dtype_indices)
    
    cdef np.ndarray[DTYPE_8, ndim=1] new_si_idx_array
    cdef np.ndarray[DTYPE_32, ndim=1] new_nid_array
    cdef np.ndarray[DTYPE_32, ndim=1] left_children
    cdef np.ndarray[DTYPE_32, ndim=1] right_children

    left_children = self.left_children
    right_children = self.right_children
    new_si_idx_array = np.empty(new_queue_size, dtype = np.uint8)
    new_nid_array = np.empty(new_queue_size, dtype = np.uint32)
    
    """ Put the new request in queue"""
    for i in xrange(self.queue_size):
      if self.si_idx_array[i] == 1:
        si = self.sorted_indices_gpu
        si_ = self.sorted_indices_gpu_ 
      else:
        si = self.sorted_indices_gpu_
        si_ = self.sorted_indices_gpu
      
      nid = self.nid_array[i]
      row = feature_idx[i]
      col = min_split[i]     
      left_imp = imp_min[2 * i]
      right_imp = imp_min[2 * i + 1]
      start_idx = self.idx_array[2 * i]
      stop_idx = self.idx_array[2 * i + 1]
       
      cuda.memcpy_dtoh(self.threshold_value_idx, si_.ptr +  
          int(row * self.stride + col) * int(self.dtype_indices.itemsize))
      
      self.feature_idx_array[nid] = row
      self.feature_threshold_array[nid] = (float(self.samples[row, self.threshold_value_idx[0]]) + self.samples[row, self.threshold_value_idx[1]]) / 2
    
      if left_imp + right_imp == 4.0:
        self.__turn_to_leaf(nid, start_idx, stop_idx - start_idx, si_)
        continue
      
      left_nid = self.n_nodes
      self.n_nodes += 1
      right_nid = self.n_nodes
      self.n_nodes += 1
      
      right_children[nid] = right_nid
      left_children[nid] = left_nid

      if left_imp != 0.0:
        n_samples_left = col + 1 - start_idx 
        if n_samples_left < self.min_samples_split:
          self.__turn_to_leaf(left_nid, start_idx, n_samples_left, si)
        else:
          new_idx_array[2 * queue_size] = start_idx
          new_idx_array[2 * queue_size + 1] = col + 1
          new_si_idx_array[queue_size] = si.idx
          new_nid_array[queue_size] = left_nid
          queue_size += 1
      else:
        self.__turn_to_leaf(left_nid, start_idx, 1, si)

      if right_imp != 0.0:
        n_samples_right = stop_idx - col - 1
        if n_samples_right < self.min_samples_split:
          self.__turn_to_leaf(right_nid, col + 1, n_samples_right, si)
        else:
          new_idx_array[2 * queue_size] = col + 1
          new_idx_array[2 * queue_size + 1] = stop_idx
          new_si_idx_array[queue_size] = si.idx
          new_nid_array[queue_size] = right_nid
          queue_size += 1
      else:
        self.__turn_to_leaf(right_nid, col + 1, 1, si)
    
    self.idx_array = new_idx_array
    self.si_idx_array = new_si_idx_array
    self.nid_array = new_nid_array
    self.queue_size = queue_size


  def fit(self, samples, target, sorted_indices, n_samples): 
    self.samples_itemsize = self.dtype_samples.itemsize
    self.labels_itemsize = self.dtype_labels.itemsize
    self.threshold_value_idx = np.zeros(2, self.dtype_indices)
    self.min_imp_info = np.zeros(4, dtype = np.float32)  
    
    if self.max_features is None:
      self.max_features = int(math.ceil(math.log(self.n_features, 2)))

    assert self.max_features > 0 and self.max_features <= self.n_features, "max_features must be between 0 and n_features." 
    self.__allocate_gpuarrays()
    self.__compile_kernels() 
    self.sorted_indices_gpu = sorted_indices #gpuarray.to_gpu(self.sorted_indices)
    self.sorted_indices_gpu_ = self.sorted_indices_gpu.copy()
    self.n_samples = n_samples    

    self.sorted_indices_gpu.idx = 0
    self.sorted_indices_gpu_.idx = 1

    assert self.sorted_indices_gpu.strides[0] == target.size * self.sorted_indices_gpu.dtype.itemsize 
    assert self.samples_gpu.strides[0] == target.size * self.samples_gpu.dtype.itemsize   
    
    self.samples = samples
    self.target = target
    self.left_children = np.zeros(self.n_samples * 2, dtype = np.uint32)
    self.right_children = np.zeros(self.n_samples * 2, dtype = np.uint32)
    
    self.feature_idx_array = np.zeros(2 * self.n_samples, dtype = np.uint16)
    self.feature_threshold_array = np.zeros(2 * self.n_samples, dtype = np.float32)
    self.idx_array = np.zeros(2 * self.n_samples, dtype = self.dtype_indices)
    self.si_idx_array = np.zeros(self.n_samples, dtype = np.uint8)
    self.subset_indices_array = np.zeros(self.n_samples * self.max_features, dtype = self.dtype_indices)
    self.queue_size = 0
    self.nid_array = np.zeros(self.n_samples, dtype = np.uint32)
    self.values_idx_array = np.zeros(2 * self.n_samples, dtype = self.dtype_indices)
    self.values_si_idx_array = np.zeros(2 * self.n_samples, dtype = np.uint8)

    self.n_nodes = 0 
    self.root = self.__dfs_construct(1, 1.0, 0, self.n_samples, self.sorted_indices_gpu, self.sorted_indices_gpu_, self.get_indices())  
    self.__bfs_construct() 
    self.__gpu_decorate_nodes(samples, target)
    self.__release_gpuarrays()


  def __gpu_decorate_nodes(self, samples, labels):
    si_0 = np.empty(self.n_samples, dtype = self.dtype_indices)
    si_1 = np.empty(self.n_samples, dtype = self.dtype_indices)
    sis = (si_0, si_1)
    
    self.values_array = np.empty(self.n_nodes, dtype = self.dtype_labels)
    cuda.memcpy_dtoh(si_0, self.sorted_indices_gpu.ptr)
    cuda.memcpy_dtoh(si_1, self.sorted_indices_gpu_.ptr)
    
    cdef int i
    for i in range(self.n_nodes):
      self.values_array[i] = self.target[sis[self.values_si_idx_array[i]][self.values_idx_array[i]]] 
    
    self.values_idx_array = None
    self.values_si_idx_array = None
    self.left_children = self.left_children[0 : self.n_nodes]
    self.right_children = self.right_children[0 : self.n_nodes]
    self.feature_threshold_array = self.feature_threshold_array[0 : self.n_nodes]
    self.feature_idx_array = self.feature_idx_array[0 : self.n_nodes]


  def __turn_to_leaf(self, nid, start_idx, n_samples, si):
    """ Pick the indices to record on the leaf node. We'll choose the most common label """
    self.values_idx_array[nid] = start_idx
    self.values_si_idx_array[nid] = si.idx
     # if n_samples < 5:
     #   start_timer("memcpy")
     #   cuda.memcpy_dtoh(self.target_value_idx, si.ptr + int(start_idx * self.dtype_indices.itemsize))
     #   #self.target_value_idx[0] = 0
     #   end_timer("memcpy")
     #   self.values_array[nid] = self.target[self.target_value_idx[0]]
     # else:
     #   si_labels = np.empty(n_samples, dtype=self.dtype_indices)
     #   start_timer("memcpy")
     #   cuda.memcpy_dtoh(si_labels, si.ptr + int(start_idx * self.dtype_indices.itemsize))
     #   end_timer("memcpy")
     #   self.values_array[nid]  = self._find_most_common_label(self.target[si_labels])
  

  def  __dfs_construct(self, depth, error_rate, start_idx, stop_idx, si_gpu_in, si_gpu_out, subset_indices):
    def check_terminate():
      if error_rate == 0.0:
        return True
      else:
        return False 
    
    cdef int n_samples
    cdef int nid
    cdef int col
    cdef int row
    cdef float min_left
    cdef float min_right

    n_samples = stop_idx - start_idx 
    indices_offset =  start_idx * self.dtype_indices.itemsize    
    
    nid = self.n_nodes
    self.n_nodes += 1

    if check_terminate():
      self.__turn_to_leaf(nid, start_idx, 1, si_gpu_in)
      return
    
    if n_samples < self.min_samples_split:
      self.__turn_to_leaf(nid, start_idx, n_samples, si_gpu_in)
      return
    
    if n_samples <= self.bfs_threshold:
      self.idx_array[self.queue_size * 2] = start_idx
      self.idx_array[self.queue_size * 2 + 1] = stop_idx
      self.si_idx_array[self.queue_size] = si_gpu_in.idx
      self.subset_indices_array[self.queue_size * self.max_features : 
          (self.queue_size + 1) * self.max_features] = subset_indices
      self.nid_array[self.queue_size] = nid
      self.queue_size += 1
      return
    
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
    row = subset_indices[row]
    
    if min_left + min_right == 0:
      return

    cuda.memcpy_dtoh(self.threshold_value_idx, si_gpu_in.ptr + int(indices_offset) + 
        int(row * self.stride + col) * int(self.dtype_indices.itemsize)) 
    self.feature_idx_array[nid] = row
    self.feature_threshold_array[nid] = (float(self.samples[row, self.threshold_value_idx[0]]) + self.samples[row, self.threshold_value_idx[1]]) / 2
  


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
    
    subset_indices_left = self.get_indices()
    subset_indices_right = self.get_indices()
    
    self.left_children[nid] = self.n_nodes
    self.__dfs_construct(depth + 1, min_left, 
        start_idx, start_idx + col + 1, si_gpu_out, si_gpu_in, subset_indices_left)
    
    self.right_children[nid] = self.n_nodes
    self.__dfs_construct(depth + 1, min_right, 
        start_idx + col + 1, stop_idx, si_gpu_out, si_gpu_in, subset_indices_right)

