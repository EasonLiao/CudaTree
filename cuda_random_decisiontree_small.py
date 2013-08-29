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

class RandomDecisionTreeSmall(RandomBaseTree): 
  COMPT_THREADS_PER_BLOCK = 32  #The number of threads do computation per block.
  RESHUFFLE_THREADS_PER_BLOCK = 64 

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

  def fit(self, samples, target, max_features = None, max_depth = None):
    def compile_kernels():
      ctype_indices = dtype_to_ctype(self.dtype_indices)
      ctype_labels = dtype_to_ctype(self.dtype_labels)
      ctype_counts = dtype_to_ctype(self.dtype_counts)
      ctype_samples = dtype_to_ctype(self.dtype_samples)
      n_labels = self.num_labels
      n_threads = self.COMPT_THREADS_PER_BLOCK
      n_shf_threads = self.RESHUFFLE_THREADS_PER_BLOCK
      
      self.fill_kernel = mk_kernel((ctype_indices,), "fill_table", "fill_table_si.cu") 
      #self.scan_reshuffle_kernel = mk_kernel((ctype_indices, n_shf_threads), 
      #    "scan_reshuffle", "pos_scan_reshuffle_si_c.cu")
      self.scan_total_kernel = mk_kernel((n_threads, n_labels, ctype_labels, ctype_counts, ctype_indices), 
          "count_total", "scan_kernel_total_si.cu") 
      self.comput_total_kernel, tex_ref = mk_tex_kernel((n_threads, n_labels, ctype_samples, ctype_labels, 
        ctype_counts, ctype_indices), "compute", "tex_label_total", "comput_kernel_total_rand.cu")
      self.label_total.bind_to_texref_ext(tex_ref)
      self.scan_reshuffle_tex, tex_ref = mk_tex_kernel((ctype_indices, n_shf_threads), 
          "scan_reshuffle", "tex_mark", "pos_scan_reshuffle_si_c_tex.cu")   
      self.mark_table.bind_to_texref_ext(tex_ref) 
      self.comput_label_loop_kernel, tex_ref = mk_tex_kernel((n_threads, n_labels, ctype_samples, 
        ctype_labels, ctype_counts, ctype_indices), "compute", "tex_label_total", "comput_kernel_label_loop_si.cu") 
      self.label_total.bind_to_texref_ext(tex_ref)
      
      self.comput_label_loop_rand_kernel, tex_ref = mk_tex_kernel((n_threads, n_labels, ctype_samples, 
        ctype_labels, ctype_counts, ctype_indices), "compute", "tex_label_total", "comput_kernel_label_loop_rand.cu") 
      self.label_total.bind_to_texref_ext(tex_ref)

      """ Use prepare to improve speed """
      self.fill_kernel.prepare("PiiPi")
      #self.scan_reshuffle_kernel.prepare("PPPiiiii")
      self.scan_reshuffle_tex.prepare("PPPiii") 
      self.scan_total_kernel.prepare("PPPi")
      self.comput_total_kernel.prepare("PPPPPPPPii")
      self.comput_label_loop_kernel.prepare("PPPPPPPii")
      self.comput_label_loop_rand_kernel.prepare("PPPPPPPPii")

    def allocate_gpuarrays():
      """ Pre-allocate the GPU memory, don't allocate everytime in __construct"""
      self.impurity_left = gpuarray.empty(self.max_features, dtype = np.float32)
      self.impurity_right = gpuarray.empty(self.max_features, dtype = np.float32)
      self.min_split = gpuarray.empty(self.max_features, dtype = self.dtype_counts)
      self.mark_table = gpuarray.empty(target.size, dtype = np.uint8)
      #self.pos_mark_table = gpuarray.empty(self.RESHUFFLE_THREADS_PER_BLOCK * self.n_features, dtype = self.dtype_indices)
      self.label_total = gpuarray.empty(self.num_labels, self.dtype_indices)  
      self.subset_indices = gpuarray.empty(self.max_features, dtype = self.dtype_indices)
    
    def sort_arrays():
      sorted_indices = np.empty((self.n_features, target.size), dtype = self.dtype_indices)
      with timer("argsort"):
        for i,f in enumerate(samples):
          sort_idx = np.argsort(f)
          sorted_indices[i] = sort_idx  
      self.sorted_indices_gpu = gpuarray.to_gpu(sorted_indices)
      self.sorted_indices_gpu_ = self.sorted_indices_gpu.copy()
      self.samples_gpu = gpuarray.to_gpu(samples)
      self.labels_gpu = gpuarray.to_gpu(target)
   
    def release_gpuarrays():
      self.impurity_left = None
      self.impurity_right = None
      self.min_split = None
      self.mark_table = None
      self.label_total = None
      self.subset_indices = None
      self.sorted_indices_gpu = None
      self.sorted_indices_gpu_ = None
      self.samples_gpu = None
      self.labels_gpu = None
    
    def compact_labels():
      self.compt_table = np.unique(target)
      self.compt_table.sort()
      
      if self.compt_table.size != int(np.max(target)) + 1:
        for i, val in enumerate(self.compt_table):
          np.place(target, target == val, i)  
      self.num_labels = self.compt_table.size
    
    target = target.copy()
    assert isinstance(samples, np.ndarray)
    assert isinstance(target, np.ndarray)
    assert samples.size / samples[0].size == target.size
    
    compact_labels()
    self.max_depth = max_depth
    self.stride = target.size
    
    self.dtype_indices = get_best_dtype(target.size)
    self.dtype_counts = self.dtype_indices
    self.dtype_labels = get_best_dtype(self.num_labels)
    self.dtype_samples = samples.dtype
      
    samples = np.require(np.transpose(samples), requirements = 'C')
    target = np.require(np.transpose(target), dtype = self.dtype_labels, requirements = 'C') 
    
    self.samples_itemsize = self.dtype_samples.itemsize
    self.labels_itemsize = self.dtype_labels.itemsize
    self.target_value_idx = np.zeros(1, self.dtype_indices)
    self.threshold_value_idx = np.zeros(2, self.dtype_indices)
    self.n_features = samples.shape[0]
    
    if max_features is None:
      self.max_features = int(math.ceil(math.log(self.n_features, 2)))
    else: self.max_features = max_features
    
    assert self.max_features > 0 and self.max_features <= self.n_features, "max_features must be between 0 and n_features."
    
    allocate_gpuarrays()
    compile_kernels()
    sort_arrays() 
    
    assert self.sorted_indices_gpu.strides[0] == target.size * self.sorted_indices_gpu.dtype.itemsize 
    assert self.samples_gpu.strides[0] == target.size * self.samples_gpu.dtype.itemsize   
    self.root = self.__construct(1, 1.0, 0, target.size, self.sorted_indices_gpu, self.sorted_indices_gpu_) 
    release_gpuarrays()
    self.decorate_nodes(samples, target) 


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


  def __construct(self, depth, error_rate, start_idx, stop_idx, si_gpu_in, si_gpu_out):
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

    if check_terminate():
      cuda.memcpy_dtoh(self.target_value_idx, si_gpu_in.ptr + int(start_idx * self.dtype_indices.itemsize))
      ret_node.value = self.target_value_idx[0] 
      return ret_node
    

    subset_indices = self.get_indices()
    cuda.memcpy_htod(self.subset_indices.ptr, subset_indices)

    grid = (self.max_features, 1) 
    block = (self.COMPT_THREADS_PER_BLOCK, 1, 1)
    
    self.scan_total_kernel.prepared_call(
                (1, 1),
                block,
                si_gpu_in.ptr + indices_offset,
                self.labels_gpu.ptr,
                self.label_total.ptr,
                n_samples)
    """ 
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
    """
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
    
    imp_right = self.impurity_right.get()
    imp_left = self.impurity_left.get() 
    imp_total = imp_left + imp_right 
    min_idx = imp_total.argmin()

    ret_node.feature_index = subset_indices[min_idx]

    row = ret_node.feature_index 
    col = self.min_split.get()[min_idx]
    
    if imp_total[min_idx] == 4:
      print "imp min == 4"
      cuda.memcpy_dtoh(self.target_value_idx, si_gpu_in.ptr + int(start_idx * self.dtype_indices.itemsize))
      ret_node.value = self.target_value_idx[0] 
      #print "######## depth : %d, n_samples: %d, row: %d, col: %d, start: %d, stop: %d" % 
      #(depth, n_samples, row, col, start_idx, stop_idx)
      return ret_node
    
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

    
    ret_node.left_child = self.__construct(depth + 1, imp_left[min_idx], 
        start_idx, start_idx + col + 1, si_gpu_out, si_gpu_in)
    ret_node.right_child = self.__construct(depth + 1, imp_right[min_idx], 
        start_idx + col + 1, stop_idx, si_gpu_out, si_gpu_in)
    return ret_node 


if __name__ == "__main__":
  x_train, y_train = datasource.load_data("train") 
  
  with timer("Cuda"):
    d = RandomDecisionTreeSmall()  
    d.fit(x_train, y_train)
    #d.print_tree()
    #print np.allclose(d.predict(x_train), y_train)
