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
from threading import Condition
import Queue

class RandomDecisionTreeSmall(RandomBaseTree): 
  #COMPT_THREADS_PER_BLOCK = 64  #The number of threads do computation per block.
  #RESHUFFLE_THREADS_PER_BLOCK = 64 

  def __init__(self, task_queue, samples_gpu, labels_gpu, sorted_indices, compt_table, dtype_labels, dtype_samples, 
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
    self.task_queue = task_queue
    if self.max_features is None:
      self.max_features = int(math.ceil(math.log(self.n_features, 2)))

    self.n_remain_task = 0
    
    #self.cond = Condition()
    self.resp_queue =  Queue.Queue() 
    self.imp_left = None
    self.imp_right = None
    self.min_split = None

  def set_kernels(self, scan_kernel, comput_total_kernel, comput_label_kernel, fill_kernel, reshuffle_kernel):
    self.scan_kernel = scan_kernel
    self.comput_total_kernel = comput_total_kernel
    self.comput_label_kernel = comput_label_kernel
    self.fill_kernel = fill_kernel
    self.reshuffle_kernel = reshuffle_kernel
     
    self.impurity_left = gpuarray.empty(self.max_features, dtype = np.float32)
    self.impurity_right = gpuarray.empty(self.max_features, dtype = np.float32)
    self.min_split_gpu = gpuarray.empty(self.max_features, dtype = self.dtype_counts)
    self.mark_table = gpuarray.empty(self.stride, dtype = np.uint8)
    self.label_total = gpuarray.empty(self.n_labels, self.dtype_indices)  
    self.subset_indices = gpuarray.empty(self.max_features, dtype = self.dtype_indices)
    self.sorted_indices_gpu = gpuarray.to_gpu(self.sorted_indices)
    self.sorted_indices_gpu_ = self.sorted_indices_gpu.copy() 
    self.stream = cuda.Stream()

  def release_resources(self):
    self.impurity_left = None
    self.impurity_right = None
    self.min_split_gpu = None
    self.mark_table = None
    self.label_total = None
    self.subset_indices = None
    self.sorted_indices_gpu = None
    self.sorted_indices_gpu_ = None
    
  def __finish_job(self):
    self.task_queue.put(False)

  def notify(self):
    self.resp_queue.put(1)

  def __wait(self):
    self.resp_queue.get()

  def __send_task_req(self, req):
    self.n_remain_task += 1
    self.task_queue.put((req, self))

  def fit(self, samples, target): 
    self.samples_itemsize = self.dtype_samples.itemsize
    self.labels_itemsize = self.dtype_labels.itemsize
    self.target_value_idx = np.zeros(1, self.dtype_indices)
    self.threshold_value_idx = np.zeros(2, self.dtype_indices)
    
    
    assert self.max_features > 0 and self.max_features <= self.n_features, "max_features must be between 0 and n_features." 
    assert self.sorted_indices_gpu.strides[0] == target.size * self.sorted_indices_gpu.dtype.itemsize 
    assert self.samples_gpu.strides[0] == target.size * self.samples_gpu.dtype.itemsize   
    
    self.root = self.__construct(1, 1.0, 0, target.size, self.sorted_indices_gpu, self.sorted_indices_gpu_, self.get_indices()) 
    #self.decorate_nodes(samples, target) 
    self.__finish_job()
    print "fnish job"

  def decorate_nodes(self, samples, labels):
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

    if check_terminate():
      #cuda.memcpy_dtoh(self.target_value_idx, si_gpu_in.ptr + int(start_idx * self.dtype_indices.itemsize))
      ret_node.value = 1#self.target_value_idx[0] 
      return ret_node
    
    grid = (self.max_features, 1) 
    block = (self.COMPT_THREADS_PER_BLOCK, 1, 1)
    
    self.__send_task_req((self.scan_kernel, ((1, 1), block, self.stream, si_gpu_in.ptr + indices_offset, 
      self.labels_gpu.ptr, self.label_total.ptr, n_samples))) 
    
    self.__send_task_req((self.subset_indices.ptr, subset_indices))
    
    
    self.__send_task_req((self.comput_label_kernel, (grid, block, self.stream, si_gpu_in.ptr + indices_offset, self.samples_gpu.ptr,
      self.labels_gpu.ptr, self.impurity_left.ptr, self.impurity_right.ptr, self.label_total.ptr, self.min_split_gpu.ptr,
      self.subset_indices.ptr, n_samples, self.stride)))
    
    subset_indices_left = self.get_indices()
    subset_indices_right = self.get_indices()

    self.__send_task_req((self.impurity_left, self.impurity_right, self.min_split_gpu)) 
    self.__wait()

    imp_total = self.imp_left + self.imp_right 
    min_idx = imp_total.argmin()
    imp_min_left = self.imp_left[min_idx]
    imp_min_right = self.imp_right[min_idx]
    
    ret_node.feature_index = subset_indices[min_idx]

    row = ret_node.feature_index 
    col = self.min_split[min_idx]
    

    if imp_total[min_idx] == 4:
      print "!!!!!!!!!!!"
      #print "imp min == 4, n_samples : %s" % (n_samples, )
      #cuda.memcpy_dtoh(self.target_value_idx, si_gpu_in.ptr + int(start_idx * self.dtype_indices.itemsize))
      #ret_node.value = self.target_value_idx[0] 
      #print "######## depth : %d, n_samples: %d, row: %d, col: %d, start: %d, stop: %d" % 
      #(depth, n_samples, row, col, start_idx, stop_idx)
      return ret_node
    
    """
    cuda.memcpy_dtoh(self.threshold_value_idx, si_gpu_in.ptr + int(indices_offset) + 
        int(row * self.stride + col) * int(self.dtype_indices.itemsize))
    ret_node.feature_threshold = (row, self.threshold_value_idx[0], self.threshold_value_idx[1])
    """
    
    self.__send_task_req((self.fill_kernel, ((1, 1), (1024, 1, 1), self.stream, si_gpu_in.ptr + row * self.stride * 
      self.dtype_indices.itemsize + indices_offset, n_samples, col, self.mark_table.ptr, self.stride))) 
    
    
    block = (self.RESHUFFLE_THREADS_PER_BLOCK, 1, 1)

    self.__send_task_req((self.reshuffle_kernel, ((self.n_features, 1), block, self.stream, self.mark_table.ptr, si_gpu_in.ptr + 
      indices_offset, si_gpu_out.ptr + indices_offset, n_samples, col, self.stride))) 
    
    
    ret_node.left_child = self.__construct(depth + 1, imp_min_left, 
        start_idx, start_idx + col + 1, si_gpu_out, si_gpu_in, subset_indices_left)
    ret_node.right_child = self.__construct(depth + 1, imp_min_right, 
        start_idx + col + 1, stop_idx, si_gpu_out, si_gpu_in, subset_indices_right)
    return ret_node 


if __name__ == "__main__":
  x_train, y_train = datasource.load_data("train") 
  
  with timer("Cuda"):
    d = RandomDecisionTreeSmall()  
    d.fit(x_train, y_train)
    print "end"
    #d.print_tree()
    #print np.allclose(d.predict(x_train), y_train)
