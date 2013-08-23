import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from sklearn import tree
import sklearn.datasets
from sklearn.datasets import load_iris
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import math
import sys
import time 
from time import sleep
import datasource 

class timer(object):
  def __init__(self, name):
    self.name = name

  def __enter__(self, *args):
    print "Running %s" % self.name 
    self.start_t = time.time()

  def __exit__(self, *args):
    print "Time for %s: %s" % (self.name, time.time() - self.start_t)

def dtype_to_ctype(dtype):
  if dtype.kind == 'f':
    if dtype == 'float32':
      return 'float'
    else:
      assert dtype == 'float64', "Unsupported dtype %s" % dtype
      return 'double'
  assert dtype.kind in ('u', 'i')
  return "%s_t" % dtype 

def mk_kernel(n_threads, n_labels, dtype_samples, dtype_labels, dtype_counts, kernel_file,  _cache = {}):
  kernel_file = "cuda_kernels/" + kernel_file
  key = (n_threads, n_labels, dtype_samples, dtype_labels)
  if key in _cache:
    return _cache[key]
  
  with open(kernel_file) as code_file:
    code = code_file.read() 
    src = code % (n_threads, n_labels, dtype_to_ctype(dtype_samples), dtype_to_ctype(dtype_labels), dtype_to_ctype(dtype_counts))
    mod = SourceModule(src)
    fn = mod.get_function("compute")
    _cache[key] = fn
    return fn

def mk_scan_kernel(n_samples, n_labels, n_threads, dtype_labels, dtype_counts, kernel_file, _cache = {}):
  kernel_file = "cuda_kernels/" + kernel_file
  key = (n_samples, n_labels)
  if key in _cache:
    return _cache[key]
  
  with open(kernel_file) as code_file:
    code = code_file.read()  
    src = code % (n_samples, n_labels, n_threads, dtype_to_ctype(dtype_labels), dtype_to_ctype(dtype_counts))
    mod = SourceModule(src)
    fn = mod.get_function("prefix_scan")
    _cache[key] = fn
    return fn

def mk_fill_table_kernel(dtype_indices, kernel_file = "fill_table.cu"):
  kernel_file = "cuda_kernels/" + kernel_file
  with open(kernel_file) as code_file:
    code = code_file.read()
    src = code % (dtype_to_ctype(dtype_indices), )
    mod = SourceModule(src)
    fn = mod.get_function("fill_table")
    return fn

def mk_shuffle_kernel(dtype_samples, dtype_labels, dtype_indices, kernel_file = "reshuffle.cu"):
  kernel_file = "cuda_kernels/" + kernel_file
  with open(kernel_file) as code_file:
    code = code_file.read()
    src = code % (dtype_to_ctype(dtype_samples), dtype_to_ctype(dtype_labels), dtype_to_ctype(dtype_indices)) 
    mod = SourceModule(src)
    fn = mod.get_function("reshuffle")
    return fn

def mk_pos_scan_shuffle_kernel(dtype_indices, dtype_samples, dtype_labels, n_threads, kernel_file = "pos_scan_reshuffle_reg_c.cu"):
  kernel_file = "cuda_kernels/" + kernel_file
  with open(kernel_file) as code_file:
    code = code_file.read()
    src = code % (dtype_to_ctype(dtype_indices), dtype_to_ctype(dtype_samples), dtype_to_ctype(dtype_labels), n_threads) 
    mod = SourceModule(src)
    fn = mod.get_function("scan_reshuffle")
    return fn

def mk_scan_compute_kernel(n_labels, n_threads, dtype_labels, dtype_counts, dtype_samples, kernel_file = "scan_comput_kernel.cu"):
  kernel_file = "cuda_kernels/" + kernel_file
  with open(kernel_file) as code_file:
    code = code_file.read()
    src = code % (n_labels, n_threads, dtype_to_ctype(dtype_labels), dtype_to_ctype(dtype_counts), dtype_to_ctype(dtype_samples)) 
    mod = SourceModule(src)
    fn = mod.get_function("scan_compute")
    return fn

def mk_scan_total(n_threads, n_labels, dtype_labels, dtype_counts, kernel_file = "scan_kernel_total.cu"):
  kernel_file = "cuda_kernels/" + kernel_file
  with open(kernel_file) as code_file:
    code = code_file.read()
    src = code % (n_threads, n_labels, dtype_to_ctype(dtype_labels), dtype_to_ctype(dtype_counts)) 
    mod = SourceModule(src)
    fn = mod.get_function("count_total")
    return fn

def mk_compute_total(n_threads, n_labels, dtype_samples, dtype_labels, dtype_counts, kernel_file = "comput_kernel_total.cu"):
  kernel_file = "cuda_kernels/" + kernel_file
  with open(kernel_file) as code_file:
    code = code_file.read()
    src = code % (n_threads, n_labels, dtype_to_ctype(dtype_samples), dtype_to_ctype(dtype_labels), dtype_to_ctype(dtype_counts)) 
    mod = SourceModule(src)
    fn = mod.get_function("compute")
    return fn


class Node(object):
  def __init__(self):
    self.value = None 
    self.error = None
    self.samples = None
    self.feature_threshold = None
    self.feature_index = None
    self.left_child = None
    self.right_child = None
    self.height = None


class DecisionTree(object): 

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
    def get_best_dtype(max_value):
      """ Find the best dtype to minimize the memory usage"""
      if max_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
      if max_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
      if max_value <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
      else:
        return np.dtype(np.uint64)
         
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

    self.kernel = mk_kernel(self.COMPT_THREADS_PER_BLOCK, self.num_labels, self.dtype_samples, self.dtype_labels, self.dtype_counts, "comput_kernel_pp_part.cu")
    self.scan_kernel = mk_scan_kernel(target.size, self.num_labels, self.COMPT_THREADS_PER_BLOCK, self.dtype_labels, self.dtype_counts, "scan_kernel_p_part.cu")
    self.fill_kernel = mk_fill_table_kernel(dtype_indices = self.dtype_indices)
    #self.shuffle_kernel = mk_shuffle_kernel(self.dtype_samples, self.dtype_labels, self.dtype_indices)
    self.shuffle_kernel = mk_pos_scan_shuffle_kernel(self.dtype_indices, self.dtype_samples, self.dtype_labels, self.RESHUFFLE_THREADS_PER_BLOCK, "pos_scan_reshuffle_c.cu")
    #self.scan_comput_kernel = mk_scan_compute_kernel(self.num_labels, self.COMPT_THREADS_PER_BLOCK, self.dtype_labels, self.dtype_counts, self.dtype_samples)
    self.count_total_kernel = mk_scan_total(self.COMPT_THREADS_PER_BLOCK, self.num_labels, self.dtype_labels, self.dtype_counts) 
    self.comput_total_kernel = mk_compute_total(self.COMPT_THREADS_PER_BLOCK, self.num_labels, self.dtype_samples, self.dtype_labels, self.dtype_counts, "comput_kernel_total.cu")


    """ Use prepare to improve speed """
    self.kernel.prepare("PPPPPPiii")
    self.scan_kernel.prepare("PPiii") 
    self.fill_kernel.prepare("PiiPi")
    self.shuffle_kernel.prepare("PPPPPPPiii")
    #self.shuffle_kernel.prepare("PPPPPPPiii")
    #self.scan_comput_kernel.prepare("PPPPPPiiii") 
    self.count_total_kernel.prepare("PPi")
    self.comput_total_kernel.prepare("PPPPPPii") 
    

    n_features = samples.shape[0]
    self.n_features = n_features

    """ Pre-allocate the GPU memory, don't allocate everytime in __construct"""
    self.impurity_left = gpuarray.empty(n_features, dtype = np.float32)
    self.impurity_right = gpuarray.empty(n_features, dtype = np.float32)
    self.min_split = gpuarray.empty(n_features, dtype = self.dtype_counts)
    self.mark_table = gpuarray.empty(target.size, dtype = np.uint8)

    sorted_indices = np.empty((n_features, target.size), dtype = self.dtype_indices)
    sorted_labels = np.empty((n_features, target.size), dtype = self.dtype_labels)
    sorted_samples = np.empty((n_features, target.size), dtype = self.dtype_samples)
    

    #self.label_count = gpuarray.empty(2, dtype = self.dtype_counts)  
    self.label_count = gpuarray.empty((self.COMPT_THREADS_PER_BLOCK + 1) * self.num_labels * samples.shape[0], dtype = self.dtype_counts)  
    self.label_total = gpuarray.empty(self.num_labels, dtype = self.dtype_counts)
    

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
     
    
    sorted_samples = None
    sorted_indices = None
    sorted_labels = None

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

  def __predict(self, val):
    temp = self.root
    while True:
      if temp.left_child and temp.right_child:
        if val[temp.feature_index] < temp.feature_threshold:
          temp = temp.left_child
        else:
          temp = temp.right_child
      else: 
          return temp.value

  def predict(self, inputs):
    res = []
    for val in inputs:
      res.append(self.__predict(val))
    return np.array(res)

  def print_tree(self):
    def recursive_print(node):
      if node.left_child and node.right_child:
        print "Height : %s,  Feature Index : %s,  Threshold : %s Samples: %s" % (node.height, node.feature_index, node.feature_threshold, node.samples)  
        recursive_print(node.left_child)
        recursive_print(node.right_child)
      else:
        print "Leaf Height : %s,  Samples : %s,  Value: %s" % (node.height, node.samples, node.value)  
    assert self.root is not None
    recursive_print(self.root)

if __name__ == "__main__": 
  x_train, y_train = datasource.load_data("db") 

  """
  with timer("Scikit-learn"):
    clf = tree.DecisionTreeClassifier()    
    clf.max_depth = max_depth 
    clf = clf.fit(x_train, y_train) 
  """
  with timer("Cuda"):
    d = DecisionTree()  
    #dataset = sklearn.datasets.load_digits()
    #num_labels = len(dataset.target_names)  
    #cProfile.run("d.fit(x_train, y_train, DecisionTree.SCAN_KERNEL_P, DecisionTree.COMPUTE_KERNEL_CP)")
    d.fit(x_train, y_train)
    #d.print_tree()
    print np.allclose(d.predict(x_train), y_train)

