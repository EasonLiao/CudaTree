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

def mk_kernel(n_threads, n_labels, dtype_samples, dtype_labels, dtype_counts, kernel_file = "comput_kernel_per_feature.cu",  _cache = {}):
  kernel_file = "cuda_kernels/" + kernel_file
  key = (n_threads, n_labels, dtype_samples, dtype_labels, dtype_counts)
  if key in _cache:
    return _cache[key]
  
  with open(kernel_file) as code_file:
    code = code_file.read() 
    src = code % (n_threads, n_labels, dtype_to_ctype(dtype_samples), dtype_to_ctype(dtype_labels), dtype_to_ctype(dtype_counts))
    mod = SourceModule(src)
    fn = mod.get_function("compute")
    _cache[key] = fn
    return fn

def mk_scan_kernel(n_labels, dtype_labels, dtype_counts,  kernel_file = "scan_kernel_per_feature_1.cu", _cache = {}):
  kernel_file = "cuda_kernels/" + kernel_file
  with open(kernel_file) as code_file:
    code = code_file.read()  
    src = code % (n_labels, dtype_to_ctype(dtype_labels), dtype_to_ctype(dtype_counts)) 
    mod = SourceModule(src)
    fn = mod.get_function("prefix_scan")
    return fn

def mk_scan_kernel_2(n_labels, dtype_counts, kernel_file = "scan_kernel_per_feature_2.cu"):
  kernel_file = "cuda_kernels/" + kernel_file
  with open(kernel_file) as code_file:
    code = code_file.read()  
    src = code % (n_labels, dtype_to_ctype(dtype_counts)) 
    mod = SourceModule(src)
    fn = mod.get_function("prefix_scan_2")
    return fn

def mk_scan_kernel_3(n_labels, dtype_counts, kernel_file = "scan_kernel_per_feature_3.cu"):
  kernel_file = "cuda_kernels/" + kernel_file
  with open(kernel_file) as code_file:
    code = code_file.read()  
    src = code % (n_labels, dtype_to_ctype(dtype_counts)) 
    mod = SourceModule(src)
    fn = mod.get_function("prefix_scan_3")
    return fn

def mk_fill_table_kernel(dtype_indices, kernel_file = "fill_table_per_feature.cu"):
  kernel_file = "cuda_kernels/" + kernel_file
  with open(kernel_file) as code_file:
    code = code_file.read()
    src = code % (dtype_to_ctype(dtype_indices), )
    mod = SourceModule(src)
    fn = mod.get_function("fill_table")
    return fn

def mk_shuffle_kernel(dtype_samples, dtype_labels, dtype_indices, kernel_file = "reshuffle_per_feature.cu"):
  kernel_file = "cuda_kernels/" + kernel_file
  with open(kernel_file) as code_file:
    code = code_file.read()
    src = code % (dtype_to_ctype(dtype_samples), dtype_to_ctype(dtype_labels), dtype_to_ctype(dtype_indices))
    mod = SourceModule(src)
    fn = mod.get_function("reshuffle")
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


class DecisionTreeMedium(object): 

  COMPT_THREADS_PER_BLOCK = 256  #The number of threads do computation per block.
  MAX_NUM_BLOCKS = 256

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
    self.num_blocks = None

  def __get_grid_and_range_size(self, n_samples):
    """ Get the range size of each thread and grid size. Return a tuple of (range_size, grid_size)"""
    grid_size = self.MAX_NUM_BLOCKS
    range_size = int(math.ceil(float(n_samples) / (self.COMPT_THREADS_PER_BLOCK * self.MAX_NUM_BLOCKS)))
    
    return (range_size, int(math.ceil(float(n_samples) / (self.COMPT_THREADS_PER_BLOCK * range_size))))


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
   
    #Get the size of grid and the size of range for each thread.
    range_size, num_blocks = self.__get_grid_and_range_size(target.size)

    self.max_depth = max_depth
    self.num_labels = np.max(target) + 1    
    self.stride = target.size
  
    self.dtype_indices = get_best_dtype(target.size)
    self.dtype_counts = self.dtype_indices
    self.dtype_labels = get_best_dtype(self.num_labels)
    self.dtype_samples = samples.dtype
    
    samples = np.require(np.transpose(samples), requirements = 'C')
    target = np.require(np.transpose(target), dtype = self.dtype_labels, requirements = 'C') 

    self.samples_itemsize = self.dtype_samples.itemsize
    self.labels_itemsize = self.dtype_labels.itemsize
   
    self.kernel = mk_kernel(self.COMPT_THREADS_PER_BLOCK, self.num_labels, self.dtype_samples, self.dtype_labels, self.dtype_counts)
    self.scan_kernel = mk_scan_kernel(self.num_labels, self.dtype_labels, self.dtype_counts)
    self.scan_kernel_2 = mk_scan_kernel_2(self.num_labels, self.dtype_counts)
    self.scan_kernel_3 = mk_scan_kernel_3(self.num_labels, self.dtype_counts)
    self.fill_kernel = mk_fill_table_kernel(dtype_indices = self.dtype_indices) 
    self.shuffle_kernel = mk_shuffle_kernel(self.dtype_samples, self.dtype_labels, self.dtype_indices)
      
   # """ Use prepare to improve speed """
    self.kernel.prepare("PPPPPPiii")
    self.scan_kernel.prepare("PPii") 
    self.scan_kernel_2.prepare("Piiii")
    self.scan_kernel_3.prepare("Piii")
    self.fill_kernel.prepare("PiiP")
    self.shuffle_kernel.prepare("PPPPPPPii")
    
    n_features = samples.shape[0]
    self.n_features = n_features

    """ Pre-allocate the GPU memory, don't allocate everytime in __construct"""
    self.impurity_left = gpuarray.empty(num_blocks, dtype = np.float32)
    self.impurity_right = gpuarray.empty(num_blocks, dtype = np.float32)
    self.min_split = gpuarray.empty(num_blocks, dtype = self.dtype_counts)
    self.mark_table = gpuarray.empty(target.size, dtype = np.dtype(np.uint8))
  
    sorted_indices = np.empty((n_features, target.size), dtype = self.dtype_indices)
    sorted_labels = np.empty((n_features, target.size), dtype = self.dtype_labels)
    sorted_samples = np.empty((n_features, target.size), dtype = self.dtype_samples)
 
    self.label_count = gpuarray.empty((num_blocks * self.COMPT_THREADS_PER_BLOCK + 1) * self.num_labels, dtype = self.dtype_counts)  

    with timer("argsort"):
      for i,f in enumerate(samples):
        sort_idx = np.argsort(f)
        sorted_indices[i] = sort_idx  
        sorted_labels[i] = target[sort_idx]
        sorted_samples[i] = samples[i][sort_idx]
    
    self.sorted_samples_gpu = gpuarray.to_gpu(sorted_samples)
    self.sorted_labels_gpu = gpuarray.to_gpu(sorted_labels)
    self.sorted_indices_gpu = gpuarray.to_gpu(sorted_indices)
    self.sorted_samples_gpu_ = gpuarray.empty(target.size, self.dtype_samples) #gpuarray.to_gpu(sorted_samples)
    self.sorted_indices_gpu_ = gpuarray.empty(target.size, self.dtype_indices) #gpuarray.to_gpu(sorted_indices)
    self.sorted_labels_gpu_ = gpuarray.empty(target.size, self.dtype_labels)    
    
    sorted_samples = None
    sorted_indices = None
    sorted_labels = None
   
    assert self.sorted_indices_gpu.strides[0] == target.size * self.sorted_indices_gpu.dtype.itemsize 
    assert self.sorted_labels_gpu.strides[0] == target.size * self.sorted_labels_gpu.dtype.itemsize 
    assert self.sorted_samples_gpu.strides[0] == target.size * self.sorted_samples_gpu.dtype.itemsize 
    
    self.root = self.__construct(1, 1.0, 0, target.size)
    

  def __construct(self, depth, error_rate, start_idx, stop_idx):
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
      ret_node.value = 1 #target[0]
      return ret_node
  
    range_size, num_blocks = self.__get_grid_and_range_size(n_samples)
    grid = (num_blocks, 1, 1)
    block = (self.COMPT_THREADS_PER_BLOCK, 1, 1)
    n_active_threads = int(math.ceil(float(n_samples) / range_size))
    
    min_row_idx = None
    min_col_idx = None
    min_imp_val = 4.0
    min_imp_left = 2.0
    min_imp_right = 2.0

    for f_idx in xrange(self.n_features):
      self.scan_kernel.prepared_call(
                  grid,
                  block,
                  np.intp(self.sorted_labels_gpu[f_idx].gpudata) + labels_offset, 
                  self.label_count.gpudata,
                  range_size, 
                  n_samples)
     
      self.scan_kernel_2.prepared_call(
                  (1, 1),
                  (1, 1, 1),
                  self.label_count.gpudata,
                  n_active_threads,
                  self.COMPT_THREADS_PER_BLOCK,
                  num_blocks,
                  n_samples)
   

      self.scan_kernel_3.prepared_call(
                  grid,
                  block,
                  self.label_count.gpudata,
                  n_active_threads,
                  range_size,
                  n_samples)
    
      self.kernel.prepared_call(
                  grid,
                  block,
                  np.intp(self.sorted_samples_gpu[f_idx].gpudata) + samples_offset,
                  np.intp(self.sorted_labels_gpu[f_idx].gpudata) + labels_offset,
                  self.label_count.gpudata,
                  self.impurity_left.gpudata,
                  self.impurity_right.gpudata,
                  self.min_split.gpudata,
                  range_size,
                  n_active_threads,
                  n_samples)
      
      imp_total = (self.impurity_left + self.impurity_right).get()
      idx = imp_total[:num_blocks].argmin()
      feature_min = imp_total[idx]

      if feature_min < min_imp_val:
        min_imp_val = feature_min
        min_row_idx = f_idx
        min_col_idx = self.min_split.get()[idx]
        min_imp_left = self.impurity_left.get()[idx]
        min_imp_right = self.impurity_right.get()[idx]

    if feature_min >= 4.0:
      return ret_node

    self.fill_kernel.prepared_call(
                      (1, 1),
                      (1024, 1, 1),
                      np.intp(self.sorted_indices_gpu[min_row_idx].gpudata) + indices_offset, 
                      n_samples, 
                      min_col_idx, 
                      self.mark_table.gpudata)
    
        
    for f_idx in xrange(self.n_features): 
      self.shuffle_kernel.prepared_call(
                        (1, 1),
                        (1, 1, 1),
                        self.mark_table.gpudata,
                        np.intp(self.sorted_labels_gpu[f_idx].gpudata) + labels_offset,
                        np.intp(self.sorted_indices_gpu[f_idx].gpudata) + indices_offset,
                        np.intp(self.sorted_samples_gpu[f_idx].gpudata) + samples_offset,
                        self.sorted_labels_gpu_.gpudata,
                        self.sorted_indices_gpu_.gpudata,
                        self.sorted_samples_gpu_.gpudata,
                        n_samples,
                        min_col_idx)
      
      cuda.memcpy_dtod(self.sorted_samples_gpu[f_idx].gpudata + int(samples_offset), self.sorted_samples_gpu_.gpudata, int(n_samples * self.dtype_samples.itemsize))  
      cuda.memcpy_dtod(self.sorted_labels_gpu[f_idx].gpudata + int(labels_offset), self.sorted_labels_gpu_.gpudata, int(n_samples * self.dtype_labels.itemsize))  
      cuda.memcpy_dtod(self.sorted_indices_gpu[f_idx].gpudata + int(indices_offset), self.sorted_indices_gpu_.gpudata, int(n_samples * self.dtype_indices.itemsize))  
  
    print "depth:%s  row:%s  col:%s min:%s" % (depth, min_row_idx, min_col_idx, min_imp_val)
    
    ret_node.left_child = self.__construct(depth + 1, min_imp_left, start_idx, start_idx + min_col_idx + 1)
    ret_node.right_child = self.__construct(depth + 1, min_imp_right, start_idx + min_col_idx + 1, stop_idx)
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
        print "Leaf Height : %s,  Samples : %s" % (node.height, node.samples)  
    assert self.root is not None
    recursive_print(self.root)


if __name__ == "__main__":
  import cPickle
  with open('data_batch_1', 'r') as f:
    train = cPickle.load(f)
    x_train = train['data']
    y_train = np.array(train['labels'])
  
  ds = sklearn.datasets.load_digits()
  x_train = ds.data
  y_train = ds.target
  import cProfile 
  max_depth = 6
  
  """
  with timer("Scikit-learn"):
    clf = tree.DecisionTreeMediumClassifier()    
    clf.max_depth = max_depth 
    clf = clf.fit(x_train, y_train) 
  """
  with timer("Cuda"):
    d = DecisionTreeMedium()  
    #dataset = sklearn.datasets.load_digits()
    #num_labels = len(dataset.target_names)  
    #cProfile.run("d.fit(x_train, y_train, DecisionTreeMedium.SCAN_KERNEL_P, DecisionTreeMedium.COMPUTE_KERNEL_CP)")
    
    
    d.fit(x_train, y_train,  max_depth = None)
    #d.print_tree()
