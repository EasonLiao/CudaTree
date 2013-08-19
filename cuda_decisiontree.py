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

def mk_kernel(n_samples, n_labels, dtype_samples, dtype_labels, dtype_counts, kernel_file,  _cache = {}):
  kernel_file = "cuda_kernels/" + kernel_file
  key = (n_samples, n_labels, dtype_samples, dtype_labels)
  if key in _cache:
    return _cache[key]
  
  with open(kernel_file) as code_file:
    code = code_file.read() 
    src = code % (n_samples, n_labels, dtype_to_ctype(dtype_samples), dtype_to_ctype(dtype_labels), dtype_to_ctype(dtype_counts))
    mod = SourceModule(src)
    fn = mod.get_function("compute")
    _cache[key] = fn
    return fn

def mk_scan_kernel(n_samples, n_labels, kernel_file, n_threads, dtype_labels, dtype_counts,  _cache = {}):
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
  """
  COMPUTE_KERNEL_SS = "comput_kernel_ss.cu"   #One thread per feature.
  COMPUTE_KERNEL_PS = "comput_kernel_ps.cu"   #One block per feature.
  COMPUTE_KERNEL_PP = "comput_kernel_pp.cu"   #Based on kernel 2, add parallel reduction to find minimum impurity.
  COMPUTE_KERNEL_CP = "comput_kernel_cp.cu"   #Based on kernel 3, utilized the coalesced memory access.
  COMPUTE_KERNEL_CP_CM = "comput_kernel_cp_cm.cu"   #Based on kernel 3, utilized the coalesced memory access.
  SCAN_KERNEL_S = "scan_kernel_s.cu"          #Serialized prefix scan.
  SCAN_KERNEL_P = "scan_kernel_p.cu"          #Simple parallel prefix scan.
  SCAN_KERNEL_P_CM = "scan_kernel_p_cm.cu"          #Simple parallel prefix scan. 
  """
  COMPUTE_KERNEL_SS = "comput_kernel_ss.cu"   #One thread per feature.
  
  SHUFFLE_KERNEL = "reshuffle.cu"
  COMPUTE_KERNEL_PART = "comput_kernel_pp_part.cu"   
  COMPUTE_KERNEL_PART_SK = "comput_kernel_pp_part_skip.cu"   
  SCAN_KERNEL_PART = "scan_kernel_p_part.cu"          

  COMPT_THREADS_PER_BLOCK = 64  #The number of threads do computation per block.
  SCAN_THREADS_PER_BLOCK = 64   #The number of threads do prefix scan per block.

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

  def fit(self, samples, target, scan_kernel_type, compt_kernel_type, max_depth = None):
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
      

    self.compt_kernel_type = compt_kernel_type

    samples = np.require(np.transpose(samples), requirements = 'C')
    target = np.require(np.transpose(target), dtype = self.dtype_labels, requirements = 'C') 
    
    self.samples_itemsize = self.dtype_samples.itemsize
    self.labels_itemsize = self.dtype_labels.itemsize
    
    self.kernel = mk_kernel(target.size, self.num_labels, self.dtype_samples, self.dtype_labels, self.dtype_counts, compt_kernel_type)
    self.scan_kernel = mk_scan_kernel(target.size, self.num_labels, scan_kernel_type, self.SCAN_THREADS_PER_BLOCK, self.dtype_labels, self.dtype_counts)
    self.fill_kernel = mk_fill_table_kernel(dtype_indices = self.dtype_indices)
    self.shuffle_kernel = mk_shuffle_kernel(self.dtype_samples, self.dtype_labels, self.dtype_indices)
    
    """ Use prepare to improve speed """
    self.kernel.prepare("PPPPPPiii")
    self.scan_kernel.prepare("PPiii") 
    self.fill_kernel.prepare("PiiPi")
    self.shuffle_kernel.prepare("PPPPPPPiii")

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
      ret_node.value = 1 #target[0]
      return ret_node
  
    grid = (self.n_features, 1) 
    if self.compt_kernel_type !=  self.COMPUTE_KERNEL_SS:
      block = (self.COMPT_THREADS_PER_BLOCK, 1, 1)
    else:
      block = (1, 1, 1)
    
    self.scan_kernel.prepared_call(
                grid,
                (self.SCAN_THREADS_PER_BLOCK, 1, 1),
                np.intp(gpuarrays_in[1].gpudata) + labels_offset, 
                self.label_count.gpudata,
                self.n_features, 
                n_samples, 
                self.stride) 



    self.kernel.prepared_call(
              grid,
              block,
              np.intp(gpuarrays_in[2].gpudata) + samples_offset,
              np.intp(gpuarrays_in[1].gpudata) + labels_offset,
              self.impurity_left.gpudata,
              self.impurity_right.gpudata,
              self.label_count.gpudata,
              self.min_split.gpudata,
              self.n_features, 
              n_samples, 
              self.stride)
            

    imp_right = self.impurity_right.get()
    imp_left = self.impurity_left.get()
     

    imp_total = imp_left + imp_right
    
    ret_node.feature_index =  imp_total.argmin()

    if imp_total[ret_node.feature_index] == 4:
      return ret_node

    row = ret_node.feature_index
    col = self.min_split.get()[row]
    
    #ret_node.feature_threshold = (sorted_samples[row][col] + sorted_samples[row][col + 1]) / 2.0 
    self.fill_kernel.prepared_call(
                      (1, 1),
                      (1024, 1, 1),
                      np.intp(gpuarrays_in[0].gpudata) + row * self.stride * self.dtype_indices.itemsize + indices_offset, 
                      n_samples, 
                      col, 
                      self.mark_table.gpudata, 
                      self.stride
                      )
    
    self.shuffle_kernel.prepared_call(
                      (self.n_features, 1),
                      (1, 1, 1),
                      self.mark_table.gpudata,
                      np.intp(gpuarrays_in[1].gpudata) + labels_offset,
                      np.intp(gpuarrays_in[0].gpudata) + indices_offset,
                      np.intp(gpuarrays_in[2].gpudata) + samples_offset,
                      np.intp(gpuarrays_out[1].gpudata) + labels_offset,
                      np.intp(gpuarrays_out[0].gpudata) + indices_offset,
                      np.intp(gpuarrays_out[2].gpudata) + samples_offset,
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
        print "Leaf Height : %s,  Samples : %s" % (node.height, node.samples)  
    assert self.root is not None
    recursive_print(self.root)

if __name__ == "__main__":
  
  x_train, y_train = datasource.load_data("db") 

  """
  data = cPickle.load(open("/scratch1/imagenet-pickle/train-data.pickle.0"))
  x_train = data['fc']
  print x_train.shape
  y_train = data['labels'].reshape(122111)
  """

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
    d.fit(x_train, y_train, DecisionTree.SCAN_KERNEL_PART, DecisionTree.COMPUTE_KERNEL_PART)
    #d.print_tree()
