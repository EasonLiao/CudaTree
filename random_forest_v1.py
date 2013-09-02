import numpy as np
from cuda_random_decisiontree_small_v1 import RandomDecisionTreeSmall
from datasource import load_data
from util import timer, get_best_dtype, dtype_to_ctype, mk_kernel, mk_tex_kernel
from pycuda import gpuarray
from threading import Thread
from time import sleep
import Queue
import threading
import pycuda.driver as cuda

class BuildingWorker(threading.Thread):
  def set_params(self, tree, samples, target):
    self.tree = tree
    self.samples = samples
    self.target = target

  def run(self):
    #print self.tree
    self.tree.fit(self.samples, self.target)
  
  def release(self):
    self.tree.release_resources()

class RandomForest(object):
  COMPT_THREADS_PER_BLOCK = 32 
  RESHUFFLE_THREADS_PER_BLOCK = 64 
  
  def __init__(self):
    self.task_queue = Queue.Queue() 

  def __compile_kernels(self):
    ctype_indices = dtype_to_ctype(self.dtype_indices)
    ctype_labels = dtype_to_ctype(self.dtype_labels)
    ctype_counts = dtype_to_ctype(self.dtype_counts)
    ctype_samples = dtype_to_ctype(self.dtype_samples)
    n_labels = self.n_labels
    n_threads = self.COMPT_THREADS_PER_BLOCK
    n_shf_threads = self.RESHUFFLE_THREADS_PER_BLOCK
    
    self.fill_kernel = mk_kernel((ctype_indices,), "fill_table", "fill_table_si.cu") 
    self.scan_kernel = mk_kernel((n_threads, n_labels, ctype_labels, ctype_counts, ctype_indices), 
        "count_total", "scan_kernel_total_si.cu") 
    self.comput_total_kernel = mk_kernel((n_threads, n_labels, ctype_samples, ctype_labels, 
      ctype_counts, ctype_indices), "compute",  "comput_kernel_total_rand.cu")    
    self.reshuffle_kernel = mk_kernel((ctype_indices, n_shf_threads), 
        "scan_reshuffle",  "pos_scan_reshuffle_si_c.cu")   
    self.comput_label_kernel  = mk_kernel((n_threads, n_labels, ctype_samples, 
      ctype_labels, ctype_counts, ctype_indices), "compute",  "comput_kernel_label_loop_rand.cu") 
    
    if hasattr(self.fill_kernel, "is_prepared"):
      return
    
    self.fill_kernel.is_prepared = True
    """ Use prepare to improve speed """
    self.fill_kernel.prepare("PiiPi")
    self.reshuffle_kernel.prepare("PPPiii") 
    self.scan_kernel.prepare("PPPi")
    self.comput_total_kernel.prepare("PPPPPPPPii")
    self.comput_label_kernel.prepare("PPPPPPPPii")


  def __compact_labels(self, target):
    self.compt_table = np.unique(target)
    self.compt_table.sort()   
    if self.compt_table.size != int(np.max(target)) + 1:
      for i, val in enumerate(self.compt_table):
        np.place(target, target == val, i) 
    self.n_labels = self.compt_table.size 


  def __wait_and_launch_kernels(self, n_remain):
    while True:
      req = self.task_queue.get()
      if isinstance(req, bool):
        n_remain -= 1
        if n_remain == 0:
          return
        else:
          continue

      req, receiver = req

      if isinstance(req, tuple):
        if isinstance(req[1], np.ndarray):
          cuda.memcpy_htod(*req)  
        elif isinstance(req[0], gpuarray.GPUArray) and isinstance(req[1], gpuarray.GPUArray):
          receiver.imp_left = req[0].get()
          receiver.imp_right = req[1].get()
          receiver.min_split = req[2].get()
          receiver.notify()            
        else:
          req[0].prepared_async_call(*req[1])
      else:
        print req
        print ""


  def __construct_forest(self, n_jobs, samples, target):
    n_remain_trees = len(self.forest)
    n_tree_idx = 0

    while n_remain_trees > 0:
      if n_remain_trees > n_jobs:
        n_threads = n_jobs
      else:
        n_threads = n_remain_trees
      
      workers = [BuildingWorker() for i in xrange(n_threads)]

      for i in xrange(n_threads):
        self.forest[n_tree_idx].set_kernels(self.scan_kernel, self.comput_total_kernel, self.comput_label_kernel,
            self.fill_kernel, self.reshuffle_kernel)
         
        workers[i].set_params(self.forest[n_tree_idx], samples, target)
        n_tree_idx += 1
      
      for i in xrange(n_threads):
        print "start"
        workers[i].start()

      self.__wait_and_launch_kernels(n_threads)
      
      for i, w in enumerate(workers):
        print "worker %s finish..." % (i, )
        workers[i].join()
        #self.forest[i].print_tree()
        print " --------  "

      n_remain_trees -= n_threads


  def fit(self, samples, target, n_trees = 4, n_jobs = 1, max_features = None, max_depth = None):
    assert isinstance(samples, np.ndarray)
    assert isinstance(target, np.ndarray)
    assert samples.size / samples[0].size == target.size
    
    target = target.copy()
    self.__compact_labels(target) 
    self.dtype_indices = get_best_dtype(target.size)
    self.dtype_counts = self.dtype_indices
    self.dtype_labels = get_best_dtype(self.n_labels)
    self.dtype_samples = samples.dtype
   
    samples = np.require(np.transpose(samples), requirements = 'C')
    target = np.require(np.transpose(target), dtype = self.dtype_labels, requirements = 'C') 
    
    self.n_features = samples.shape[0]
    self.n_samples = target.size

    samples_gpu = gpuarray.to_gpu(samples)
    labels_gpu = gpuarray.to_gpu(target) 
    
    sorted_indices = np.empty((self.n_features, target.size), dtype = self.dtype_indices)
    
    self.__compile_kernels()

    with timer("argsort"):
      for i,f in enumerate(samples):
        sort_idx = np.argsort(f)
        sorted_indices[i] = sort_idx  
    
    self.forest = [RandomDecisionTreeSmall(self.task_queue, samples_gpu, labels_gpu, sorted_indices, self.compt_table, 
      self.dtype_labels,self.dtype_samples, self.dtype_indices, self.dtype_counts,
      self.n_features, self.n_samples, self.n_labels, self.COMPT_THREADS_PER_BLOCK,
      self.RESHUFFLE_THREADS_PER_BLOCK, max_features, max_depth) for i in xrange(n_trees)]   
    
    self.__construct_forest(n_jobs, samples, target)
    return

    for i, tree in enumerate(self.forest):
      with timer("Tree %s" % (i,)):
        tree.fit(samples, target)
   

  def predict(self, x):
    res = []
    for tree in self.forest:
      res.append(tree.predict(x))
    res = np.array(res)
    return np.array([np.argmax(np.bincount(res[:,i])) for i in xrange(res.shape[1])])


if __name__ == "__main__":
  x_train, y_train = load_data("db")
  x_test, y_test = load_data("db")

  ft = RandomForest()
  with timer("Cuda fit"):
    ft.fit(x_train, y_train)
  
  """ 
  with timer("Cuda predict"):
    pre_res  = ft.predict(x_test)
  """
  """
  diff = pre_res - y_test
  print "diff: %s, total: %s" % (np.count_nonzero(diff), pre_res.size)
  """
  """
  t = RandomDecisionTreeSmall()
  t.fit(x_train, y_train, 100)
  print t.predict(x_test)[0:20]
  print y_test[0:20]
  """

  #t.print_tree()
  #ft = RandomForest() 
  
