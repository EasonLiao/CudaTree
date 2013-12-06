from sklearn.ensemble import RandomForestClassifier as skRF
from pycuda import driver
import multiprocessing
import numpy as np
from cudatree import RandomForestClassifier as cdRF
from cudatree import RandomClassifierTree, util 

class CPUBuilder(multiprocessing.Process):
  """
  Build some trees on cpu, the cpu classifier should be cpu 
  implementation of random forest classifier.
  """
  def __init__(self,
                cpu_classifier,
                X,
                Y,
                bootstrap,
                max_features,
                n_jobs,
                remain_trees,
                lock):
    multiprocessing.Process.__init__(self)
    self.cpu_classifier = cpu_classifier
    self.X = X
    self.Y = Y
    self.bootstrap = bootstrap
    self.max_features = max_features
    self.n_jobs = n_jobs
    self.remain_trees = remain_trees
    self.lock = lock
    self.result_queue = multiprocessing.Queue()

  def run(self):
    lock = self.lock
    remain_trees = self.remain_trees
    cpu_classifier = self.cpu_classifier
    max_features = self.max_features
    bootstrap = self.bootstrap
    n_jobs = self.n_jobs
    result_queue = self.result_queue
    forests = list()
    X = self.X
    Y = self.Y

    if max_features == None:
      max_features = "auto"
    
    self.Y = self.Y.astype(np.uint16) 
    classifier_name = cpu_classifier.__name__

    while True:
      lock.acquire()
      if remain_trees.value < 80:
        lock.release()
        break

      remain_trees.value -= n_jobs
      lock.release()

      util.log_info("%s got %s jobs.", classifier_name, n_jobs)
      f = cpu_classifier(n_estimators = n_jobs, n_jobs = n_jobs, 
          bootstrap = bootstrap, max_features = max_features)
      f.fit(X, Y)
      forests.append(f)

    result_queue.put(forests)
    util.log_info("CPU DONE")

  def get_result(self):
    return self.result_queue.get()


class GPUBuilder(multiprocessing.Process):
  "Build some trees on gpu" 
  def __init__(self, 
                gpu_id,
                X,
                Y,
                bootstrap,
                max_features,
                bfs_threshold,
                remain_trees,
                lock):
    multiprocessing.Process.__init__(self)
    self.gpu_id = gpu_id
    self.X = X
    self.Y = Y
    self.bootstrap = bootstrap
    self.max_features = max_features
    self.bfs_threshold = bfs_threshold
    self.remain_trees = remain_trees
    self.lock = lock
    self.result_queue = multiprocessing.Queue()

  def run(self):
    reload(util)
    driver.init()
    ctx = driver.Device(self.gpu_id).make_context()

    X = self.X
    Y = self.Y
    bootstrap = self.bootstrap
    max_features = self.max_features
    bfs_threshold = self.bfs_threshold
    remain_trees = self.remain_trees
    lock = self.lock

    trees = list()
    forest = cdRF(n_estimators = 1,
                    bootstrap = bootstrap, 
                    max_features = max_features)
    forest.fit_init(X, Y)
    
    while True:
      lock.acquire()
      if remain_trees.value == 0:
        lock.release()
        break
      
      remain_trees.value -= 1
      lock.release()
      
      util.log_info("CUDA gets 1 job")
      tree = RandomClassifierTree(forest)   
      
      si, n_samples = forest._get_sorted_indices(forest.sorted_indices)
      tree.fit(forest.samples, forest.target, si, n_samples)
      trees.append(tree)

    forest.fit_release()
    self.result_queue.put(trees)

    ctx.detach()
    del ctx
    util.log_info("GPU DONE")

  def get_result(self):
    return self.result_queue.get()
