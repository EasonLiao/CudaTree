#!/usr/bin/python
import sklearn
from sklearn.ensemble import RandomForestClassifier as skRF
import numpy as np
import time
from cudatree import load_data, RandomDecisionTreeSmall, timer, convert_result
from cudatree import RandomForestClassifier as cdRF
import multiprocessing
from multiprocessing import Value, Lock
import itertools
from PyWiseRF import WiseRF

def sklearn_build(X, Y, n_estimators, bootstrap, max_features, remain_trees, result_queue, lock):
  n_trees = 5
  forests = list()

  if max_features == None:
    max_features = "auto"
  
  while True:
    lock.acquire()
    if remain_trees.value == 0:
      lock.release()
      break

    if remain_trees.value < n_trees:
      n_trees = remain_trees.value

    remain_trees.value -= n_trees
    lock.release()

    print "sklearn : ", n_trees 
    f = skRF(n_estimators = n_trees, n_jobs = n_trees, bootstrap = bootstrap, max_features = max_features)
    f.fit(X, Y)
    forests.append(f)

  result_queue.put(forests)
  print "SK DONE"


class RandomForestClassifier(object):
  def __init__(self, n_estimators = 10, max_features = None, bootstrap = True):
    self.n_estimators = n_estimators
    self.max_features = max_features
    self.bootstrap = bootstrap
    self._sk_forests = None
    self._cuda_forest = None


  def _cuda_fit(self, X, Y, bfs_threshold, remain_trees, lock):
    self._cuda_forest = cdRF(n_estimators = self.n_estimators, bootstrap = self.bootstrap, 
        max_features = self.max_features) 
    
    #allocate resource
    self._cuda_forest.fit_init(X, Y)
    f = self._cuda_forest

    if bfs_threshold == None:
      bfs_threshold = 5000
    
    while True:
      lock.acquire()
      if remain_trees.value == 0:
        lock.release()
        break
      
      remain_trees.value -= 1
      lock.release()
      
      tree = RandomDecisionTreeSmall(f.samples_gpu, f.labels_gpu, f.compt_table, f.dtype_labels, 
          f.dtype_samples, f.dtype_indices, f.dtype_counts, f.n_features, f.stride, 
          f.n_labels, f.COMPT_THREADS_PER_BLOCK, f.RESHUFFLE_THREADS_PER_BLOCK, 
          f.max_features, f.min_samples_split, bfs_threshold, f.debug, f)   
      
      print remain_trees.value
      si, n_samples = f._get_sorted_indices(f.sorted_indices)
      tree.fit(f.samples, f.target, si, n_samples)
      f.forest.append(tree) 
    
    #release the resource
    self._cuda_forest.fit_release()
    print "CUDA DONE"


  def fit(self, X, Y, bfs_threshold = None):
    #shared memory value which tells two processes when should stop
    remain_trees = Value("i", self.n_estimators)
    result_queue = multiprocessing.Queue(100)
    lock = Lock()
    
    self.n_classes = np.unique(Y).size

    #Start a new process to do sklearn random forest
    p = multiprocessing.Process(target = sklearn_build, args = (X, Y, self.n_estimators, 
      self.bootstrap, self.max_features, remain_trees, result_queue, lock))
    
    p.daemon = False
    p.start() 

    #At same time, we construct cuda radom forest
    self._cuda_fit(X, Y, bfs_threshold, remain_trees, lock)
    
    #get the result
    self._sk_forests = result_queue.get()


  def predict(self, X):
    sk_proba = np.zeros((X.shape[0], self.n_classes), np.float64)

    if self._sk_forests is not None:
      for f in self._sk_forests:
        sk_proba += f.predict_proba(X) * len(f.estimators_)
     
    n_sk_trees = self.n_estimators - len(self._cuda_forest.forest)
    n_cd_trees = self.n_estimators - n_sk_trees
    

    cuda_proba = self._cuda_forest.predict_proba(X) * n_cd_trees

    final_proba = (sk_proba  + cuda_proba ) / self.n_estimators
    res = np.array([np.argmax(final_proba[i]) for i in xrange(final_proba.shape[0])])
    
    if hasattr(self._cuda_forest, "compt_table"):
      res = convert_result(self._cuda_forest.compt_table, res)
    return res

  def score(self, X, Y):
    return np.mean(self.predict(X) == Y)


if __name__ == "__main__":
  x_train, y_train = load_data("inet")
  x_test = x_train[0:100]
  y_test = y_train[0:100]
  x_test, y_test = load_data("inet_test")

  with timer("hybird version"):
    f = RandomForestClassifier(50, max_features = None, bootstrap = False)
    f.fit(x_train, y_train)
  
  print "before prediction"
  print f.score(x_test, y_test)

