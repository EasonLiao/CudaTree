#!/usr/bin/python
import sklearn
from sklearn.ensemble import RandomForestClassifier as skRF
import numpy as np
import time
from cudatree import load_data, RandomDecisionTreeSmall, timer
from cudatree import RandomForestClassifier as cdRF
import multiprocessing
from multiprocessing import Value, Lock
import itertools


def parallel_sklearn_job(X, Y, n_estimators, bootstrap, max_features, remain_trees, result_queue, lock):
  n_trees = 4
  estimators = list()
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
    estimators.append(f.estimators_)
  
  result_queue.put(list(itertools.chain(*estimators)))
  print "SK DONE"

class RandomForestClassifier(object):
  def __init__(self, n_estimators = 10, max_features = None, bootstrap = True):
    self.n_estimators = n_estimators
    self.max_features = max_features
    self.bootstrap = bootstrap
    self._sk_estimators = None
    self._cuda_forest = None


  def _cuda_fit(self, X, Y, bfs_threshold, remain_trees, lock):
    self._cuda_forest = cdRF(n_estimators = self.n_estimators, bootstrap = self.bootstrap, 
        max_features = self.max_features) 
    
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

    self._cuda_forest.fit_relase()
    print "CUDA DONE"


  def fit(self, X, Y, bfs_threshold = None):
    remain_trees = Value("i", self.n_estimators)
    result_queue = multiprocessing.Queue(100)
    lock = Lock()
    
    #Start a new process to do sklearn random forest
    p = multiprocessing.Process(target = parallel_sklearn_job, args = (X, Y, self.n_estimators, 
      self.bootstrap, self.max_features, remain_trees, result_queue, lock))
    p.start() 

    #At same time, we construct cuda radom forest
    self._cuda_fit(X, Y, bfs_threshold, remain_trees, lock)
    
    #get the result
    self._sk_estimators = result_queue.get()
    print self._sk_estimators
    #print self._cuda_forest.score(X, Y)

  def predict(X):
    pass

x_train, y_train = load_data("kdd")

with timer("hybird version"):
  f = RandomForestClassifier(50, max_features = None, bootstrap = False)
  f.fit(x_train, y_train)
