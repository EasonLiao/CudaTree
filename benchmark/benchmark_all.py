#!/usr/bin/python
from cudatree import RandomForestClassifier, load_data, timer
from cudatree import util
from hybridforest import RandomForestClassifier as hybridForest
import numpy as np

global_bfs_threshold = 2000
debug = False
verbose = False
bootstrap = False

def best_threshold_prct(n_classes, n_samples, n_features):
  return np.exp(7.61724796 - 0.02185684 * np.log2(n_classes)  - 1.09395467 * np.log10(n_samples) - 0.09912969 * np.log10(n_features)) / 100

def benchmark_cuda(dataset, auto_best_threshold = False):
  x_train, y_train = load_data(dataset)

  if auto_best_threshold:
    print np.unique(y_train).size, y_train.size, x_train.shape[1]
    bfs_threshold = best_threshold_prct(np.unique(y_train).size, y_train.size, x_train.shape[1])
    bfs_threshold = int(bfs_threshold * y_train.size)
  else: 
    bfs_threshold = global_bfs_threshold

  print "bfs_threshold : ", bfs_threshold
  
  #Just use this forest to compile the code.
  throw_away = RandomForestClassifier(n_estimators = 1, bootstrap = bootstrap, verbose = False, 
        max_features = None, debug = debug)
  throw_away.fit(x_train, y_train, bfs_threshold = bfs_threshold)

  with timer("%s benchmark cuda" % (dataset,)): 
    forest = RandomForestClassifier(n_estimators = 50, bootstrap = bootstrap, verbose = verbose, 
        max_features = None, debug = debug)
    forest.fit(x_train, y_train, bfs_threshold = bfs_threshold)


def benchmark_hybrid(dataset, auto_best_threshold = False):
  x_train, y_train = load_data(dataset)
  
  if auto_best_threshold:
    print np.unique(y_train).size, y_train.size, x_train.shape[1]
    bfs_threshold = best_threshold_prct(np.unique(y_train).size, y_train.size, x_train.shape[1])
    bfs_threshold = int(bfs_threshold * y_train.size)
  else: 
    bfs_threshold = global_bfs_threshold

  print "bfs_threshold : ", bfs_threshold
  
  #Just use this forest to compile the code.
  throw_away = hybridForest(n_estimators = 1, bootstrap = bootstrap,  
        max_features = None)
  throw_away.fit(x_train, y_train, bfs_threshold = bfs_threshold)

  with timer("%s benchmark hybrid" % (dataset,)): 
    forest = hybridForest(n_estimators = 50, bootstrap = bootstrap, 
        max_features = None)
    forest.fit(x_train, y_train, bfs_threshold = bfs_threshold)

benchmark_hybrid("cf100")
benchmark_hybrid("kdd")
benchmark_hybrid("covtype")
benchmark_hybrid("cf10")

benchmark_cuda("cf100", True)
benchmark_cuda("kdd", True)
benchmark_cuda("covtype", True)
benchmark_cuda("cf10", True)

benchmark_cuda("cf100")
benchmark_cuda("kdd")
benchmark_cuda("covtype")
benchmark_cuda("cf10")
