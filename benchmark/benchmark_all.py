#!/usr/bin/python
from cudatree import RandomForestClassifier, load_data, timer
from cudatree import util
from hybridforest import RandomForestClassifier as hybridForest
import numpy as np
import math
from PyWiseRF import WiseRF

global_bfs_threshold = None
debug = False
verbose = False
bootstrap = False
n_estimators = 100

def best_threshold_prct(n_classes, max_features):
  print n_classes, max_features
  return 6058 + 17 * n_classes - 147 * max_features

def benchmark_cuda(dataset, bfs_threshold = None):
  x_train, y_train = load_data(dataset)
  #Just use this forest to compile the code.
  throw_away = RandomForestClassifier(n_estimators = 1, bootstrap = bootstrap, verbose = False, 
        max_features = None, debug = debug)
  throw_away.fit(x_train, y_train, bfs_threshold = bfs_threshold)

  with timer("%s benchmark cuda (bfs_threshold = %s)" % (dataset, bfs_threshold)): 
    forest = RandomForestClassifier(n_estimators = n_estimators, bootstrap = bootstrap, verbose = verbose, 
        max_features = None, debug = debug)
    forest.fit(x_train, y_train, bfs_threshold = bfs_threshold)
  forest = None


def benchmark_hybrid(dataset, bfs_threshold = None):
  x_train, y_train = load_data(dataset)
  
  #Just use this forest to compile the code.
  throw_away = hybridForest(n_estimators = 1, bootstrap = bootstrap,  
        max_features = None, cpu_classifier = WiseRF)
  throw_away.fit(x_train, y_train, bfs_threshold = bfs_threshold)

  with timer("%s benchmark hybrid (bfs_threshold = %s)" % (dataset, bfs_threshold)): 
    forest = hybridForest(n_estimators = n_estimators, bootstrap = bootstrap, 
        max_features = None)
    forest.fit(x_train, y_train, bfs_threshold = bfs_threshold)
  forest = None

#benchmark_hybrid("pamap", None)
#benchmark_cuda("pamap", None)
#benchmark_cuda("cf100", 10000)
#benchmark_cuda("inet", 1000)
#benchmark_hybrid("inet", 5000)
#benchmark_hybrid("covtype", global_bfs_threshold)
#benchmark_hybrid("kdd", global_bfs_threshold)
#benchmark_hybrid("covtype", global_bfs_threshold)
#benchmark_hybrid("poker", global_bfs_threshold)
"""
global_bfs_threshold = 50000
benchmark_cuda("cf100")
benchmark_cuda("kdd")
benchmark_cuda("covtype")
benchmark_cuda("cf10")
"""
"""
global_bfs_threshold = 10000
benchmark_hybrid("cf100")
benchmark_hybrid("kdd")
benchmark_hybrid("covtype")
benchmark_hybrid("cf10")
benchmark_cuda("cf100", True)
benchmark_cuda("kdd", True)
benchmark_cuda("covtype", True)
benchmark_cuda("cf10", True)
"""
