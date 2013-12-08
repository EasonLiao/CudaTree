#!/usr/bin/python
from cudatree import RandomForestClassifier, load_data, timer
from cudatree import util
from hybridforest import RandomForestClassifier as hybridForest
import numpy as np
import math
#from PyWiseRF import WiseRF

debug = False
verbose = False
bootstrap = False
n_estimators = 100

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
  throw_away = hybridForest(n_estimators = 2, bootstrap = bootstrap,  
        max_features = None)
  throw_away.fit(x_train, y_train, bfs_threshold = bfs_threshold)

  with timer("%s benchmark hybrid (bfs_threshold = %s)" % (dataset, bfs_threshold)): 
    forest = hybridForest(n_estimators = n_estimators, bootstrap = bootstrap, n_jobs = 2,
        max_features = None)
    forest.fit(x_train, y_train, bfs_threshold = bfs_threshold)
  forest = None

benchmark_hybrid("covtype", None)
#benchmark_cuda("pamap", None)
#benchmark_cuda("cf100", 10000)
#benchmark_cuda("inet", 1000)
#benchmark_hybrid("inet", 5000)
benchmark_hybrid("poker")
#benchmark_hybrid("inet")

#benchmark_hybrid("cf100")
benchmark_hybrid("cf100")
#benchmark_hybrid("covtype")
#benchmark_hybrid("poker")
#benchmark_hybrid("inet")

"""
benchmark_hybrid("cf100")
benchmark_hybrid("kdd")
benchmark_hybrid("covtype")
benchmark_hybrid("cf10")
benchmark_cuda("cf100", True)
benchmark_cuda("kdd", True)
benchmark_cuda("covtype", True)
benchmark_cuda("cf10", True)
"""
