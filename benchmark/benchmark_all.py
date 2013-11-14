#!/usr/bin/python
from cudatree import RandomForestClassifier, load_data, timer
from cudatree import util
from hybridforest import RandomForestClassifier as hybridForest

bfs_threshold = 1000
debug = False
verbose = False
bootstrap = False

def benchmark_cuda(dataset):
  x_train, y_train = load_data(dataset)
  #Just use this forest to compile the code.
  throw_away = RandomForestClassifier(n_estimators = 1, bootstrap = bootstrap, verbose = False, 
        max_features = None, debug = debug)
  throw_away.fit(x_train, y_train, bfs_threshold = bfs_threshold)

  with timer("%s benchmark cuda" % (dataset,)): 
    forest = RandomForestClassifier(n_estimators = 50, bootstrap = bootstrap, verbose = verbose, 
        max_features = None, debug = debug)
    forest.fit(x_train, y_train, bfs_threshold = bfs_threshold)

def benchmark_hybrid(dataset):
  x_train, y_train = load_data(dataset)
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

benchmark_cuda("cf100")
benchmark_cuda("kdd")
benchmark_cuda("covtype")
benchmark_cuda("cf10")

