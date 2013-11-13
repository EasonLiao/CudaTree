#!/usr/bin/python
from cudatree import RandomForestClassifier, load_data, timer
from cudatree import util

bfs_threshold = 1000
debug = False
verbose = False
bootstrap = False

def benchmark(dataset):
  x_train, y_train = load_data(dataset)
  #Just use this forest to compile the code.
  throw_away = RandomForestClassifier(n_estimators = 1, bootstrap = bootstrap, verbose = False, 
        max_features = None, debug = debug)
  throw_away.fit(x_train, y_train, bfs_threshold = bfs_threshold)

  with timer("%s benchmark" % (dataset,)): 
    forest = RandomForestClassifier(n_estimators = 50, bootstrap = bootstrap, verbose = verbose, 
        max_features = None, debug = debug)
    forest.fit(x_train, y_train, bfs_threshold = bfs_threshold)

benchmark("cf100")
benchmark("kdd")
benchmark("covtype")
benchmark("cf10")
benchmark("inet")


