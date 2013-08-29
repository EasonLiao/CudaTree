import numpy as np
from cuda_random_decisiontree_small import RandomDecisionTreeSmall
from datasource import load_data
from util import timer

class RandomForest(object):
  def fit(self, x, target, n_trees = 10, max_features = None):
    self.forest = [RandomDecisionTreeSmall() for i in xrange(n_trees)]
    for i, tree in enumerate(self.forest):
      with timer("Tree %d" % (i,)):
        tree.fit(x_train, y_train, max_features)
  
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
  
  with timer("Cuda predict"):
    pre_res  = ft.predict(x_test)

  diff = pre_res - y_test
  print "diff: %s, total: %s" % (np.count_nonzero(diff), pre_res.size)
 

  """
  t = RandomDecisionTreeSmall()
  t.fit(x_train, y_train, 100)
  print t.predict(x_test)[0:20]
  print y_test[0:20]
  """

  #t.print_tree()
  #ft = RandomForest() 
  
