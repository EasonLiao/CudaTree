import numpy as np
from cuda_random_decisiontree_small import RandomDecisionTreeSmall
from datasource import load_data

class RandomForest(object):
  def fit(self, x, target, n_trees = 20, max_features = 10):
    self.forest = [RandomDecisionTreeSmall() for i in xrange(n_trees)]
    for tree in self.forest:
      tree.fit(x_train, y_train, max_features)
  
  def predict(self, x):
    res = []
    for tree in self.forest:
      res.append(tree.predict(x))
    res = np.array(res)
    return np.array([np.argmax(np.bincount(res[:,i])) for i in xrange(res.shape[1])])

if __name__ == "__main__":
  x_train, y_train = load_data("db")
  x_test, y_test = load_data("db_test")
  
  ft = RandomForest()
  ft.fit(x_train, y_train)
  print ft.predict(x_test)[0:20]
  print y_test[0:20]

  """
  t = RandomDecisionTreeSmall()
  t.fit(x_train, y_train, 100)
  print t.predict(x_test)[0:20]
  print y_test[0:20]
  """

  #t.print_tree()
  #ft = RandomForest() 
  
