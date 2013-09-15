import numpy as np
from cudatree import load_data, RandomForestClassifier, timer
from cudatree import util

x_train, y_train = load_data("iris")
x_test, y_test = load_data("iris")

def test_iris():
  with timer("Cuda treelearn"):
    forest = RandomForestClassifier()
    forest.fit(x_train, y_train, n_trees=10, max_features = 4)
  with timer("Predict"):
    diff, total = util.test_diff(forest.predict(x_test), y_test)  
    print "%s(Wrong)/%s(Total). The error rate is %f." % (diff, total, diff/float(total))
  assert diff == 0

if __name__ == "__main__":
  test_iris()
