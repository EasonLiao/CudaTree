import numpy as np
from cudatree import load_data, RandomForestClassifier, timer
from cudatree import util

x_train, y_train = load_data("digits")
x_test, y_test = load_data("digits")

def test_digits():
  with timer("Cuda treelearn"):
    forest = RandomForestClassifier(n_estimators = 10, bootstrap = False)
    forest.fit(x_train, y_train)
  with timer("Predict"):
    diff, total = util.test_diff(forest.predict(x_test), y_test)  
    print "%s(Wrong)/%s(Total). The error rate is %f." % (diff, total, diff/float(total))
  assert diff == 0

if __name__ == "__main__":
  test_digits()
