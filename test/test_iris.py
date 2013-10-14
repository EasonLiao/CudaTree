import numpy as np
from cudatree import load_data, RandomForestClassifier, timer
from cudatree import util

x, y = load_data("iris")

def test_iris_memorize():
  with timer("Cuda treelearn"):
    forest = RandomForestClassifier(bootstrap = False)
    forest.fit(x, y)
  with timer("Predict"):
    diff, total = util.test_diff(forest.predict(x), y)  
    print "%s(Wrong)/%s(Total). The error rate is %f." % (diff, total, diff/float(total))
  assert diff == 0, "Didn't perfectly memorize, got %d wrong" % diff

from helpers import compare_accuracy
def test_iris_accuracy():
  compare_accuracy(x,y)


if __name__ == "__main__":
  test_iris_memorize()
  test_iris_accuracy()
  
