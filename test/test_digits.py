import numpy as np
from cudatree import load_data, RandomForestClassifier, timer
from cudatree import util

x,y = load_data("digits")

n_estimators = 13 
bootstrap = True

def test_digits_memorize():
  with timer("Cuda treelearn"):
    forest = RandomForestClassifier(n_estimators = n_estimators/2, bootstrap = False)
    forest.fit(x, y)
  with timer("Predict"):
    diff, total = util.test_diff(forest.predict(x), y)  
    print "%s (Wrong) / %s (Total). The error rate is %f." % (diff, total, diff/float(total))
  assert diff == 0, "Didn't memorize, got %d wrong" % diff 

from helpers import compare_accuracy 
def test_digits_vs_sklearn():
  compare_accuracy(x,y)

if __name__ == "__main__":
  test_digits_memorize()
  test_digits_vs_sklearn()
