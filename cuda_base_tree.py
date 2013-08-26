import numpy as np

class BaseTree(object):
  def __init__(self):
    self.root = None
    self.max_depth = None

  def print_tree(self):
    def recursive_print(node):
      if node.left_child and node.right_child:
        print node
        recursive_print(node.left_child)
        recursive_print(node.right_child)
      else:
        print node
    assert self.root is not None
    recursive_print(self.root)

  def __predict(self, val):
    temp = self.root
    while True:
      if temp.left_child and temp.right_child:
        if val[temp.feature_index] < temp.feature_threshold:
          temp = temp.left_child
        else:
          temp = temp.right_child
      else:
          return temp.value

  def predict(self, inputs):
    res = []
    for val in inputs:
      res.append(self.__predict(val))
    return np.array(res)


