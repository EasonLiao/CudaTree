from sklearn import tree
from sklearn.datasets import load_iris, load_digits
import numpy as np
import cProfile

class Node(object):
  def __init__(self):
    self.value = None 
    self.error = None
    self.samples = None
    self.feature_threshold = None
    self.feature_index = None
    self.left_child = None
    self.right_child = None
    self.height = None

class SplitInfo(object):
  def __init__(self):
    self.min_impurity = None
    self.impurity_left = None
    self.impurity_right = None
    self.min_threshold = None
    self.boolean_mask_left = None
    self.boolean_mask_right = None

class DecisionTree(object): 
  def __init__(self):
    self.root = None
    self.num_features = None
    self.num_examples = None

  def fit(self, X, Target):
    self.num_features = X[0].size
    self.num_examples = Target.size
    assert isinstance(X, np.ndarray)
    assert isinstance(Target, np.ndarray)
    assert X.size / X[0].size == Target.size
    
    error_rate = self.calc_impurity(Target)
    self.root = self.construct(X, Target, error_rate, 1)

  def calc_impurity(self, Y):
      """ Calculate the impurity of Y """
      values = {}
      impurity = 1.0

      for v in Y:
        if v not in values:
          values[v] = 0.0
        values[v] += 1.0
     
      for k in values:
        impurity -= pow(values[k]/Y.size, 2)
      return impurity

  def find_min_impurity(self, F,  Y):
    Temp = F.copy()
    Temp.sort()

    split_info = SplitInfo()

    for i in range(Y.size - 1):
      Threshold  = (Temp[i] + Temp[i+1]) / 2.0
      boolean_mask_left = (F < Threshold)
      boolean_mask_right = (F >= Threshold)
    
      imp_left = self.calc_impurity(Y[boolean_mask_left])
      imp_right = self.calc_impurity(Y[boolean_mask_right])
      
      if split_info.min_impurity is None or imp_left + imp_right < split_info.min_impurity: 
        split_info.min_impurity = imp_left + imp_right
        split_info.min_threshold = Threshold
        split_info.boolean_mask_left = boolean_mask_left
        split_info.boolean_mask_right = boolean_mask_right
        split_info.impurity_left = imp_left
        split_info.impurity_right = imp_right

    return split_info

  def construct(self, E, Y, ErrRate, Height):    
    assert E.size / self.num_features == Y.size
    def check_terminate():
      if ErrRate == 0:
        return True
      return False

    min_split_info = None
    ret_node = Node()
    ret_node.error = ErrRate
    ret_node.samples = Y.size
    ret_node.height = Height

    print "Height : %s, Samples: %s" % (Height, ret_node.samples)
    

    if check_terminate():
      ret_node.value = Y[0]
      print "Height : ", Height
      return ret_node

    
    for i in range(self.num_features):
      print i
      res = self.find_min_impurity(E[:,i], Y)
      if min_split_info is None or res.min_impurity < min_split_info.min_impurity:
        min_split_info = res
        ret_node.feature_index = i
    
    assert np.all(min_split_info.boolean_mask_left | min_split_info.boolean_mask_right) == True
    assert np.any(min_split_info.boolean_mask_left & min_split_info.boolean_mask_right) == False

    ret_node.left_child = self.construct(E[min_split_info.boolean_mask_left], Y[min_split_info.boolean_mask_left], min_split_info.impurity_left, Height + 1)
    ret_node.right_child = self.construct(E[min_split_info.boolean_mask_right], Y[min_split_info.boolean_mask_right], min_split_info.impurity_right, Height + 1)
    ret_node.feature_threshold = min_split_info.min_threshold
    return ret_node

  def _predict(self, val):
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
      res.append(self._predict(val))
    return np.array(res)

  def print_tree(self):
    def recursive_print(node):
      if node.left_child and node.right_child:
        print "Height : %s,  Feature Index : %s,  Threshold : %s,  Samples : %s" % (node.height, node.feature_index, node.feature_threshold, node.samples)  
        recursive_print(node.left_child)
        recursive_print(node.right_child)
      else:
        print "Leaf Height : %s,  Samples : %s" % (node.height, node.samples)  

    recursive_print(self.root)
    

if __name__ == "__main__":
  d = DecisionTree()
  iris = load_digits()
  
  print iris.data
  print iris.data.shape
  print ""
  print iris.target
  print iris.target.shape
  d.fit(iris.data, iris.target)
  #d.print_tree()
  #res = d.predict(iris.data)
  #print np.allclose(iris.target, res)


