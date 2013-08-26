import numpy as np
from cuda_base_tree import BaseTree
import random

class RandomBaseTree(BaseTree):
  def __init__(self):
    self.max_features = None
    self.n_features = None
    self.dtype_indices = None

  def get_indices(self):
    if self.max_features < self.n_features / 2:
      return np.array(random.sample(xrange(self.n_features), self.max_features), dtype=self.dtype_indices)
    else:
      return np.random.permutation(self.n_features)[:self.max_features].astype(self.dtype_indices)

