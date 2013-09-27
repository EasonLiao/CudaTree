import numpy as np
from cuda_base_tree import BaseTree
import random

class RandomBaseTree(BaseTree):
  def __init__(self):
    self.max_features = None
    self.n_features = None
    self.dtype_indices = None

