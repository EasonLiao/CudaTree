
class Node(object):
  def __init__(self):
    self.value = None 
    self.feature_threshold = None
    self.feature_index = None
    self.left_child = None
    self.start_idx = None
    self.stop_idx = None
    self.right_child = None
    self.depth = None
    self.nid = None

  def __str__(self):
    if self.left_child and self.right_child:
      return "[NODE] Height: %s, FeatureIndex: %s, Samples: %d" % (self.height, self.feature_index, self.samples)
    else:
      return "[LEAF] Height: %s, Samples: %s" % (self.height, self.samples)



