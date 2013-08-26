
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

  def __str__(self):
    if self.left_child and self.right_child:
      return "[NODE] Height: %s, FeatureIndex: %s, Threshold: %.1f, Samples: %d" % (self.height, self.feature_index, self.feature_threshold, self.samples)
    else:
      return "[LEAF] Height: %s, Samples: %s" % (self.height, self.samples)

