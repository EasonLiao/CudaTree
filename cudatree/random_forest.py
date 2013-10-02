import numpy as np
from cuda_random_decisiontree_small import RandomDecisionTreeSmall
from util import timer, get_best_dtype, dtype_to_ctype, mk_kernel, mk_tex_kernel
from pycuda import gpuarray
from util import start_timer, end_timer, show_timings

class RandomForestClassifier(object):
  """A random forest classifier.

    A random forest is a meta estimator that fits a number of classifical
    decision trees on various sub-samples of the dataset and use averaging
    to improve the predictive accuracy and control over-fitting.
    
    Usage:
    See RandomForestClassifier.fit
  """ 
  COMPT_THREADS_PER_BLOCK = 128
  RESHUFFLE_THREADS_PER_BLOCK = 256
  
  def __compact_labels(self, target):
    def check_is_compacted(x):
      return x.size == int(np.max(x)) + 1 and int(np.min(x)) == 0
    def convert_to_dict(x):
      d = {}
      for i, val in enumerate(x):
        d[val] = i
      return d

    self.compt_table = np.unique(target)
    self.compt_table.sort()        
    if not check_is_compacted(self.compt_table):
      trans_table = convert_to_dict(self.compt_table)
      for i, val in enumerate(target):
        target[i] = trans_table[val]
   
  def __init_bootstrap_kernel(self):
    """ Compile the kernels and GPUArrays needed to generate the bootstrap samples"""
    ctype_indices = dtype_to_ctype(self.dtype_indices)
    self.bootstrap_fill= mk_kernel((ctype_indices,), "bootstrap_fill",
        "bootstrap_fill.cu")
    self.bootstrap_reshuffle, tex_ref = mk_tex_kernel((ctype_indices, 128), "bootstrap_reshuffle",
        "tex_mark", "bootstrap_reshuffle.cu")
    
    self.bootstrap_fill.prepare("PPii")
    self.bootstrap_reshuffle.prepare("PPPi")
    self.mark_table = gpuarray.empty(self.stride, np.uint8) 
    self.mark_table.bind_to_texref_ext(tex_ref)

  def __get_sorted_indices(self):
    """ Generate sorted indices, if bootstrap == False, then the sorted indices is as same as original sorted indices """
    if not self.bootstrap:
      sorted_indices_gpu = self.sorted_indices_gpu.copy()
      return sorted_indices_gpu, sorted_indices_gpu.shape[1]
    else:
      sorted_indices_gpu = gpuarray.empty((self.n_features, self.stride), dtype = self.dtype_indices)
      random_sample_idx = np.unique(np.random.randint(0, self.stride, size = self.stride)).astype(self.dtype_indices)
      random_sample_idx_gpu = gpuarray.to_gpu(random_sample_idx)
      n_samples = random_sample_idx.size
      
      self.bootstrap_fill.prepared_call(
                (1, 1),
                (512, 1, 1),
                random_sample_idx_gpu.ptr,
                self.mark_table.ptr,
                n_samples,
                self.stride)

      self.bootstrap_reshuffle.prepared_call(
                (self.n_features, 1),
                (128, 1, 1),
                self.mark_table.ptr,
                self.sorted_indices_gpu.ptr,
                sorted_indices_gpu.ptr,
                self.stride)
      
      return sorted_indices_gpu, n_samples 

  def fit(self, samples, target, n_trees = 10, min_samples_split = 1, max_features = None, bfs_threshold = 64, bootstrap = True):
    """Construce multiple trees in the forest.

    Parameters
    ----------
    samples:numpy.array of shape = [n_samples, n_features]
            The training input samples.

    target: numpy.array of shape = [n_samples]
            The training input labels.
    
    n_trees : integer, optional (default=10)
        The number of trees in the forest.

    max_features : int or None, optional (default="log2(n_features)")
        The number of features to consider when looking for the best split:
          - If None, then `max_features=log2(n_features)`.

    min_samples_split : integer, optional (default=2)
        The minimum number of samples required to split an internal node.
        Note: this parameter is tree-specific.
    
    bfs_threshold: integer, optional (default=64)
            The n_samples threshold of changing to bfs
    
    Returns
    -------
    None
    """
    assert isinstance(samples, np.ndarray)
    assert isinstance(target, np.ndarray)
    assert samples.size / samples[0].size == target.size

    target = target.copy()
    self.__compact_labels(target)
    self.n_labels = self.compt_table.size 
    self.bootstrap = bootstrap
    self.dtype_indices = get_best_dtype(target.size)

    if self.dtype_indices == np.dtype(np.uint8):
      self.dtype_indices = np.dtype(np.uint16)

    self.dtype_counts = self.dtype_indices
    self.dtype_labels = get_best_dtype(self.n_labels)
    self.dtype_samples = samples.dtype
   
    samples = np.require(np.transpose(samples), requirements = 'C')
    target = np.require(np.transpose(target), dtype = self.dtype_labels, requirements = 'C') 
    self.n_features = samples.shape[0]
    self.stride = target.size
    
    if self.COMPT_THREADS_PER_BLOCK > self.stride:
      self.COMPT_THREADS_PER_BLOCK = 32
    if self.RESHUFFLE_THREADS_PER_BLOCK > self.stride:
      self.RESHUFFLE_THREADS_PER_BLOCK = 32

    samples_gpu = gpuarray.to_gpu(samples)
    labels_gpu = gpuarray.to_gpu(target) 
    
    sorted_indices = np.empty((self.n_features, self.stride), dtype = self.dtype_indices)
    
    for i,f in enumerate(samples):
      sort_idx = np.argsort(f)
      sorted_indices[i] = sort_idx  
  
    self.sorted_indices_gpu = gpuarray.to_gpu(sorted_indices)
  
    if self.bootstrap:
      self.__init_bootstrap_kernel()

    self.forest = [RandomDecisionTreeSmall(samples_gpu, labels_gpu, self.compt_table, 
      self.dtype_labels,self.dtype_samples, self.dtype_indices, self.dtype_counts,
      self.n_features, self.stride, self.n_labels, self.COMPT_THREADS_PER_BLOCK,
      self.RESHUFFLE_THREADS_PER_BLOCK, max_features, min_samples_split, bfs_threshold) for i in xrange(n_trees)]   
   
    for i, tree in enumerate(self.forest):
      si, n_samples = self.__get_sorted_indices()
      with timer("Tree %s" % (i,)):
        tree.fit(samples, target, si, n_samples)
      
      print ""

    self.sorted_indices = None
    self.mark_table = None

  def predict(self, x):
    """Predict labels for giving samples.

    Parameters
    ----------
    x:numpy.array of shape = [n_samples, n_features]
            The predicting input samples.
    
    Returns
    -------
    y: Array of shape [n_samples].
        The predicted labels.
    """
    x = np.require(x.copy(), requirements = "C")
    res = []
    
    start_timer("predict kernel")
    for tree in self.forest:
      res.append(tree.gpu_predict(x))
    end_timer("predict kernel")

    start_timer("predict loop")
    res = np.array(res)
    res =  np.array([np.argmax(np.bincount(res[:,i])) for i in xrange(res.shape[1])]) 
    end_timer("predict loop")
    
    show_timings()
    return res


