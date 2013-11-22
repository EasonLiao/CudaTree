import numpy as np
from random_tree import RandomClassifierTree
from util import timer, get_best_dtype, dtype_to_ctype, mk_kernel, mk_tex_kernel, compile_module
from pycuda import gpuarray
from pycuda import driver
from util import start_timer, end_timer, show_timings
from parakeet import jit
import math

@jit
def convert_result(tran_table, res):
    return np.array([tran_table[i] for i in res])


class RandomForestClassifier(object):
  """A random forest classifier.

    A random forest is a meta estimator that fits a number of classifical
    decision trees on various sub-samples of the dataset and use averaging
    to improve the predictive accuracy and control over-fitting.
    
    Usage:
    See RandomForestClassifier.fit
  """ 
  COMPUTE_THREADS_PER_BLOCK = 128
  RESHUFFLE_THREADS_PER_BLOCK = 256
  BFS_THREADS = 64
  MAX_BLOCK_PER_FEATURE = 50
  MAX_BLOCK_BFS = 10000
  
  def __init__(self, 
              n_estimators = 10, 
              max_features = None, 
              min_samples_split = 1, 
              bootstrap = True, 
              verbose = False, 
              debug = False):
    """Construce multiple trees in the forest.

    Parameters
    ----------
    n_estimator : integer, optional (default=10)
        The number of trees in the forest.

    max_features : int or None, optional (default="log2(n_features)")
        The number of features to consider when looking for the best split:
          - If None, then `max_features=log2(n_features)`.

    min_samples_split : integer, optional (default=1)
        The minimum number of samples required to split an internal node.
        Note: this parameter is tree-specific.
    
    bootstrap : boolean, optional (default=True)
        Whether use bootstrap samples
    
    verbose : boolean, optional (default=False) 
        Display the time of each tree construction takes if verbose = True.
    
    Returns
    -------
    None
    """    
    self.max_features = max_features
    self.min_samples_split = min_samples_split
    self.bootstrap = bootstrap
    self.verbose = verbose
    self.n_estimators = n_estimators
    self.debug = debug
    self.forest = list()

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
    self.mark_table.bind_to_texref_ext(tex_ref)

  def _get_sorted_indices(self, sorted_indices):
    """ Generate sorted indices, if bootstrap == False, 
    then the sorted indices is as same as original sorted indices """
    
    sorted_indices_gpu_original = self.sorted_indices_gpu.copy()

    if not self.bootstrap:
      return sorted_indices_gpu_original, sorted_indices.shape[1]
    else:
      sorted_indices_gpu = gpuarray.empty((self.n_features, self.stride), dtype = self.dtype_indices)
      random_sample_idx = np.unique(np.random.randint(
        0, self.stride, size = self.stride)).astype(self.dtype_indices)
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
                sorted_indices_gpu_original.ptr,
                sorted_indices_gpu.ptr,
                self.stride)
      
      return sorted_indices_gpu, n_samples 
  
  def _allocate_arrays(self):
    #allocate gpu arrays and numpy arrays.
    if self.max_features < 4:
      imp_size = 4
    else:
      imp_size = self.max_features
    
    #allocate gpu arrays
    self.impurity_left = gpuarray.empty(imp_size, dtype = np.float32)
    self.impurity_right = gpuarray.empty(self.max_features, dtype = np.float32)
    self.min_split = gpuarray.empty(self.max_features, dtype = self.dtype_counts)
    self.label_total = gpuarray.empty(self.n_labels, self.dtype_indices)  
    self.label_total_2d = gpuarray.zeros(self.max_features * (self.MAX_BLOCK_PER_FEATURE + 1) * self.n_labels, 
        self.dtype_indices)
    self.impurity_2d = gpuarray.empty(self.max_features * self.MAX_BLOCK_PER_FEATURE * 2, np.float32)
    self.min_split_2d = gpuarray.empty(self.max_features * self.MAX_BLOCK_PER_FEATURE, self.dtype_counts)
    self.features_array_gpu = gpuarray.empty(self.n_features, np.uint16)
    self.mark_table = gpuarray.empty(self.stride, np.uint8) 

    #allocate numpy arrays
    self.idx_array = np.zeros(2 * self.n_samples, dtype = np.uint32)
    self.si_idx_array = np.zeros(self.n_samples, dtype = np.uint8)
    self.nid_array = np.zeros(self.n_samples, dtype = np.uint32)
    self.values_idx_array = np.zeros(2 * self.n_samples, dtype = self.dtype_indices)
    self.values_si_idx_array = np.zeros(2 * self.n_samples, dtype = np.uint8)
    self.threshold_value_idx = np.zeros(2, self.dtype_indices)
    self.min_imp_info = driver.pagelocked_zeros(4, dtype = np.float32)  
    self.features_array = driver.pagelocked_zeros(self.n_features, dtype = np.uint16)
    self.features_array[:] = np.arange(self.n_features, dtype = np.uint16)


  def _release_arrays(self):
    #relase gpu arrays
    self.impurity_left = None
    self.impurity_right = None
    self.min_split = None
    self.label_total = None
    #self.sorted_indices_gpu = None
    #self.sorted_indices_gpu_ = None
    self.label_total_2d = None
    self.min_split_2d = None
    self.impurity_2d = None
    self.feature_mask = None
    self.features_array_gpu = None
    
    #Release kernels
    self.fill_kernel = None
    self.scan_reshuffle_tex = None 
    self.scan_total_kernel = None
    self.comput_label_loop_rand_kernel = None
    self.find_min_kernel = None
    self.scan_total_bfs = None
    self.comput_bfs = None
    self.fill_bfs = None
    self.reshuffle_bfs = None
    self.reduce_bfs_2d = None
    self.comput_bfs_2d = None
    #self.predict_kernel = None
    self.get_thresholds = None
    self.scan_reduce = None
    self.mark_table = None
    
    #Release numpy arrays
    self.idx_array = None
    self.si_idx_array = None
    self.nid_array = None
    self.values_idx_array = None
    self.values_si_idx_array = None
    self.threshold_value_idx = None
    self.min_imp_info = None
    self.features_array = None


  def fit_init(self, samples, target):
    assert isinstance(samples, np.ndarray)
    assert isinstance(target, np.ndarray)
    assert samples.size / samples[0].size == target.size
    target = target.copy()
    self.__compact_labels(target)
    
    self.n_samples = len(target)
    self.n_labels = self.compt_table.size 
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
    
    if self.COMPUTE_THREADS_PER_BLOCK > self.stride:
      self.COMPUTE_THREADS_PER_BLOCK = 32
    if self.RESHUFFLE_THREADS_PER_BLOCK > self.stride:
      self.RESHUFFLE_THREADS_PER_BLOCK = 32
    
    samples_gpu = gpuarray.to_gpu(samples)
    labels_gpu = gpuarray.to_gpu(target) 
    
    sorted_indices = np.argsort(samples).astype(self.dtype_indices)
    self.sorted_indices_gpu = gpuarray.to_gpu(sorted_indices)
      
    if self.max_features is None:
      self.max_features = int(math.ceil(np.sqrt(self.n_features)))
    
    self._allocate_arrays()
    self.__compile_kernels()
       
    if self.bootstrap:
      self.__init_bootstrap_kernel()
    
    #get default best bfs threshold
    self.bfs_threshold = self._get_best_bfs_threshold(self.n_labels, self.n_samples, self.max_features)
    self.sorted_indices = sorted_indices
    self.target = target
    self.samples = samples
    self.samples_gpu = samples_gpu
    self.labels_gpu = labels_gpu
    assert self.max_features > 0 and self.max_features <= self.n_features
    

  def fit_release(self):
    self.target = None
    self.samples = None
    self.samples_gpu = None
    self.labels_gpu = None
    self.sorted_indices_gpu = None
    self.sorted_indices = None
    self._release_arrays()
     
  def _get_best_bfs_threshold(self, n_labels, n_samples, max_features):
    # coefficients estimated by regression over best thresholds for randomly generated data sets 
    # estimate from GTX 580:
    bfs_threshold = int(3702 + 1.58 * n_labels + 0.05766 * n_samples + 21.84 * self.max_features)
    # estimate from Titan: 
    bfs_threshold = int(4746 + 4 * n_labels + 0.0651 * n_samples - 75 * max_features)
    # don't let it grow too big
    bfs_threshold = min(bfs_threshold, 50000)
    # ...or too small
    bfs_threshold = max(bfs_threshold, 2000)
    #bfs_threshold = max(bfs_threshold, 2000)
    return bfs_threshold 


  def fit(self, samples, target, bfs_threshold = None):
    """Construce multiple trees in the forest.

    Parameters
    ----------
    samples:numpy.array of shape = [n_samples, n_features]
            The training input samples.

    target: numpy.array of shape = [n_samples]
            The training input labels.
    
    bfs_threshold: integer, optional (default= n_samples / 40)
            The n_samples threshold of changing to bfs
    
    Returns
    -------
    self : object
      Returns self
    """
    self.fit_init(samples, target)
    
    if bfs_threshold is not None: 
      self.bfs_threshold = bfs_threshold
    
    if self.verbose: 
      print "bsf_threadshold : %d; bootstrap : %r; min_samples_split : %d" % (bfs_threshold, 
          self.bootstrap,  self.min_samples_split)
      print "n_samples : %d; n_features : %d; n_labels : %d; max_features : %d" % (self.stride, 
          self.n_features, self.n_labels, self.max_features)

    self.forest = [RandomClassifierTree(self) for i in xrange(self.n_estimators)]   
   
    for i, tree in enumerate(self.forest):
      si, n_samples = self._get_sorted_indices(self.sorted_indices)

      if self.verbose: 
        with timer("Tree %s" % (i,)):
          tree.fit(self.samples, self.target, si, n_samples)   
        print ""
      else:
        tree.fit(self.samples, self.target, si, n_samples)   
    
    self.fit_release()
    return self


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
    res = np.ndarray((len(self.forest), x.shape[0]), dtype = self.dtype_labels)

    for i, tree in enumerate(self.forest):
      res[i] =  tree.gpu_predict(x)

    res =  np.array([np.argmax(np.bincount(res[:,i])) for i in xrange(res.shape[1])]) 
    if hasattr(self, "compt_table"):
      res = convert_result(self.compt_table, res) 

    return res

  def predict_proba(self, x):
    x = np.require(x.copy(), requirements = "C")
    res = np.ndarray((len(self.forest), x.shape[0]), dtype = self.dtype_labels)
    res_proba = np.ndarray((x.shape[0], self.n_labels), np.float64)
    
    for i, tree in enumerate(self.forest):
      res[i] =  tree.gpu_predict(x)
    
    for i in xrange(x.shape[0]):
      tmp_res = np.bincount(res[:, i])
      tmp_res.resize(self.n_labels)
      res_proba[i] = tmp_res.astype(np.float64) / len(self.forest)

    return res_proba


  def score(self, X, Y):
    return np.mean(self.predict(X) == Y) 


  def __compile_kernels(self):
    ctype_indices = dtype_to_ctype(self.dtype_indices)
    ctype_labels = dtype_to_ctype(self.dtype_labels)
    ctype_counts = dtype_to_ctype(self.dtype_counts)
    ctype_samples = dtype_to_ctype(self.dtype_samples)
    n_labels = self.n_labels
    n_threads = self.COMPUTE_THREADS_PER_BLOCK
    n_shf_threads = self.RESHUFFLE_THREADS_PER_BLOCK
    
    """ DFS module """
    dfs_module = compile_module("dfs_module.cu", (n_threads, n_shf_threads, n_labels, 
      ctype_samples, ctype_labels, ctype_counts, ctype_indices, self.MAX_BLOCK_PER_FEATURE, 
      self.debug))
    
    const_stride = dfs_module.get_global("stride")[0]
    driver.memcpy_htod(const_stride, np.uint32(self.stride))

    self.find_min_kernel = dfs_module.get_function("find_min_imp")
    self.find_min_kernel.prepare("PPPi")
  
    self.fill_kernel = dfs_module.get_function("fill_table")
    self.fill_kernel.prepare("PiiP")
    
    self.scan_reshuffle_tex = dfs_module.get_function("scan_reshuffle")
    self.scan_reshuffle_tex.prepare("PPii")
    tex_ref = dfs_module.get_texref("tex_mark")
    self.mark_table.bind_to_texref_ext(tex_ref) 
      
    self.comput_total_2d = dfs_module.get_function("compute_2d")
    self.comput_total_2d.prepare("PPPPPPPii")

    self.reduce_2d = dfs_module.get_function("reduce_2d")
    self.reduce_2d.prepare("PPPPPi")
    
    self.scan_total_2d = dfs_module.get_function("scan_gini_large")
    self.scan_total_2d.prepare("PPPPii")
    
    self.scan_reduce = dfs_module.get_function("scan_reduce")
    self.scan_reduce.prepare("Pi")

    """ BFS module """
    bfs_module = compile_module("bfs_module.cu", (self.BFS_THREADS, n_labels, ctype_samples,
      ctype_labels, ctype_counts, ctype_indices,  self.debug))

    const_stride = bfs_module.get_global("stride")[0]
    const_n_features = bfs_module.get_global("n_features")[0]
    const_max_features = bfs_module.get_global("max_features")[0]
    driver.memcpy_htod(const_stride, np.uint32(self.stride))
    driver.memcpy_htod(const_n_features, np.uint16(self.n_features))
    driver.memcpy_htod(const_max_features, np.uint16(self.max_features))

    self.scan_total_bfs = bfs_module.get_function("scan_bfs")
    self.scan_total_bfs.prepare("PPPP")

    self.comput_bfs_2d = bfs_module.get_function("compute_2d")
    self.comput_bfs_2d.prepare("PPPPPPPPP")

    self.fill_bfs = bfs_module.get_function("fill_table")
    self.fill_bfs.prepare("PPPPP")

    self.reshuffle_bfs = bfs_module.get_function("scan_reshuffle")
    tex_ref = bfs_module.get_texref("tex_mark")
    self.mark_table.bind_to_texref_ext(tex_ref) 
    self.reshuffle_bfs.prepare("PPP") 

    self.reduce_bfs_2d = bfs_module.get_function("reduce")
    self.reduce_bfs_2d.prepare("PPPPPPi")
    
    self.get_thresholds = bfs_module.get_function("get_thresholds")
    self.get_thresholds.prepare("PPPPP")
   
    self.predict_kernel = mk_kernel(
        params = (ctype_indices, ctype_samples, ctype_labels), 
        func_name = "predict", 
        kernel_file = "predict.cu", 
        prepare_args = "PPPPPPPii")
  
    self.bfs_module = bfs_module
    self.dfs_module = dfs_module
