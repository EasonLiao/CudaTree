#!/usr/bin/python
import sklearn
from sklearn.ensemble import RandomForestClassifier as skRF
import numpy as np
from cudatree import load_data, RandomClassifierTree, timer, convert_result
from cudatree import RandomForestClassifier as cdRF
import multiprocessing
from multiprocessing import Value, Lock, cpu_count
import atexit
from cudatree import util

def cpu_build(cpu_classifier, 
              X, 
              Y, 
              n_estimators, 
              bootstrap, 
              max_features, 
              n_jobs, 
              remain_trees, 
              result_queue, lock):
  """
  Build some trees on cpu, the cpu classifier should be cpu 
  implementation of random forest classifier.
  """
  forests = list()
  if max_features == None:
    max_features = "auto"
  
  Y = Y.astype(np.uint16) 
  classifier_name = cpu_classifier.__name__

  while True:
    lock.acquire()
    if remain_trees.value < 2 * n_jobs:
      lock.release()
      break

    remain_trees.value -= n_jobs
    lock.release()

    util.log_info("%s got %s jobs.", classifier_name, n_jobs)
    f = cpu_classifier(n_estimators = n_jobs, n_jobs = n_jobs, 
        bootstrap = bootstrap, max_features = max_features)
    f.fit(X, Y)
    forests.append(f)

  result_queue.put(forests)
  util.log_info("%s's job done", classifier_name)


#kill the child process if any
def cleanup(proc):
  if proc.is_alive():
    proc.terminate()


class RandomForestClassifier(object):
  """
  This RandomForestClassifier uses both CudaTree and cpu 
  implementation of RandomForestClassifier(default is sklearn) 
  to construct random forest. The reason is that CudaTree only 
  use one CPU core, the main computation is done at GPU side, 
  so in order to get maximum utilization of the system, we can 
  train one CudaTree random forest with GPU and one core of CPU,
  and simultaneously we construct some trees on other cores by 
  other multicore implementaion of random forest.
  """
  def __init__(self, 
              n_estimators = 10, 
              n_jobs = -1, 
              max_features = None, 
              bootstrap = True, 
              cpu_classifier = skRF):
    """Construce random forest on GPU and multicores.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    max_features : int or None, optional (default="log2(n_features)")
        The number of features to consider when looking for the best split:
          - If None, then `max_features=log2(n_features)`.
    
    bootstrap : boolean, optional (default=True)
        Whether use bootstrap samples
    
    n_jobs : int (default=-1)
        How many cores to use when construct random forest.
          - If -1, them use number of cores you CPU has.
    
    cpu_classifier : class(default=sklearn.ensemble.RandomForestClassifier)
        Which random forest classifier class to use when construct trees on CPU.
          The default is sklearn.ensemble.RandomForestClassifier. You can also pass 
          some other classes like WiseRF.

    Returns
    -------
    None
    """ 
    assert hasattr(cpu_classifier, "fit"),\
              "cpu classifier must support fit method."
    assert hasattr(cpu_classifier, "predict_proba"),\
              "cpu classifier must support predict proba method."
    
    self.n_estimators = n_estimators
    self.max_features = max_features
    self.bootstrap = bootstrap
    self._cpu_forests = None
    self._cuda_forest = None
    self._cpu_classifier = cpu_classifier
    
    if n_jobs == -1:
      n_jobs = cpu_count()
    self.n_jobs = n_jobs


  def _cuda_fit(self, X, Y, bfs_threshold, remain_trees, lock):
    self._cuda_forest = cdRF(n_estimators = self.n_estimators,
                            bootstrap = self.bootstrap, 
                            max_features = self.max_features) 
    #allocate resource
    self._cuda_forest.fit_init(X, Y)
    f = self._cuda_forest

    while True:
      lock.acquire()
      if remain_trees.value == 0:
        lock.release()
        break
      
      remain_trees.value -= 1
      lock.release()
      
      #util.log_info("Cudatree got 1 job.")
      tree = RandomClassifierTree(f)   
      
      si, n_samples = f._get_sorted_indices(f.sorted_indices)
      tree.fit(f.samples, f.target, si, n_samples)
      f.forest.append(tree) 
    
    #release the resource
    self._cuda_forest.fit_release()
    #util.log_info("cudatee's job done")


  def fit(self, X, Y, bfs_threshold = None):
    #shared memory value which tells two processes when should stop
    remain_trees = Value("i", self.n_estimators)
    result_queue = multiprocessing.Queue(100)
    lock = Lock()
    #how many labels    
    self.n_classes = np.unique(Y).size

    #Start a new process to do sklearn random forest
    p = multiprocessing.Process(target = cpu_build, 
                                args = (self._cpu_classifier,
                                          X, 
                                          Y, 
                                          self.n_estimators, 
                                          self.bootstrap, 
                                          self.max_features, 
                                          self.n_jobs - 1, 
                                          remain_trees, 
                                          result_queue, 
                                          lock)
                                )
    
    #kill the child process when program aborts
    atexit.register(cleanup, p)
    
    #set daemon to false to enable child process to spawn new processes
    p.daemon = False
    p.start() 

    #At same time, we construct cuda radom forest
    self._cuda_fit(X, Y, bfs_threshold, remain_trees, lock)    
    #get the result
    self._cpu_forests = result_queue.get()
    p.join()


  def predict(self, X):
    sk_proba = np.zeros((X.shape[0], self.n_classes), np.float64)

    if self._cpu_forests is not None:
      for f in self._cpu_forests:
        sk_proba += f.predict_proba(X) * len(f.estimators_)
     
    n_sk_trees = self.n_estimators - len(self._cuda_forest.forest)
    n_cd_trees = self.n_estimators - n_sk_trees
    cuda_proba = self._cuda_forest.predict_proba(X) * n_cd_trees
    final_proba = (sk_proba  + cuda_proba ) / self.n_estimators
    res = np.array([np.argmax(final_proba[i]) for i in xrange(final_proba.shape[0])])
    
    if hasattr(self._cuda_forest, "compt_table"):
      res = convert_result(self._cuda_forest.compt_table, res)
    return res

  def score(self, X, Y):
    return np.mean(self.predict(X) == Y)
