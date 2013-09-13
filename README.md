CudaTree
==================

CudaTree is an implementation of Leo Breiman's [Random Forests](http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)
adapted to run on the GPU. 
A random forest is an ensemble of randomized decision trees which  vote together to predict new labels.
CudaTree parallelizes the construction of each individual tree in the ensemble and thus is able to train faster than 
the latest version of [scikits-learn](http://scikit-learn.org/stable/modules/tree.html). 

### Usage


```python
  import numpy as np
  from cuda_tree import load_data, RandomForestClassifier

  x_train, y_train = load_data("digits")
  forest = RandomForestClassifier()
  forest.fit(x_train, y_train, n_trees=50, min_samples_split=20)
  forest.predict(x_train)
```

### Dependencies 

CudaTree is writen for Python 2.7 and depends on:

* [Scikit-learn](http://scikit-learn.org/stable/)
* [Numpy](http://www.scipy.org/install.html)
* [PyCUDA](http://documen.tician.de/pycuda/#)
* [Nose](https://nose.readthedocs.org/en/latest/)


### Limitations:
* CudaTree needs GPU memory more than the size of training data, so make sure you have enough GPU memory.
  Specifically:
  
  <b>GPUMem = bytes of training data + 2 * n_features * n_samples * n_bytes_per_idx + n_features * n_samples</b>
  <i>(n_bytes_per_idx is 1 when the number of samples <= 256
  <br />
  n_bytes_per_idx is 2 when the number of samples <= 65536
  <br />
  n_bytes_per_idx is 4 when the number of samples <= 4294967296 
  <br />
  n_bytes_per_idx is 8 when the number of samples > 4294967296)</i>
  <br/>
 
  Let's assume you have a training data which is 200MB, and the n_samples is 10000, n_features is 3000, then the 
  GPU memory you need is:<br />
  Mem = 200MB + (2 * 3000 * 10000 * 2 + 3000 * 10000) / 1024 / 1024 = 314MB

* The maximum number of features allowed is 65536.
* The maximum number of categories allowed is 10000(However, CudaTree performs well only when the number of categories is small(<=100)).
* The compute ability of your GPU should >= 2.0.
* The performance gain is about 1.5X ~ 2X faster than Scikits-learn. The exact number depends on how powerful your GPU is and what your training data looks like. 



### Implementation Details 

Trees are first constructed in depth-first order, with a separate kernel launch for each node's subset of the data. 
Eventually the data gets split into very small subsets and at that point CudaTree switches to breadth-first grouping
of multiple subsets for each kernel launch. 


