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
  from cudatree import load_data, RandomForestClassifier

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

It's important to remember that a dataset which fits into your computer's main memory may not necessarily fit on a GPU's smaller memory. 
Furthermore, CudaTree uses several temporary arrays during tree construction which will limit how much space is available. 
A formula for the total number of bytes required to fit a decision tree for a given dataset is given below. If less than this quantity is available 
on your GPU, then CudaTree will fail. 




<!-- 
\mathrm{GPU}\;\mathrm{memory}\;\mathrm{in}\;\mathrm{bytes} = \mathit{DatasetSize} + 2\cdot \mathit{Samples} \cdot \mathit{Features} \cdot \left\lceil \frac{\log_2 \mathit{Samples}}{8} \right\rceil + \mathit{Features} \cdot \mathit{Samples}
-->
  <div align="center">
  ![gpu memory = dataset + 2*samples*features*ceil(log2(samples)/8) + samples*features](https://raw.github.com/EasonLiao/CudaTree/master/doc/gpumem.png) 
  </div>

<!--     
  <i>(n_bytes_per_idx is 1 when the number of samples <= 256
  <br />
  n_bytes_per_idx is 2 when the number of samples <= 65536
  <br />
  n_bytes_per_idx is 4 when the number of samples <= 4294967296 
  <br />
  n_bytes_per_idx is 8 when the number of samples > 4294967296)</i>
  <br/>
 --> 
 
  For example, let's assume you have a training dataset which takes up 200MB, and the number of samples = 10000 and 
  the number of features is 3000, then the total GPU memory required will be: <br>
  <div align="center" style="font-style:italic;">
  200MB + (2 * 3000 * 10000 * 2 + 3000 * 10000) / 1024 / 1024 = 314MB
  </div>

In addition to memory requirement, there are several other limitations hard-coded into CudaTree: 

* The maximum number of features allowed is 65,536.
* The maximum number of categories allowed is 10000 (CudaTree performs best when the number of categories is <=100).
* Your NVIDIA GPU must have compute capability >= 2.0.
* Currently, the only splitting criterion is GINI impurity, which means CudaTree can't yet do regression (splitting by variance for continuous outputs is planned)

The performance gain over scikits-learn is typically about 1.5X ~ 2X, though the exact number depends on how powerful your GPU is and what your training data looks like. 



### Implementation Details 

Trees are first constructed in depth-first order, with a separate kernel launch for each node's subset of the data. 
Eventually the data gets split into very small subsets and at that point CudaTree switches to breadth-first grouping
of multiple subsets for each kernel launch. 


