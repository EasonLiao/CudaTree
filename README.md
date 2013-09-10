CudaTree
==================
Construct Decision Tree on Cuda.


```python
  import numpy as np
  from cuda_tree import load_data, RandomForest

  x_train, y_train = load_data("digits")
  forest = RandomForest()
  forest.fit(x_train, y_train)
  forest.predict(x_train)
```

CudaTree is writen for Python 2.7 and depends on:
* [Scikit-learn](http://scikit-learn.org/stable/)
* [Numpy](http://www.scipy.org/install.html)
* [PyCUDA](http://documen.tician.de/pycuda/#)
* [Nose](https://nose.readthedocs.org/en/latest/)

