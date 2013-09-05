Cuda-Decision-Tree
==================
Construct Decision Tree on Cuda.
It's still in developing.

```python
  import numpy as np
  from treelearn import load_data, RandomForest

  x_train, y_train = load_data("digits")
  forest = RandomForest()
  forest.fit(x_train, y_train)
  forest.predict(x_train)
```

Treelearn is writen for Python 2.7 and depends on:
* [scikit-learn](http://scikit-learn.org/stable/)
* [Numpy](http://www.scipy.org/install.html)
* [PyCUDA](http://documen.tician.de/pycuda/#)

