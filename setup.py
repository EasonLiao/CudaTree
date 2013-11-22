#!/usr/bin/python
from setuptools import setup, find_packages, Extension
import cudatree

setup(
    name = "cudatree",
    long_description='''
CudaTree
========

CudaTree is an implementation of Leo Breiman's Random Forests adapted to run on the GPU. A random forest is an ensemble of randomized decision trees which vote together to predict new labels. CudaTree parallelizes the construction of each individual tree in the ensemble and thus is able to train faster than the latest version of scikits-learn.

Usage
-------------

::

  import numpy as np
  from cudatree import load_data, RandomForestClassifier
  x_train, y_train = load_data("digits")
  forest = RandomForestClassifier(n_estimators = 50, max_features = 6)
  forest.fit(x_train, y_train)
  forest.predict(x_train)

Dependencies
--------------

CudaTree is writen for Python 2.7 and depends on:

* scikit-learn
* NumPy
* PyCUDA
*  Parakeet
''',
    version = cudatree.__version__,
    description = "Random Forests for the GPU using PyCUDA",
    author =  "Yisheng Liao and Alex Rubinsteyn", 
    author_email = ["yl1912@nyu.edu / alexr@cs.nyu.edu"],
    packages = find_packages() + ['cudatree.test'],
    package_dir = {'cudatree.test' : './test'},
    package_data = {'cudatree' : ['cuda_kernels/*.cu']},
    url = "https://github.com/EasonLiao/CudaTree",
    install_requires = [
        'numpy',
        'scikit-learn',
        'pycuda',
        "parakeet"
      ],
    license="BSD",
    classifiers=['Development Status :: 3 - Alpha',
                  'Topic :: Software Development :: Libraries',
                  'Topic :: Scientific/Engineering', 
                  'License :: OSI Approved :: BSD License',
                  'Intended Audience :: Developers',
                  'Intended Audience :: Science/Research', 
                  'Programming Language :: Python :: 2.7',
                ])
