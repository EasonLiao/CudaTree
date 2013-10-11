#!/usr/bin/python
from setuptools import setup, find_packages, Extension
import cudatree
#from Cython.Distutils import build_ext


#ext_modules = [Extension("cuda_random_decisiontree_small", ["cudatree/cuda_random_decisiontree_small.pyx"])]

setup(
    name = "cudatree",
    long_description='''
CudaTree
========

CudaTree is an implementation of Leo Breiman's Random Forests adapted to run on the GPU. A random forest is an ensemble of randomized decision trees which vote together to predict new labels. CudaTree parallelizes the construction of each individual tree in the ensemble and thus is able to train faster than the latest version of scikits-learn.

### Usage ###

  import numpy as np
  from cudatree import load_data, RandomForestClassifier

  x_train, y_train = load_data("digits")
  forest = RandomForestClassifier()
  forest.fit(x_train, y_train, n_trees=50, verbose = True, bootstrap = False)
  forest.predict(x_train)

### Dependencies ###

CudaTree is writen for Python 2.7 and depends on:

 * scikit-learn
 * NumPy
 * PyCUDA
 *  Parakeet
''',
    version = cudatree.__version__,
    description = "Random Forests for the GPU using PyCUDA",
    author = [ "Yisheng Liao", "Alex Rubinsteyn"], 
    author_email = ["yl1912@nyu.edu", "alexr@cs.nyu.edu"],
    packages = find_packages() + ['cudatree.test'],
    package_dir = {'cudatree.test' : './test'},
    package_data = {'cudatree' : ['cuda_kernels/*.cu']},
    #ext_modules = ext_modules,
    #cmdclass = {'build_ext' : build_ext},
    install_requires = [
        'numpy',
        'sklearn',
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
