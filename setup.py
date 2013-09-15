#!/usr/bin/python
from setuptools import setup, find_packages
import cudatree

setup(
    name = "cudatree",
    long_decsription='''
CudaTree
========
Building Decision Tree on Cuda.
''',
    version = cudatree.__version__,
    description = "Building decison tree on Cuda",
    author = ["Alex Rubinsteyn", "Yisheng Liao"],
    author_email = ["alexr@cs.nyu.edu", "yl1912@nyu.edu"],
    packages = find_packages() + ['cudatree.test'],
    package_dir = {'cudatree.test' : './test'},
    package_data = {'cudatree' : ['cuda_kernels/*.cu']},
    requires = [
        'numpy',
        'sklearn',
        'pycuda'
      ],
    classifiers=['Development Status :: 3 - Alpha',
                  'Topic :: Software Development :: Libraries',
                  'License :: OSI Approved :: BSD License',
                  'Intended Audience :: Developers',
                  'Programming Language :: Python :: 2.7',
                ])
