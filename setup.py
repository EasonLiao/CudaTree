#!/usr/bin/python
from setuptools import setup, find_packages, Extension
#import cudatree
from Cython.Distutils import build_ext


#ext_modules = [Extension("cuda_random_decisiontree_small", ["cudatree/cuda_random_decisiontree_small.pyx"])]

setup(
    name = "cudatree",
    long_decsription='''
CudaTree
========
Building Decision Tree on Cuda.
''',
    version = 1, #cudatree.__version__,
    description = "Building decison tree on Cuda",
    author = ["Alex Rubinsteyn", "Yisheng Liao"],
    author_email = ["alexr@cs.nyu.edu", "yl1912@nyu.edu"],
    packages = find_packages() + ['cudatree.test'],
    package_dir = {'cudatree.test' : './test'},
    package_data = {'cudatree' : ['cuda_kernels/*.cu']},
    #ext_modules = ext_modules,
    #cmdclass = {'build_ext' : build_ext},
    requires = [
        'numpy',
        'sklearn',
        'pycuda',
        "parakeet"
      ],
    classifiers=['Development Status :: 3 - Alpha',
                  'Topic :: Software Development :: Libraries',
                  'License :: OSI Approved :: BSD License',
                  'Intended Audience :: Developers',
                  'Programming Language :: Python :: 2.7',
                ])
