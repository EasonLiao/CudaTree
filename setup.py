#!/usr/bin/python
from setuptools import setup, find_packages
import cuda_tree

setup(
    name = "cuda_tree",
    version = cuda_tree.__version__,
    description = "building decison tree on Cuda",
    author = ["Alex Rubinsteyn", "Yisheng Liao"],
    author_email = ["alexr@cs.nyu.edu", "yl1912@nyu.edu"],
    packages = find_packages() + ['cuda_tree.test'],
    package_dir = {'cuda_tree.test' : './test'},
    package_data = {'cuda_tree' : ['cuda_kernels/*.cu']},
    requires = [
        'numpy',
        'sklearn',
        'pycuda'
      ],
    classifiers=['Development Status :: 1 - Alpha',
                  'Topic :: Software Development :: Libraries',
                  'License :: OSI Approved :: BSD License',
                  'Intended Audience :: Developers',
                  'Programming Language :: Python :: 2.7',
                ])


