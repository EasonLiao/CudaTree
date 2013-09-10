from setuptools import setup, find_packages
import cuda_tree

setup(
    name = "cuda_tree",
    version = cuda_tree.__version__,
    description = "building decison tree on Cuda",
    author = ["Alex Rubinsteyn", "Yisheng Liao"],
    packages = find_packages() + ['cuda_tree.test'],
    package_dir = {'cuda_tree.test' : './test'},
    package_data = {'cuda_tree' : ['cuda_kernels/*.cu']},
    requires = [
        'numpy',
        'sklearn',
        'pycuda'
      ])


