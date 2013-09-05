import time
import numpy as np
from pycuda.compiler import SourceModule
from os import path

_kernel_cache = {}

def get_best_dtype(max_value):
  """ Find the best dtype to minimize the memory usage"""
  if max_value <= np.iinfo(np.uint8).max:
    return np.dtype(np.uint8)
  if max_value <= np.iinfo(np.uint16).max:
    return np.dtype(np.uint16)
  if max_value <= np.iinfo(np.uint32).max:
    return np.dtype(np.uint32)
  else:
    return np.dtype(np.uint64)

class timer(object):
  def __init__(self, name):
    self.name = name

  def __enter__(self, *args):
    print "Running %s" % self.name 
    self.start_t = time.time()

  def __exit__(self, *args):
    print "Time for %s: %s" % (self.name, time.time() - self.start_t)


def dtype_to_ctype(dtype):
  if dtype.kind == 'f':
    if dtype == 'float32':
      return 'float'
    else:
      assert dtype == 'float64', "Unsupported dtype %s" % dtype
      return 'double'
  assert dtype.kind in ('u', 'i')
  return "%s_t" % dtype 


def mk_kernel(params, func_name, kernel_file):
  kernel_file = path.dirname(__file__) + "/cuda_kernels/" + kernel_file
  key = (params, kernel_file)
  if key in _kernel_cache:
    return _kernel_cache[key]

  with open(kernel_file) as code_file:
    code = code_file.read()
    src = code % params
    mod = SourceModule(src)
    fn = mod.get_function(func_name)
    _kernel_cache[key] = fn
    return fn

def mk_tex_kernel(params, func_name, tex_name, kernel_file):
  kernel_file = path.dirname(__file__) + "/cuda_kernels/" + kernel_file
  key = (params, kernel_file)
  if key in _kernel_cache:
    return _kernel_cache[key]

  with open(kernel_file) as code_file:
    code = code_file.read()
    src = code % params
    mod = SourceModule(src)
    fn = mod.get_function(func_name)
    tex = mod.get_texref(tex_name)
    _kernel_cache[key] = (fn, tex)
    return fn, tex

