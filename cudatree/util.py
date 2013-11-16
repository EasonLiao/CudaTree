import time
import numpy as np
from pycuda.compiler import SourceModule
from os import path
import operator
import os
import logging

logging.basicConfig(level=logging.DEBUG)

log_debug = logging.debug
log_info = logging.info
log_warn = logging.warn
log_error = logging.error
log_fatal = logging.fatal


_kernel_cache = {}
_module_cache = {}

kernels_dir = path.dirname(os.path.realpath(__file__)) +\
                "/cuda_kernels/"

def compile_module(module_file, params):
  module_file = kernels_dir + module_file
  key = (module_file, params)
  if key in _module_cache:
    return _module_cache[key]
  
  with open(module_file) as code_file:
    start_timer("compile module")
    code = code_file.read()
    src = code % params
    mod = SourceModule(src, include_dirs = [kernels_dir])
    _module_cache[key] = mod
    end_timer("compile module")
    return mod


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

def mk_kernel(params, func_name, kernel_file, prepare_args = None):
  kernel_file = kernels_dir + kernel_file
  key = (params, kernel_file, prepare_args)
  if key in _kernel_cache:
    return _kernel_cache[key]
  
  with open(kernel_file) as code_file:
    code = code_file.read()
    src = code % params
    mod = SourceModule(src, include_dirs = [kernels_dir])
    fn = mod.get_function(func_name)
    if prepare_args is not None: fn.prepare(prepare_args)
    _kernel_cache[key] = fn
    return fn

def mk_tex_kernel(params, 
                  func_name, 
                  tex_name, 
                  kernel_file, 
                  prepare_args = None):
  kernel_file = kernels_dir + kernel_file
  key = (params, kernel_file, prepare_args)
  if key in _kernel_cache:
    return _kernel_cache[key]

  with open(kernel_file) as code_file:
    code = code_file.read()
    src = code % params
    mod = SourceModule(src, include_dirs = [kernels_dir])
    fn = mod.get_function(func_name)
    tex = mod.get_texref(tex_name)
    if prepare_args is not None: fn.prepare(prepare_args)
    _kernel_cache[key] = (fn, tex)
    return fn, tex

def test_diff(x, y):
  """ Test how many elements betweenn array 
  x and y are different. """
  assert isinstance(x, np.ndarray)
  assert isinstance(y, np.ndarray)
  assert x.size == y.size
  diff = x - y
  return (np.count_nonzero(diff), x.size)


start_times = {}
total_times = {}

def start_timer(name):
  start_times[name] = time.time()
  
def end_timer(name):
  total = total_times.get(name, 0)
  total += time.time() - start_times[name]
  total_times[name] = total

def show_timings(limit = 100):
  tables = sorted(total_times.iteritems(), 
                  key = operator.itemgetter(1), 
                  reverse = True) 
  idx = 0
  print "---------Timings---------"
  for key, value in tables:
    print key.ljust(15), ":", value
    idx += 1
    if idx == limit:
      break

  print "-------------------------"
