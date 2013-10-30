import cPickle
import numpy as np
from os import path
from sklearn.datasets import load_digits, load_iris, load_diabetes, fetch_covtype 

_img_data = None

def load_data(ds_name):
  data_dir = path.dirname(__file__) + "/../data/"
  global _img_data
  if ds_name == "digits":
    ds = load_digits()
    x_train = ds.data
    y_train = ds.target
  elif ds_name == "iris":
    ds = load_iris()
    x_train = ds.data
    y_train = ds.target
  elif ds_name == "diabetes":
    ds = load_diabetes()
    x_train = ds.data 
    y_train = ds.target > 140 
  elif ds_name == "covtype":
    ds = fetch_covtype(download_if_missing = True)
    x_train = ds.data 
    y_train = ds.target 
  elif ds_name == "db":
    with open(data_dir + "data_batch_1", "r") as f:
      ds = cPickle.load(f)
      x_train = ds['data']
      y_train = np.array(ds['labels'])
  elif ds_name == "train":
    with open(data_dir + "train", "r") as f:
      ds = cPickle.load(f)
      x_train = ds['data']
      y_train = np.array(ds['fine_labels'])
  elif ds_name == "db_test":
    with open(data_dir + "test_batch", "r") as f:
      ds = cPickle.load(f)
      x_train = ds['data']
      y_train = np.array(ds['labels'])
  elif ds_name == "train_test":
    with open(data_dir + "test", "r") as f:
      ds = cPickle.load(f)
      x_train = ds['data']
      y_train = np.array(ds['fine_labels'])
  elif ds_name == "inet":
    if _img_data is None:
      with open("/ssd/imagenet-subset.pickle", "r") as f:
        _img_data = cPickle.load(f)
    return _img_data['x'][0:10000],  _img_data['Y'][0:10000] 
  elif ds_name == "inet_test":
    if _img_data is None:
      with open("/ssd/imagenet-subset.pickle", "r") as f:
        _img_data = cPickle.load(f)
    return _img_data['x'][10000:],  _img_data['Y'][10000:] 
  elif ds_name == "kdd":
    data = np.load(data_dir + "data.npy")
    x_train = data[:, :-1]
    y_train = data[:, -1]
  else:
    assert False, "Unrecognized data set name %s" % ds_name
  return x_train, y_train



