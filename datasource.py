from sklearn.datasets import load_digits, load_iris
import cPickle
import numpy as np

_img_data = None

def load_data(ds_name):
  data_dir = "data/"
  global _img_data
  if ds_name == "digits":
    ds = load_digits()
    x_train = ds.data
    y_train = ds.target
  elif ds_name == "iris":
    ds = load_iris()
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

  return x_train, y_train



