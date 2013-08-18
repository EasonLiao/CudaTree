from sklearn.datasets import load_digits, load_iris
import cPickle
import numpy as np

def load_data(ds_name):
  if ds_name == "digits":
    ds = load_digits()
    x_train = ds.data
    y_train = ds.target
  elif ds_name == "iris":
    ds = load_iris()
    x_train = ds.data
    y_train = ds.target
  elif ds_name == "db":
    with open("data_batch_1", "r") as f:
      ds = cPickle.load(f)
      x_train = ds['data']
      y_train = np.array(ds['labels'])
  elif ds_name == "train":
    with open("train", "r") as f:
      ds = cPickle.load(f)
      x_train = ds['data']
      y_train = np.array(ds['fine_labels'])
  return x_train, y_train



