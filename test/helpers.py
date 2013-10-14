
from cudatree import RandomForestClassifier

def compare_accuracy(x,y, n_estimators = 21, bootstrap = True, slop = 0.98):
  n = x.shape[0] / 2 
  xtrain = x[:n]
  ytrain = y[:n]
  xtest = x[n:]
  ytest = y[n:]
  n_estimators = 10
  cudarf = RandomForestClassifier(n_estimators = n_estimators, bootstrap = bootstrap)
  cudarf.fit(xtrain, ytrain)
  import sklearn.ensemble
  skrf = sklearn.ensemble.RandomForestClassifier(n_estimators = n_estimators, bootstrap = bootstrap)
  skrf.fit(xtrain, ytrain)
  sk_score = skrf.score(xtest, ytest)
  cuda_score = cudarf.score(xtest, ytest)
  print "Sklearn score", sk_score 
  print "CudaTree score", cuda_score 
  assert cuda_score >= (sk_score * slop), "Getting significantly worse test accuracy than sklearn: %s vs. %s" % (cuda_score, sk_score)
