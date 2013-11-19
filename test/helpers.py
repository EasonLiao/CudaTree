from cudatree import RandomForestClassifier
import hybridforest

def compare_accuracy(x,y, n_estimators = 11, bootstrap = True, slop = 0.98, n_repeat = 10):
  n = x.shape[0] / 2 
  xtrain = x[:n]
  ytrain = y[:n]
  xtest = x[n:]
  ytest = y[n:]
  cudarf = RandomForestClassifier(n_estimators = n_estimators, bootstrap = bootstrap)
  import sklearn.ensemble
  skrf = sklearn.ensemble.RandomForestClassifier(n_estimators = n_estimators, bootstrap = bootstrap)
  cuda_score_total = 0 
  sk_score_total = 0
  for i in xrange(n_repeat):
    cudarf.fit(xtrain, ytrain)
    skrf.fit(xtrain, ytrain)
    sk_score = skrf.score(xtest, ytest)
    cuda_score = cudarf.score(xtest, ytest)
    print "Iteration", i 
    print "Sklearn score", sk_score 
    print "CudaTree score", cuda_score 
    sk_score_total += sk_score 
    cuda_score_total += cuda_score 

  assert cuda_score_total >= (sk_score_total * slop), \
    "Getting significantly worse test accuracy than sklearn: %s vs. %s"\
    % (cuda_score_total / n_repeat, sk_score_total / n_repeat)


def compare_hybrid_accuracy(x,y, n_estimators = 20, bootstrap = True, slop = 0.98, n_repeat = 5):
  n = x.shape[0] / 2 
  xtrain = x[:n]
  ytrain = y[:n]
  xtest = x[n:]
  ytest = y[n:]
  hybridrf = hybridforest.RandomForestClassifier(n_estimators = n_estimators, bootstrap = bootstrap)
  import sklearn.ensemble
  skrf = sklearn.ensemble.RandomForestClassifier(n_estimators = n_estimators, bootstrap = bootstrap)
  cuda_score_total = 0 
  sk_score_total = 0
  for i in xrange(n_repeat):
    hybridrf.fit(xtrain, ytrain)
    skrf.fit(xtrain, ytrain)
    sk_score = skrf.score(xtest, ytest)
    cuda_score = hybridrf.score(xtest, ytest)
    print "Iteration", i 
    print "Sklearn score", sk_score 
    print "Hybrid score", cuda_score 
    sk_score_total += sk_score 
    cuda_score_total += cuda_score 

  assert cuda_score_total >= (sk_score_total * slop), \
    "Getting significantly worse test accuracy than sklearn: %s vs. %s"\
    % (cuda_score_total / n_repeat, sk_score_total / n_repeat)

