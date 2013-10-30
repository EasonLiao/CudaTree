
from cudatree import RandomForestClassifier

def compare_accuracy(x,y, n_estimators = 11, bootstrap = True, slop = 0.98, n_repeat = 10):
  n = x.shape[0] / 2 
  xtrain = x[:n]
  ytrain = y[:n]
  xtest = x[n:]
  ytest = y[n:]
  cudarf = RandomForestClassifier(n_estimators = n_estimators, bootstrap = bootstrap)
  import sklearn.ensemble
  skrf = sklearn.ensemble.RandomForestClassifier(n_estimators = n_estimators, bootstrap = bootstrap, max_features = "log2")
  cuda_score_total = 0 
  sk_score_total = 0
  for i in xrange(n_repeat):
    print "cuda fit"
    cudarf.fit(xtrain, ytrain)
    print "sk fit"
    skrf.fit(xtrain, ytrain)
    print "sk predict"
    sk_score = skrf.score(xtest, ytest)
    print "cuda predict"
    cuda_score = cudarf.score(xtest, ytest)
    print "Iteration", i 
    print "Sklearn score", sk_score 
    print "CudaTree score", cuda_score 
    sk_score_total += sk_score 
    cuda_score_total += cuda_score 

  assert cuda_score_total >= (sk_score_total * slop), \
    "Getting significantly worse test accuracy than sklearn: %s vs. %s" % (cuda_score_total / n_repeat, sk_score_total / n_repeat)



from sklearn import datasets
cov = datasets.fetch_covtype()
#cov = datasets.load_lfw_people()

x_train = cov['data']#[0:10000]
y_train = cov['target']#[0:10000]

"""
n = y_train.size / 2

x_test = x_train[n:]
y_test = y_train[n:]

x_train = x_train[:n]
y_train = y_train[:n]
"""

compare_accuracy(x_train, y_train, 21)

