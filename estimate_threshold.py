import numpy as np
import cudatree
import time 


inputs = []
best_threshold_prcts = []
best_threshold_values = []


all_classes = [2, 10, 50, 100, 500]
all_examples = [10**4, 5 * 10**4, 2 * 10**5, 8 * 10**5, 2 * 10**6 ]
all_features = [8, 16, 32, 64, 512, 2048]
thresholds = [1000, 2000, 3000, 4000, 5000, 7500, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000]

# np.exp(np.linspace(np.log(1000), np.log(50000), num = 15)).astype('int')
total_iters = len(all_classes) * len(all_examples) * len(all_features) * len(thresholds)

i = 1 
# thresholds =  [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, .1, .2]
for n_classes in all_classes:
  print "n_classes", n_classes
  for n_examples in all_examples:
    print "n_examples", n_examples
    y = np.random.randint(low = 0, high = n_classes, size = n_examples)
    for n_features in all_features:
      print "n_features", n_features
      max_features = int(np.sqrt(n_features))
      print "sqrt(n_features) =", max_features 
      if n_features * n_examples > 100 * 10**6:
        print "Skipping due excessive n_features * n_examples..."
	i += len(thresholds)
        continue
      if n_examples * n_classes > 10 ** 8:
        print "Skipping due to excessive n_examples * n_classes"
	i += len(thresholds)
        continue
	 

      x = np.random.randn(n_examples, n_features)
      rf = cudatree.RandomForestClassifier(n_estimators = 3, bootstrap = False, max_features = max_features)
      # warm up
      rf.fit(x[:100],y[:100])
      best_time = np.inf
      best_threshold = None
      best_threshold_prct = None 
      print "(n_classes = %d, n_examples = %d, max_features = %d)" % (n_classes, n_examples, max_features)
      for bfs_threshold in thresholds:
        bfs_threshold_prct = float(bfs_threshold) / n_examples
        print "  -- (%d / %d) threshold %d (%0.2f%%)" % (i, total_iters,  bfs_threshold, bfs_threshold_prct * 100)
        i += 1 
        if bfs_threshold > n_examples:
          print "Skipping threshold > n_examples" 
	  continue 
       
        start_t = time.time()
        rf.fit(x, y, bfs_threshold)
        t = time.time() - start_t
        print "  ---> total time", t 
        if t < best_time:
          best_time = t
          best_threshold = bfs_threshold
          best_theshold_prct = bfs_threshold_prct

      inputs.append([1.0, n_classes, n_examples, max_features])
      best_threshold_values.append(best_threshold)
      best_threshold_prcts.append(best_threshold_prct)

X = np.array(inputs)
print "input shape", X.shape



best_threshold_prcts = np.array(best_threshold_prcts)
best_threshold_values = np.array(best_threshold_values)
Y = best_threshold_values

lstsq_result = np.linalg.lstsq(X, Y)
print "Regression coefficients:", lstsq_result[0]
n = len(best_threshold_values)
print "Regression residual:", lstsq_result[1], "RMSE:", np.sqrt(lstsq_result[1] / n)

import socket 
csv_filename = "threshold_results_" + socket.gethostname()
with open(csv_filename, 'w') as csvfile:
    for i, input_tuple in enumerate(inputs):
      csvfile.write(str(input_tuple[1:]))
      csvfile.write("," + str(best_threshold_values[i]))
      csvfile.write("," + str(best_threshold_prcts[i]))
      csvfile.write("\n")

LogX = X.copy()
LogX[:, 1:] = np.log(X[:, 1:])
LogY = np.log(Y)

log_lstsq_result = np.linalg.lstsq(LogX, LogY)

print "Log regression coefficients:", log_lstsq_result[0]
n = len(best_threshold_values)
print "Log regression residual:", log_lstsq_result[1], "RMSE:", np.sqrt(log_lstsq_result[1] / n)
log_pred = np.dot(LogX, log_lstsq_result[0])
pred = np.exp(log_pred)
residual = np.sum((Y - pred)**2)
print "Actual residual", residual 
print "Actual RMSE:", np.sqrt(residual / n)


"""
import sklearn
import sklearn.linear_model
ridge = sklearn.linear_model.RidgeCV(alphas = [0.01, 0.1, 1, 10, 100], fit_intercept = False)
ridge.fit(X, Y)
print "Ridge regression coef", ridge.coef_
print "Ridge regression alpha", ridge.alpha_

pred = ridge.predict(X)
sse = np.sum( (pred - Y) ** 2)
print "Ridge residual", sse
print "Ridge RMSE", np.sqrt(sse / n)
"""
