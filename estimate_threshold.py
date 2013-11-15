import numpy as np
import cudatree
import time 


inputs = []
best_threshold_prcts = []
best_threshold_values = []


all_classes = [2, 10, 100]
all_examples = [10**4, 10**5, 10**6]
all_features = [10, 100, 1000]
thresholds = [1000, 2000, 2500, 3000, 3500, 5000, 10000, 20000]
# thresholds =  [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, .1, .2]
for n_classes in all_classes:
  print "n_classes", n_classes
  for n_examples in all_examples:
    print "n_examples", n_examples
    y = np.random.randint(low = 0, high = n_classes, size = n_examples)
    for n_features in all_features:
      print "n_features", n_features
      if n_features * n_examples > 100 * 10**6:
        print "Skipping due excessive n_features * n_examples..."
        continue
      if n_examples * n_classes > 10 ** 7:
        print "Skipping due to excessive n_examples * n_classes"
        continue 

      x = np.random.randn(n_examples, n_features)
      rf = cudatree.RandomForestClassifier(n_estimators = 3, bootstrap = False, max_features = int(np.sqrt(n_features)))
      # warm up
      rf.fit(x[:100],y[:100])
      best_time = np.inf
      best_threshold = None
      best_threshold_prct = None 
      for bfs_threshold in thresholds:
        bfs_threshold_prct = float(bfs_threshold) / n_examples
        print "  -- threshold",  bfs_threshold, "(", bfs_threshold_prct, ")"
        start_t = time.time()
        rf.fit(x, y, bfs_threshold)
        t = time.time() - start_t
        print "  ---> total time", t 
        if t < best_time:
          best_time = t
          best_threshold = bfs_threshold
          best_theshold_prct = bfs_threshold_prct

      inputs.append([1.0, n_classes, n_examples, n_features])
      best_threshold_values.append(best_threshold)
      best_threshold_prcts.append(best_threshold_prct)

X = np.array(inputs)
print X.shape

best_threshold_prcts = np.array(best_threshold_prcts)
best_threshold_values = np.array(best_threshold_values)


result = np.linalg.lstsq(inputs, best_threshold_values)
print "Regression coefficients:", result[0]
print "Regression residuals:", result[1]
print "Rank:", result[2]


import socket 
csv_filename = "threshold_results_" + socket.gethostname()
with open(csv_filename, 'w') as csvfile:
    for i, input_tuple in enumerate(inputs):
      csvfile.write(str(input_tuple[1:]))
      csvfile.write("," + str(best_threshold_values[i]))
      csvfile.write("," + str(best_threshold_prcts[i]))
      csvfile.write("\n")

