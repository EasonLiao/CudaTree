import numpy as np
import cudatree
import time 


inputs = []
best_thresholds = []

all_log_classes = [1,3,6]
all_log_examples = [4,5,6]
all_log_features = [0.5,1,2,2.5]
thresholds =  [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, .1, .2]
for log_n_classes in all_log_classes:
  n_classes = int(2**log_n_classes)
  print "n_classes", n_classes
  for log_n_examples in all_log_examples:
    n_examples = int(10 ** log_n_examples)
    print "n_examples", n_examples
    y = np.random.randint(low = 0, high = n_classes, size = n_examples)
    
    for log_n_features in all_log_features:
      
      n_features = int(10**log_n_features)
      print "n_features", n_features
      if n_features * n_examples > 100 * 10**6:
        print "Skipping due excessive n_features * n_examples..."
        continue
      if n_examples * n_classes > 10 ** 7:
        print "Skipping due to excessive n_examples * n_classes"
        continue 

      x = np.random.randn(n_examples, n_features)
      
      throwaway = cudatree.RandomForestClassifier(n_estimators=1)
      throwaway.fit(x,y)
      best_time = np.inf
      best_threshold = None
      
      for threshold in thresholds:
        bfs_threshold = int(n_examples * threshold)
        print "  -- threshold", threshold, bfs_threshold
        start_t = time.time()
        cudatree.RandomForestClassifier(n_estimators = 3, bootstrap = False, max_features = int(np.sqrt(n_features))).fit(x,y, bfs_threshold) 
        t = time.time() - start_t
        print "  ---> total time", t 
        if t < best_time:
          best_time = t
          best_threshold = threshold
      inputs.append([1.0, log_n_classes, log_n_examples, log_n_features])
      best_thresholds.append(best_threshold)

print inputs
inputs = np.array(inputs)
print inputs.shape

print best_thresholds
best_thresholds = np.array(best_thresholds)
target_values = np.log(100 * best_thresholds)
print target_values.shape

result = np.linalg.lstsq(inputs, target_values)
for elt in result:
  print elt
