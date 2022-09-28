from sklearn.metrics import roc_auc_score
import numpy as np


def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

import scipy.io
y = scipy.io.loadmat('../target.mat')
y_out = y['target'][1000:]
y_out = one_hot(y_out, 2)
p_out = []
fi = open('example.txt', 'r')
for line in fi:
    data = list(map(float, line.strip().split()))
    p_out.append(data)
p_out = np.array(p_out)
pred = np.argmax(p_out, axis=1)
print(roc_auc_score(y_out, p_out))
print(np.mean(pred == y['target'][1000:]))
