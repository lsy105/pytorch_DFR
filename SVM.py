import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import time 

def load_dataset(snr=-10, ant=4, seed=1338):
  x = loadmat('./Kian_data/recieved1_{}db_{}_{}.mat'.format(abs(snr), ant, ant))
  y = loadmat('./Kian_data/target.mat')
  sig = x['recieved1'].reshape(-1, 1)
  target = y['target'].reshape(-1)
 # fi = loadmat('dss/dataset_{}db_Ant{}_seed{}.mat'.format(snr, ant, seed))
 # sig = fi['receivedSig'].reshape(-1, 1)
 # target = fi['target'].reshape(-1)
  fi = None
  return sig, target, fi

def run_model(model_type="SVM", temporal_window_length=1, snrdbs=(-10, -20), 
              ants=(4, 6), seeds=(1338,)):
  for snrdb in snrdbs:
    for ant in ants:
      for sd in [1000]:
        sig, target, fi = load_dataset(snrdb, ant, seed=sd)

        scaler = StandardScaler()
        sig = scaler.fit_transform(sig)
        
        sig_temp = []
        for i in range(temporal_window_length, len(sig)+1):
          sig_temp.append(sig[i-temporal_window_length:i])
        target_temp = target[temporal_window_length-1:]
        sig_train = np.array(sig_temp[:1000-temporal_window_length+1]).reshape(-1, temporal_window_length)
        sig_test = np.array(sig_temp[1000-temporal_window_length+1:]).reshape(-1, temporal_window_length)
        label_train = target[temporal_window_length-1:1000]
        label_test = target[1000:]

        if model_type == "SVM":
          clf = SVC(kernel='rbf')
          clf.fit(sig_train, label_train)
          t0 = time.time()
          preds = clf.predict(sig_test)
          t1 = time.time()
          print("time: ", t1 - t0)
          print('SVM(RBF)', end='')
        elif model_type == 'LogReg': # Logistic Regression
          clf = LogisticRegression()
          clf.fit(sig_train, label_train)
          preds = clf.predict(sig_test)
          print('LogReg', end='')
        fpr, tpr, thresholds = roc_curve(label_test, preds)
        roc_auc = auc(fpr, tpr)
        accuracy = accuracy_score(label_test, preds)
        mse = mean_squared_error(label_test, preds)
        print("\tSNR:{}db\t#Antenna:{}\tAccuracy:{:.3f}\tMSE:{:.3f}\tAUC:{:.4f}".format(snrdb, ant, accuracy, mse, roc_auc))
    print('-'*10)

run_model("SVM", 8)
