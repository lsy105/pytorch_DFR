import torch
from torch.utils import data
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, data, labels, sequence_length, noise=False):
        self.labels = labels.flatten()
        self.data = data.flatten()
        self.sequence_length = sequence_length
        self.all_data = np.concatenate((np.array([0 for _ in range(sequence_length)]), self.data), axis=0)
        self.all_labels = np.concatenate((np.array([0 for _ in range(sequence_length)]), self.labels), axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Select sample
        idx += self.sequence_length
        X = []
        for i in range(idx - self.sequence_length + 1, idx + 1, 1):
            X.append(self.all_data[i])
        y = self.all_labels[idx]
        return np.array(X), np.array(y)

class DatasetMixed(data.Dataset):
  def __init__(self, data1, data2, labels, sequence_length, noise=False):
        self.labels = labels.flatten()
        self.data1 = data1.flatten()
        self.data2 = data2.flatten()
        self.sequence_length = sequence_length
        self.data1 = np.concatenate((np.array([0 for _ in range(sequence_length)]), self.data1), axis=0)
        self.data2 = np.concatenate((np.array([0 for _ in range(sequence_length)]), self.data2), axis=0)
        self.all_labels = np.concatenate((np.array([0 for _ in range(sequence_length)]), self.labels), axis=0)

  def __len__(self):
        return len(self.labels)

  def __getitem__(self, idx):
        # Select sample
        idx += self.sequence_length
        X1, X2 = [], []
        for i in range(idx - self.sequence_length + 1, idx + 1, 1):
            X1.append(self.data1[i])
            X2.append(self.data2[i])
        y = self.all_labels[idx]
        return np.array(X1), np.array(X2), np.array(y)

class TwoDataset(data.Dataset):
  def __init__(self, data1, data2, labels, sequence_length, noise=False):
       self.labels = labels.flatten()
       self.data1 = data1.flatten()
       self.data2 = data2.flatten()
       self.sequence_length = sequence_length
       self.all_data1 = np.concatenate((np.array([0 for _ in range(sequence_length)]), self.data1), axis=0)
       self.all_data2 = np.concatenate((np.array([0 for _ in range(sequence_length)]), self.data2), axis=0)
       self.all_labels = np.concatenate((np.array([0 for _ in range(sequence_length)]), self.labels), axis=0)

  def __len__(self):
       return len(self.data1)

  def __getitem__(self, idx):
       # Select sample
       idx += self.sequence_length
       X1 = []
       X2 = []
       for i in range(idx - self.sequence_length + 1, idx + 1, 1):
           X1.append(self.all_data1[i])
           X2.append(self.all_data2[i])
       y = self.all_labels[idx]
       return np.array(X1), np.array(X2), np.array(y)
