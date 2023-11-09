import os, glob
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class Positive_Negative(Dataset):
  """ Dataset for SAR training
  """
  def __init__(self, hdf5_path, train_group):
    self.hdf5_path = hdf5_path
    self.train_group = train_group
    with h5py.File(self.hdf5_path,'r') as h5_file:
        # get num samples
        num_pos = len(h5_file[self.train_group+'_pos_target'])
        num_neg = len(h5_file[self.train_group+'_neg_target'])
        self.num_samples = min(num_pos, num_neg)
    # rand shuffle pos & neg index
    self.pos_idx = np.arange(num_pos)
    np.random.shuffle(self.pos_idx)
    self.pos_idx = self.pos_idx[0:self.num_samples]
    self.neg_idx = np.arange(num_neg)
    np.random.shuffle(self.neg_idx)
    self.neg_idx = self.neg_idx[0:self.num_samples]

  def __getitem__(self, index):
    with h5py.File(self.hdf5_path,'r') as h5_file:
        pos_data = h5_file[self.train_group+'_pos_data'][self.pos_idx[index]]
        pos_target = h5_file[self.train_group+'_pos_target'][self.pos_idx[index]]
        neg_data = h5_file[self.train_group+'_neg_data'][self.neg_idx[index]]
        neg_target = h5_file[self.train_group+'_neg_target'][self.neg_idx[index]]
    return np.array([pos_data, neg_data]), np.array([pos_target, neg_target])

  def __len__(self):
    return self.num_samples
