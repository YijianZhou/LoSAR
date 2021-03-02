""" Dataset for CNN & RNN Model
"""
import os, glob
import torch
from torch.utils.data import Dataset
import zarr
from obspy import read, Stream
import numpy as np
import config


class Events(Dataset):
  """ Data pipeline for CNN training
  """
  def __init__(self, zarr_path, zarr_group):
    pos_path = os.path.join(zarr_path, zarr_group, 'positive')
    neg_path = os.path.join(zarr_path, zarr_group, 'negative')
    self.pos_data = zarr.open(pos_path, mode='r')
    self.neg_data = zarr.open(neg_path, mode='r')
    num_pos, num_neg = self.pos_data.shape[0], self.neg_data.shape[0]
    self.num_samples = min(num_pos, num_neg)
    self.pos_idx = np.arange(num_pos)
    np.random.shuffle(self.pos_idx)
    self.pos_idx = self.pos_idx[0:self.num_samples]
    self.neg_idx = np.arange(num_neg)
    np.random.shuffle(self.neg_idx)
    self.neg_idx = self.neg_idx[0:self.num_samples]

  def __getitem__(self, index):
    neg_di = self.neg_data[self.neg_idx[index]]
    pos_di = self.pos_data[self.pos_idx[index]]
    return np.array([neg_di, pos_di]), np.array([0,1])

  def __len__(self):
    return self.num_samples


class Sequences(Dataset):
  """ Data pipeline for RNN training
  """
  def __init__(self, zarr_path, zarr_group):
    data_path = os.path.join(zarr_path, zarr_group, 'sequence_data')
    target_path = os.path.join(zarr_path, zarr_group, 'sequence_target')
    self.seq_data = zarr.open(data_path, mode='r')
    self.seq_target = zarr.open(target_path, mode='r')

  def __getitem__(self, index):
    return self.seq_data[index], self.seq_target[index]

  def __len__(self):
    return self.seq_data.shape[0]

