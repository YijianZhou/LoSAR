""" Dataset for SAR training
"""
import os, glob
import torch
from torch.utils.data import Dataset
import zarr
from obspy import read, Stream
import numpy as np

class Positive_Negative(Dataset):
  def __init__(self, zarr_path, zarr_group):
    pos_data_path = os.path.join(zarr_path, zarr_group, 'positive_data')
    pos_tar_path = os.path.join(zarr_path, zarr_group, 'positive_target')
    neg_data_path = os.path.join(zarr_path, zarr_group, 'negative_data')
    neg_tar_path = os.path.join(zarr_path, zarr_group, 'negative_target')
    self.pos_data = zarr.open(pos_data_path, mode='r')
    self.neg_data = zarr.open(neg_data_path, mode='r')
    self.pos_tar = zarr.open(pos_tar_path, mode='r')
    self.neg_tar = zarr.open(neg_tar_path, mode='r')
    num_pos, num_neg = self.pos_data.shape[0], self.neg_data.shape[0]
    self.num_samples = num_pos
    self.neg_ratio = num_neg / num_pos
    self.pos_idx = np.random.permutation(num_pos)
    self.neg_idx = np.tile(np.arange(num_neg), int(num_pos/num_neg)+1)
    self.neg_idx = np.random.permutation(self.neg_idx)[0:num_pos]

  def __getitem__(self, index):
    pos_di = self.pos_data[self.pos_idx[index]]
    neg_di = self.neg_data[self.neg_idx[index]]
    pos_ti = self.pos_tar[self.pos_idx[index]]
    neg_ti = self.neg_tar[self.neg_idx[index]]
    return np.array([pos_di, neg_di]), np.array([pos_ti, neg_ti])

  def __len__(self):
    return self.num_samples
