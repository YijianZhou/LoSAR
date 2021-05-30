""" Dataset for CNN & RNN Model
"""
import os, glob
import torch
from torch.utils.data import Dataset
from obspy import read, Stream
import numpy as np
import config

# set config
cfg = config.Config()
samp_rate = cfg.samp_rate
num_chn = cfg.num_chn
num_steps = cfg.num_steps
win_len = int(cfg.win_len * cfg.samp_rate)
step_len = int(cfg.step_len * cfg.samp_rate)
step_stride = int(cfg.step_stride * cfg.samp_rate)


class Events(Dataset):
  """ Data pipeline for Event samples
  """
  def __init__(self, sample_list):
    self.win_len = win_len
    self.num_chn = cfg.num_chn
    self.samples = np.load(sample_list)

  def __getitem__(self, index):
    # set label
    st_paths = self.samples[index]
    # read data
    st = Stream([read(st_path)[0] for st_path in st_paths])
    # output (data, target)
    data = np.zeros([self.num_chn, self.win_len], dtype=np.float32)
    for chn_idx in range(num_chn):
        npts = min(self.win_len, len(st[chn_idx]))
        data[chn_idx,0:npts] = st[chn_idx].data[0:npts]
    return data

  def __len__(self):
    return len(self.samples)


class Sequences(Dataset):
  """ Data pipeline for RNN training
  """
  def __init__(self, pos_list):
    self.num_chn = num_chn
    self.num_steps = num_steps
    self.step_len = step_len
    self.step_stride = step_stride
    self.samples = np.load(pos_list)
    self.get_seq_target = get_seq_target

  def __getitem__(self, index):
    # set label
    pos_paths = self.samples[index]
    # read data
    st_pos = Stream([read(st_path)[0] for st_path in pos_paths])
    # get target
    header = st_pos[0].stats.sac
    tp, ts = header.t0, header.t1
    target_seq = self.get_seq_target(tp, ts)
    # output (data, target)
    data_seq = np.zeros([self.num_steps, self.step_len*self.num_chn], dtype=np.float32)
    for step_idx in range(self.num_steps):
      for chn_idx in range(self.num_chn):
        chn_idx_0 = chn_idx * self.step_len
        step_idx_0 = step_idx * self.step_stride
        if len(st_pos[chn_idx]) < step_idx_0+self.step_len: continue
        data_seq[step_idx, chn_idx_0 : chn_idx_0 + self.step_len] = \
            st_pos[chn_idx].data[step_idx_0 : step_idx_0 + self.step_len]
    return data_seq, target_seq

  def __len__(self):
    return len(self.samples)


# get target of sequence samples
def get_seq_target(tp, ts):
    tp, ts = tp*samp_rate, ts*samp_rate
    idx_p = 0 if tp<step_len else int((tp-step_len)/step_stride) + 1
    idx_s = int((ts-step_len)/step_stride) + 1 
    if idx_s>num_steps: idx_s = num_steps
    if idx_s<=idx_p: idx_s = idx_p+1
    # target: Noise, P and S 
    target_n = np.zeros(idx_p, dtype=np.int_)
    target_p = np.ones(idx_s-idx_p, dtype=np.int_)
    target_s = 2*np.ones(num_steps-idx_s, dtype=np.int_)
    return np.concatenate([target_n, target_p, target_s])

