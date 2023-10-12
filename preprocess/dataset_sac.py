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
num_steps = cfg.rnn_num_steps
win_len = int(cfg.win_len * cfg.samp_rate)
step_len = int(cfg.rnn_step_len * cfg.samp_rate)
step_stride = int(cfg.rnn_step_stride * cfg.samp_rate)

class Sequences(Dataset):
  """ Dataset for sac2zarr
  """
  def __init__(self, sample_list, is_pos):
    self.samples = np.load(sample_list)
    self.is_pos = is_pos
    self.get_seq_target = get_seq_target

  def __getitem__(self, index):
    # read data
    st_paths = self.samples[index]
    st = Stream([read(st_path)[0] for st_path in st_paths])
    st_data = torch.from_numpy(np.array([tr.data[0:win_len] for tr in st])).float()
    # get target
    header = st[0].stats.sac
    if self.is_pos: tp, ts = header.t0, header.t1
    else: tp, ts = -1, -1
    target_seq = self.get_seq_target(tp, ts, self.is_pos)
    try: data_seq = st_data.unfold(1, step_len, step_stride).permute(1,0,2)
    except: data_seq = torch.zeros(num_steps, num_chn, step_len)
    data_seq = data_seq.reshape(data_seq.size(0), -1)
    if data_seq.size(0)!=num_steps: 
        num_step_pad = num_steps - data_seq.size(0)
        data_seq = torch.cat((data_seq, torch.zeros(num_step_pad, data_seq.size(1))))
    return data_seq, target_seq

  def __len__(self):
    return len(self.samples)


# get target of sequence samples
def get_seq_target(tp, ts, is_pos):
    target_seq = np.zeros(num_steps, dtype=np.int_)
    if not is_pos: return target_seq
    tp_idx, ts_idx = tp*samp_rate, ts*samp_rate
    tp_step_idx0 = 0 if tp_idx<step_len else int((tp_idx-step_len)/step_stride) + 1
    tp_step_idx1 = int(tp_idx / step_stride) + 1
    target_seq[tp_step_idx0:tp_step_idx1] = 1
    ts_step_idx0 = int((ts_idx-step_len)/step_stride) + 1 
    ts_step_idx1 = int(ts_idx / step_stride) + 1
    if ts_step_idx0<=tp_step_idx0: ts_step_idx0 = tp_step_idx0 + int((tp_step_idx1-tp_step_idx0)/2)
    target_seq[ts_step_idx0:ts_step_idx1] = 2
    return target_seq
