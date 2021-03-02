""" Make Zarr format dataset with SAC files
"""
import os, shutil, glob, sys
sys.path.append('/home/zhouyj/software/CERP_Pytorch/preprocess')
import zarr
import torch.multiprocessing as mp
import numpy as np
from torch.utils.data import DataLoader
import config_rc as config
import warnings
warnings.filterwarnings("ignore")

# i/o paths
out_path = '/data3/bigdata/zhouyj/RC_train/rc_scsn_win-20s_freq-2-40hz.zarr'
pos_root = '/data3/bigdata/zhouyj/RC_train/pos-scsn_win-20s_freq-2-40hz'
neg_root = '/data3/bigdata/zhouyj/RC_train/neg-scsn_win-20s_freq-2-40hz_1'
train_pos = os.path.join(pos_root,'train_pos.npy')
valid_pos = os.path.join(pos_root,'valid_pos.npy')
train_neg = os.path.join(neg_root,'train_neg.npy')
valid_neg = os.path.join(neg_root,'valid_neg.npy')
cdrp_prep_dir = '/home/zhouyj/software/CDRP_Pytorch/preprocess'
shutil.copyfile('config_rc.py', os.path.join(cdrp_prep_dir, 'config.py'))
from dataset_sac import Events, Sequences
num_workers = 10
chunk_size = 1
# NN config
cfg = config.Config()
samp_rate = cfg.samp_rate
num_chn = cfg.num_chn
num_steps = cfg.num_steps
win_len = int(cfg.win_len * cfg.samp_rate)
step_len = int(cfg.step_len * cfg.samp_rate)
step_stride = int(cfg.step_stride * cfg.samp_rate)
# setup dataloader
train_pos_set = Events(train_pos)
valid_pos_set = Events(valid_pos)
train_neg_set = Events(train_neg)
valid_neg_set = Events(valid_neg)
train_seq_set = Sequences(train_pos)
valid_seq_set = Sequences(valid_pos)
train_pos_loader = DataLoader(train_pos_set, batch_size=None, shuffle=False, num_workers=num_workers)
valid_pos_loader = DataLoader(valid_pos_set, batch_size=None, shuffle=False, num_workers=num_workers)
train_neg_loader = DataLoader(train_neg_set, batch_size=None, shuffle=False, num_workers=num_workers)
valid_neg_loader = DataLoader(valid_neg_set, batch_size=None, shuffle=False, num_workers=num_workers)
train_seq_loader = DataLoader(train_seq_set, batch_size=None, shuffle=False, num_workers=num_workers)
valid_seq_loader = DataLoader(valid_seq_set, batch_size=None, shuffle=False, num_workers=num_workers)


def write_sequence(zarr_dset, data_loader):
    num_samples = len(data_loader)
    data_shape = (num_samples, num_steps, step_len*num_chn)
    data_chunks = (chunk_size, num_steps, step_len*num_chn)
    target_shape = (num_samples, num_steps)
    target_chunks = (chunk_size, num_steps)
    data_out = os.path.join(out_path, zarr_dset+'_data')
    target_out = os.path.join(out_path, zarr_dset+'_target')
    print('writing %s & %s'%(data_out, target_out))
    z_data = zarr.open(data_out, mode='w', shape=data_shape, chunks=data_chunks, dtype=np.float32)
    z_target = zarr.open(target_out, mode='w', shape=target_shape, chunks=target_chunks, dtype=np.int_)
    for idx, (data, target) in enumerate(data_loader):
        if idx%1000==0: print("done / total = %d / %d" %(idx, num_samples))
        z_data[idx] = data
        z_target[idx] = target


def write_event(zarr_dset, data_loader):
    num_samples = len(data_loader)
    shape = (num_samples, num_chn, win_len)
    chunks = (chunk_size, num_chn, win_len)
    zarr_out = os.path.join(out_path, zarr_dset)
    print('writing %s'%zarr_out)
    z = zarr.open(zarr_out, mode='w', shape=shape, chunks=chunks, dtype=np.float32)
    for idx, data in enumerate(data_loader):
        if idx%1000==0: print("done / total = %d / %d" %(idx, num_samples))
        z[idx] = data


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    # start writing
    write_event('train/positive', train_pos_loader)
    write_event('valid/positive', valid_pos_loader)
    write_event('train/negative', train_neg_loader)
    write_event('valid/negative', valid_neg_loader)
    write_sequence('train/sequence', train_seq_loader)
    write_sequence('valid/sequence', valid_seq_loader)
    
