""" Make Zarr format dataset with SAC files
"""
import os
import argparse
import zarr
import torch.multiprocessing as mp
import numpy as np
from torch.utils.data import DataLoader
from dataset_sac import Events, Sequences
import config
import warnings
warnings.filterwarnings("ignore")

# model config
cfg = config.Config()
samp_rate = cfg.samp_rate
num_chn = cfg.num_chn
num_steps = cfg.num_steps
win_len = int(cfg.win_len * samp_rate)
step_len = int(cfg.step_len * samp_rate)
step_stride = int(cfg.step_stride * samp_rate)

def write_sequence(zarr_dset, data_loader):
    num_samples = len(data_loader)
    data_shape = (num_samples, num_steps, step_len*num_chn)
    data_chunks = (1, num_steps, step_len*num_chn)
    target_shape = (num_samples, num_steps)
    target_chunks = (1, num_steps)
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
    chunks = (1, num_chn, win_len)
    zarr_out = os.path.join(out_path, zarr_dset)
    print('writing %s'%zarr_out)
    z = zarr.open(zarr_out, mode='w', shape=shape, chunks=chunks, dtype=np.float32)
    for idx, data in enumerate(data_loader):
        if idx%1000==0: print("done / total = %d / %d" %(idx, num_samples))
        z[idx] = data

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--sac_root', type=str)
    parser.add_argument('--num_workers', type=int)
    args = parser.parse_args()
    # i/o paths
    out_path = args.out_path
    train_pos = os.path.join(args.sac_root,'train_pos.npy')
    valid_pos = os.path.join(args.sac_root,'valid_pos.npy')
    train_neg = os.path.join(args.sac_root,'train_neg.npy')
    valid_neg = os.path.join(args.sac_root,'valid_neg.npy')
    # setup dataloader
    train_pos_set = Events(train_pos)
    valid_pos_set = Events(valid_pos)
    train_neg_set = Events(train_neg)
    valid_neg_set = Events(valid_neg)
    train_seq_set = Sequences(train_pos)
    valid_seq_set = Sequences(valid_pos)
    train_pos_loader = DataLoader(train_pos_set, batch_size=None, shuffle=False, num_workers=args.num_workers)
    valid_pos_loader = DataLoader(valid_pos_set, batch_size=None, shuffle=False, num_workers=args.num_workers)
    train_neg_loader = DataLoader(train_neg_set, batch_size=None, shuffle=False, num_workers=args.num_workers)
    valid_neg_loader = DataLoader(valid_neg_set, batch_size=None, shuffle=False, num_workers=args.num_workers)
    train_seq_loader = DataLoader(train_seq_set, batch_size=None, shuffle=False, num_workers=args.num_workers)
    valid_seq_loader = DataLoader(valid_seq_set, batch_size=None, shuffle=False, num_workers=args.num_workers)
    # start writing
    write_event('train/positive', train_pos_loader)
    write_event('valid/positive', valid_pos_loader)
    write_event('train/negative', train_neg_loader)
    write_event('valid/negative', valid_neg_loader)
    write_sequence('train/sequence', train_seq_loader)
    write_sequence('valid/sequence', valid_seq_loader)
