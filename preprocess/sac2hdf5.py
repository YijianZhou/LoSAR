""" Make HDF5 dataset with SAC files
"""
import os
import argparse
import h5py
import torch.multiprocessing as mp
import numpy as np
from torch.utils.data import DataLoader
from dataset_sac import Sequences
import config
import warnings
warnings.filterwarnings("ignore")

# model config
cfg = config.Config()
samp_rate = cfg.samp_rate
num_chn = cfg.num_chn
num_steps = cfg.rnn_num_steps
win_len = int(cfg.win_len * samp_rate)
step_len = int(cfg.rnn_step_len * samp_rate)

def write_sequence(h5_dataset, data_loader):
    num_samples = len(data_loader)
    data_shape = (num_samples, num_steps, step_len*num_chn)
    data_chunks = (1, num_steps, step_len*num_chn)
    target_shape = (num_samples, num_steps)
    target_chunks = (1, num_steps)
    print('writing', h5_dataset)
    h5_data = h5_file.create_dataset(h5_dataset+'_data', shape=data_shape, chunks=data_chunks, dtype='float32')
    h5_target = h5_file.create_dataset(h5_dataset+'_target', shape=target_shape, chunks=target_chunks, dtype='i')
    for idx, (data, target) in enumerate(data_loader):
        if idx%1000==0: print("done / total = %d / %d" %(idx, num_samples))
        h5_data[idx] = data.numpy()
        h5_target[idx] = target.numpy()


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
    train_pos_set = Sequences(train_pos, True)
    valid_pos_set = Sequences(valid_pos, True)
    train_neg_set = Sequences(train_neg, False)
    valid_neg_set = Sequences(valid_neg, False)
    train_pos_loader = DataLoader(train_pos_set, batch_size=None, shuffle=False, num_workers=args.num_workers)
    valid_pos_loader = DataLoader(valid_pos_set, batch_size=None, shuffle=False, num_workers=args.num_workers)
    train_neg_loader = DataLoader(train_neg_set, batch_size=None, shuffle=False, num_workers=args.num_workers)
    valid_neg_loader = DataLoader(valid_neg_set, batch_size=None, shuffle=False, num_workers=args.num_workers)
    # start writing
    with h5py.File(out_path, 'w') as h5_file:
        write_sequence('train_pos', train_pos_loader)
        write_sequence('valid_pos', valid_pos_loader)
        write_sequence('train_neg', train_neg_loader)
        write_sequence('valid_neg', valid_neg_loader)
