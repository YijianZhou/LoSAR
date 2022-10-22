""" Main function for stream picking with CERP
"""
import os, shutil, glob, sys
sys.path.append('/home/zhouyj/software/CERP_TDP/preprocess')
import argparse
import numpy as np
import torch.multiprocessing as mp
from obspy import read, UTCDateTime
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import picker_stream as picker
import config
import warnings
warnings.filterwarnings("ignore")

# import readers
cfg = config.Config()
get_data_dict = cfg.get_data_dict
get_sta_dict = cfg.get_sta_dict
read_data = cfg.read_data


class Pick_One_Day(Dataset):

  def __init__(self, picker, date_list, data_dir, sta_dict, out_root):
    self.picker = picker
    self.date_list = date_list
    self.data_dir = data_dir
    self.sta_dict = sta_dict
    self.out_root = out_root

  def __getitem__(self, index):
    date = self.date_list[index]
    fout = open(os.path.join(self.out_root, '%s.pick'%(date.date)),'w')
    data_dict = get_data_dict(date, self.data_dir)
    for net_sta, data_paths in data_dict.items():
        if net_sta not in self.sta_dict: continue
        print('-'*40)
        print('picking %s %s'%(net_sta, date.date))
        st = read_data(data_paths, self.sta_dict)
        self.picker.pick(st, fout)
    fout.close()

  def __len__(self):
    return len(date_list)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_idx', type=str, default="0")
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--fsta', type=str)
    parser.add_argument('--out_root', type=str)
    parser.add_argument('--time_range', type=str)
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--cnn_ckpt', type=str)
    parser.add_argument('--rnn_ckpt', type=str)
    args = parser.parse_args()
    # setup picker
    picker = picker.CERP_Picker_Stream(args.ckpt_dir, args.cnn_ckpt, args.rnn_ckpt, args.gpu_idx)
    sta_dict = get_sta_dict(args.fsta)
    if not os.path.exists(args.out_root): os.makedirs(args.out_root)
    # start picking 
    start_time, end_time = [UTCDateTime(time) for time in args.time_range.split('-')]
    num_days = int((end_time - start_time) / 86400)
    date_list = [start_time+86400*day_idx for day_idx in range(num_days)]
    dataset = Pick_One_Day(picker, date_list, args.data_dir, sta_dict, args.out_root)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=args.num_workers)
    for i,_ in enumerate(dataloader):
        if i%10==0: print('%s days done'%i) 

