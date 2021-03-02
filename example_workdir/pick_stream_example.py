""" Pick stream data, RC
"""
import os, shutil, glob, sys
sys.path.append('/home/zhouyj/software/CERP_Pytorch')
sys.path.append('/home/zhouyj/software/PAD')
import numpy as np
import torch.multiprocessing as mp
import data_pipeline as dp
from obspy import read, UTCDateTime
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import warnings
warnings.filterwarnings("ignore")

# i/o paths
cdrp_dir = '/home/zhouyj/software/CERP_Pytorch'
fsta = 'input/rc_scsn_pad.sta'
data_root = '/data2/Ridgecrest'
time_range = ['20190704-20190725','20190704-20190715','20190715-20190725'][2]
out_root = 'output/rc_scsn'
if not os.path.exists(out_root): os.makedirs(out_root)
cnn_ckpt_dir = 'output/rc_ckpt/DetNet8'
rnn_ckpt_dir = 'output/rc_ckpt/PpkNet6'
cnn_ckpt_step = [None,15000][0]
rnn_ckpt_step = [None,5000][0]
# picking params
gpu_idx =['0','1'][1]
num_workers = 5
shutil.copyfile('config_rc.py', os.path.join(cdrp_dir, 'config.py'))
import picker_stream as picker
picker = picker.CDRP_Picker_Stream(cnn_ckpt_dir, rnn_ckpt_dir, cnn_ckpt_step, rnn_ckpt_step, gpu_idx)
get_data_dict = dp.get_rc_data
get_sta_dict = dp.get_sta_dict
sta_dict = get_sta_dict(fsta)


class Pick_One_Day(Dataset):

  def __init__(self, date_list):
    self.date_list = date_list

  def __getitem__(self, index):
    date = self.date_list[index]
    fout_pick = open(os.path.join(out_root, '%s.pick'%(date.date)),'w')
    fout_det = open(os.path.join(out_root, '%s.det'%(date.date)),'w')
    data_dict = get_data_dict(date, data_root)
    for net_sta, data_paths in data_dict.items():
        print('-'*40)
        print('picking %s %s'%(net_sta, date.date))
        try:
            st  = read(data_paths[0])
            st += read(data_paths[1])
            st += read(data_paths[2])
        except: continue
        for i in range(3): st[i].data /= float(sta_dict[net_sta]['gain'])
        picker.pick(st, fout_pick, fout_det)
    fout_det.close()
    fout_pick.close()

  def __len__(self):
    return len(date_list)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    start_time, end_time = [UTCDateTime(time) for time in time_range.split('-')]
    num_days = int((end_time - start_time) / 86400)
    date_list = [start_time+86400*day_idx for day_idx in range(num_days)]
    dataset = Pick_One_Day(date_list)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=num_workers)
    for i,_ in enumerate(dataloader):
        if i%10==0: print('%s days done'%i) 

