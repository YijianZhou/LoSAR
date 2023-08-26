""" Cut negative samples for long-term data
    1. for all events
    2. cut event and preprocess 
"""
import os, glob, shutil
import argparse
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from obspy import read, UTCDateTime
from signal_lib import preprocess, sac_ch_time
from reader import dtime2str
import config
import warnings
warnings.filterwarnings("ignore")

# cut params
cfg = config.Config()
samp_rate = cfg.samp_rate
win_len = cfg.win_len
rand_dt = win_len/2 # rand after P/S
step_len = cfg.step_len
neg_ref = cfg.neg_ref
read_fpha = cfg.read_fpha
get_data_dict = cfg.get_data_dict
train_ratio = cfg.train_ratio
valid_ratio = cfg.valid_ratio
freq_band = cfg.freq_band
to_prep = cfg.to_prep
global_max_norm = cfg.global_max_norm
num_aug = cfg.num_aug

def get_pick_date(event_list):
    pick_date_dict = {}
    for i, [_, pha_dict] in enumerate(event_list):
      for net_sta, [tp, ts] in pha_dict.items():
        date = str(tp.date)
        if date not in pick_date_dict: pick_date_dict[date] = {}
        if net_sta not in pick_date_dict[date]: pick_date_dict[date][net_sta] = []
        pick_date_dict[date][net_sta].append([tp, ts])
    return pick_date_dict

def cut_event_window(stream_paths, t0, t1, ts, out_paths):
    st  = read(stream_paths[0], starttime=t0-win_len/2, endtime=t1+win_len/2)
    st += read(stream_paths[1], starttime=t0-win_len/2, endtime=t1+win_len/2)
    st += read(stream_paths[2], starttime=t0-win_len/2, endtime=t1+win_len/2)
    if 0 in st.max() or len(st)!=3: return False
    if to_prep: st = preprocess(st, samp_rate, freq_band)
    st = st.slice(t0, t1)
    if 0 in st.max() or len(st)!=3: return False
    # check FN
    amax_sec = [np.argmax(abs(tr.data))/samp_rate for tr in st]
    if neg_ref=='P' and min(amax_sec)>win_len/2: return False 
    st = st.detrend('demean').normalize(global_max=global_max_norm)
    st = sac_ch_time(st)
    for ii, tr in enumerate(st): 
        tr.write(out_paths[ii], format='sac')
        tr = read(out_paths[ii])[0]
        tr.stats.sac.t1 = ts-t0
        tr.write(out_paths[ii], format='sac')
    return True

class Negative(Dataset):
  """ Dataset for cutting negative samples
  """
  def __init__(self, event_list, pick_date_dict, data_dir, out_root):
    self.event_list = event_list
    self.pick_date_dict =  pick_date_dict
    self.data_dir= data_dir
    self.out_root = out_root

  def __getitem__(self, index):
    train_paths_i, valid_paths_i = [], []
    # get event info
    event_loc, pick_dict = self.event_list[index]
    ot, lat, lon, dep, mag = event_loc
    event_name = dtime2str(ot)
    data_dict = get_data_dict(ot, self.data_dir)
    # cut event
    for net_sta, [tp, ts] in pick_dict.items():
        if net_sta not in data_dict: continue
        # get picks
        dtype = [('tp','O'),('ts','O')]
        picks = self.pick_date_dict[str(tp.date)][net_sta]
        picks = np.array([(tp,ts) for tp,ts in picks], dtype=dtype)
        stream_paths = data_dict[net_sta]
        net, sta = net_sta.split('.')
        # divide into train / valid
        rand = np.random.rand(1)[0]
        if rand<train_ratio: samp_class = 'train'
        elif rand<train_ratio+valid_ratio: samp_class = 'valid'
        else: continue
        out_dir = os.path.join(self.out_root, samp_class, 'negative', event_name)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        samp_name = 'neg_%s_%s_%s'%(net,sta,event_name[:-3])
        # data aug loop
        n_aug = num_aug if samp_class=='train' else 1
        for aug_idx in range(n_aug):
            out_paths = [os.path.join(out_dir,'%s.%s.%s.sac'%(aug_idx,samp_name,ii+1)) for ii in range(3)]
            if neg_ref=='P': start_time = tp + np.random.rand(1)[0]*min(rand_dt,2*(ts-tp)) + step_len
            elif neg_ref=='S': start_time = ts + np.random.rand(1)[0]*win_len
            end_time = start_time + win_len
            # check if tp-ts exists in selected win
            is_tp = (picks['tp']>max(ts, start_time)) * (picks['tp']<end_time)
            is_ts = (picks['ts']>max(ts, start_time)) * (picks['ts']<end_time)
            if sum(is_tp*is_ts)>0: continue
            is_cut = cut_event_window(stream_paths, start_time, end_time, ts, out_paths)
            if not is_cut: continue
            # record out_paths
            if samp_class=='train': train_paths_i.append(out_paths)
            if samp_class=='valid': valid_paths_i.append(out_paths)
    return train_paths_i, valid_paths_i

  def __len__(self):
    return len(self.event_list)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--fpha', type=str)
    parser.add_argument('--out_root', type=str)
    parser.add_argument('--num_workers', type=int)
    args = parser.parse_args()
    # i/o paths
    train_root = os.path.join(args.out_root,'train')
    valid_root = os.path.join(args.out_root,'valid')
    fout_train_paths = os.path.join(args.out_root,'train_neg.npy')
    fout_valid_paths = os.path.join(args.out_root,'valid_neg.npy')
    # read fpha
    event_list = read_fpha(args.fpha)
    pick_date_dict = get_pick_date(event_list)
    # for sta-date pairs
    train_paths, valid_paths = [], []
    dataset = Negative(event_list, pick_date_dict, args.data_dir, args.out_root)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=None)
    for i,[train_paths_i, valid_paths_i] in enumerate(dataloader):
        train_paths += train_paths_i
        valid_paths += valid_paths_i
        if i%50==0: print('%s/%s events done/total'%(i,len(dataset)))
    np.save(fout_train_paths, train_paths)
    np.save(fout_valid_paths, valid_paths)
