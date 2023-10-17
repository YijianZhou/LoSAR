""" Cut negative samples
    1. use num of dropped PAL picks to determine the num of neg to cut on each sta-date
    2. rand slice win on that sta-date
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
to_prep = cfg.to_prep
samp_rate = cfg.samp_rate
win_len = cfg.win_len
freq_band = cfg.freq_band
global_max_norm = cfg.global_max_norm
read_fpha = cfg.read_fpha
read_fpick = cfg.read_fpick
get_data_dict = cfg.get_data_dict
train_ratio = cfg.train_ratio
valid_ratio = cfg.valid_ratio
cut_neg_ratio = cfg.cut_neg_ratio


def get_pick_dict(event_list):
    pick_dict = {}
    for i, [_, picks] in enumerate(event_list):
      for net_sta, [tp, ts] in picks.items():
        sta_date = '%s_%s'%(net_sta, tp.date)
        if sta_date not in pick_dict: pick_dict[sta_date] = [[tp,ts]]
        else: pick_dict[sta_date].append([tp, ts])
    return pick_dict

class Negative(Dataset):
  """ Dataset for cutting negative samples
  """
  def __init__(self, pick_num_items, pick_dict, data_dir, out_root):
    self.pick_num_items = pick_num_items
    self.pick_dict =  pick_dict
    self.data_dir= data_dir
    self.out_root = out_root

  def __getitem__(self, index):
    train_paths_i, valid_paths_i = [], []
    # get one sta-date 
    sta_date, num_drop = self.pick_num_items[index]
    net_sta, date = sta_date.split('_')
    data_dict = get_data_dict(UTCDateTime(date), self.data_dir)
    num_cut = int(num_drop * cut_neg_ratio)
    if net_sta not in data_dict or num_cut==0: return train_paths_i, valid_paths_i
    # read stream
    st_paths = data_dict[net_sta]
    try:
        stream  = read(st_paths[0])
        stream += read(st_paths[1])
        stream += read(st_paths[2])
    except: return train_paths_i, valid_paths_i
    if to_prep: stream = preprocess(stream, samp_rate, freq_band)
    if len(stream)!=3: return train_paths_i, valid_paths_i
    rand_dt = stream[0].stats.endtime - stream[0].stats.starttime - win_len
    # get picks
    dtype = [('tp','O'),('ts','O')]
    picks = self.pick_dict[sta_date] if sta_date in self.pick_dict else []
    picks = np.array([(tp,ts) for tp,ts in picks], dtype=dtype)
    # cut event
    for _ in range(num_cut):
        # divide into train / valid
        rand = np.random.rand(1)[0]
        if rand<train_ratio: samp_class = 'train'
        elif rand<train_ratio+valid_ratio: samp_class = 'valid'
        else: continue
        # set out path
        out_dir = os.path.join(self.out_root, samp_class, 'negative', sta_date)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        start_time = stream[0].stats.starttime + win_len/2 + np.random.rand(1)[0]*rand_dt
        end_time = start_time + win_len
        samp_name = 'neg_%s_%s'%(net_sta,dtime2str(start_time))
        # check if tp-ts exists in selected win
        is_tp = (picks['tp']>start_time) * (picks['tp']<end_time)
        is_ts = (picks['ts']>start_time) * (picks['ts']<end_time)
        if sum(is_tp*is_ts)>0: continue
        # cut event window
        st = stream.slice(start_time, end_time).copy()
        st = sac_ch_time(st)
        if 0 in st.max() or len(st)!=3: continue
        st = st.detrend('demean').normalize(global_max=global_max_norm)  # note: no detrend here
        out_paths = [os.path.join(out_dir,'0.%s.%s.sac'%(samp_name,ii+1)) for ii in range(3)]
        for ii,tr in enumerate(st):
            # remove nan & inf
            tr.data[np.isnan(tr.data)] = 0
            tr.data[np.isinf(tr.data)] = 0
            tr.write(out_paths[ii], format='sac')
        # record out_paths
        if samp_class=='train': train_paths_i.append(out_paths)
        if samp_class=='valid': valid_paths_i.append(out_paths)
    return train_paths_i, valid_paths_i

  def __len__(self):
    return len(self.pick_num_items)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--fpha', type=str)
    parser.add_argument('--fpick', type=str)
    parser.add_argument('--out_root', type=str)
    parser.add_argument('--num_workers', type=int)
    args = parser.parse_args()
    # i/o paths
    train_root = os.path.join(args.out_root,'train')
    valid_root = os.path.join(args.out_root,'valid')
    fout_train_paths = os.path.join(args.out_root,'train_neg.npy')
    fout_valid_paths = os.path.join(args.out_root,'valid_neg.npy')
    # read fpha & fpick
    event_list = read_fpha(args.fpha)
    pick_dict = get_pick_dict(event_list)
    pick_num_dict = read_fpick(args.fpick, args.fpha)
    pick_num_items = list(pick_num_dict.items())
    # for dates
    train_paths, valid_paths = [], []
    dataset = Negative(pick_num_items, pick_dict, args.data_dir, args.out_root)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=None)
    for i,[train_paths_i, valid_paths_i] in enumerate(dataloader):
        train_paths += train_paths_i
        valid_paths += valid_paths_i
        if i%100==0: print('%s/%s sta-date pairs done/total'%(i,len(dataset)))
    np.save(fout_train_paths, train_paths)
    np.save(fout_valid_paths, valid_paths)