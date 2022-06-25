""" Cut negative samples (include preprocess)
"""
import os, glob, shutil
import argparse
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from obspy import read, UTCDateTime
from reader import dtime2str
from signal_lib import preprocess, obspy_slice
import config
import warnings
warnings.filterwarnings("ignore")

# cut params
cfg = config.Config()
num_workers = cfg.num_workers
samp_rate = cfg.samp_rate
win_len = cfg.win_len
rand_dt = win_len/2 # rand before P
read_fpha = cfg.read_fpha
get_data_dict = cfg.get_data_dict
train_ratio = cfg.train_ratio
valid_ratio = cfg.valid_ratio
freq_band = cfg.freq_band
to_prep = cfg.to_prep
global_max_norm = cfg.global_max_norm
num_aug = cfg.num_aug
neg_ref = cfg.neg_ref
data_format = cfg.data_format

def get_sta_date(event_list):
    sta_date_dict = {}
    for i, [event_loc, pha_dict] in enumerate(event_list):
        if i%1e3==0: print('%s/%s events done/total'%(i, len(event_list)))
        # 1. get event info
        ot, lat, lon = event_loc[0:3]
        event_name = dtime2str(ot)
        train_dir = os.path.join(train_root, 'negative', event_name)
        valid_dir = os.path.join(valid_root, 'negative', event_name)
        if not os.path.exists(train_dir): os.makedirs(train_dir)
        if not os.path.exists(valid_dir): os.makedirs(valid_dir)
        for net_sta, [tp, ts] in pha_dict.items():
            # 2. divide into train / valid
            rand = np.random.rand(1)[0]
            if rand<train_ratio: samp_class = 'train'
            elif rand<train_ratio+valid_ratio: samp_class = 'valid'
            else: continue
            date = str(tp.date)
            sta_date = '%s_%s'%(net_sta, date) # for one day's stream data
            if sta_date not in sta_date_dict:
                sta_date_dict[sta_date] = [[samp_class, event_name, tp, ts]]
            else: sta_date_dict[sta_date].append([samp_class, event_name, tp, ts])
    return sta_date_dict


class Negative(Dataset):
  """ Dataset for cutting negative samples
  """
  def __init__(self, sta_date_items, data_dir, out_root):
    self.sta_date_items = sta_date_items
    self.data_dir = data_dir
    self.out_root = out_root

  def __getitem__(self, index):
    train_paths_i, valid_paths_i = [], []
    # get one sta-date
    sta_date, samples = self.sta_date_items[index]
    net_sta, date = sta_date.split('_')
    net, sta = net_sta.split('.')
    date = UTCDateTime(date)
    dtype = [('tp','O'),('ts','O')]
    picks = np.array([tuple(sample[-2:]) for sample in samples if sample[-1]], dtype=dtype)
    # read & prep one day's data
    print('reading %s %s'%(net_sta, date.date))
    data_dict = get_data_dict(date, self.data_dir)
    if net_sta not in data_dict: return train_paths_i, valid_paths_i
    st_paths = data_dict[net_sta]
    try:
        stream  = read(st_paths[0])
        stream += read(st_paths[1])
        stream += read(st_paths[2])
    except: return train_paths_i, valid_paths_i
    if to_prep: stream = preprocess(stream, samp_rate, freq_band)
    if len(stream)!=3: return train_paths_i, valid_paths_i
    for [samp_class, event_name, tp, ts] in samples:
        out_dir = os.path.join(self.out_root, samp_class, 'negative', event_name)
        samp_name = 'neg_%s_%s_%s'%(net,sta,event_name[:-3])
        n_aug = num_aug if samp_class=='train' else 1
        for aug_idx in range(n_aug):
            # rand time shift 
            if neg_ref=='P': start_time = tp + np.random.rand(1)[0]*min(rand_dt,2*(ts-tp))
            elif neg_ref=='S': start_time = ts + np.random.rand(1)[0]*max(rand_dt,ts-tp)
            end_time = start_time + win_len
            # check if tp-ts exists in selected win
            is_tp = (picks['tp']>max(ts, start_time)) * (picks['tp']<end_time)
            is_ts = (picks['ts']>max(ts, start_time)) * (picks['ts']<end_time)
            if sum(is_tp*is_ts)>0: continue
            # slice & prep
            if data_format=='sac': st = obspy_slice(stream, start_time, end_time)
            else: st = stream.slice(start_time, end_time)
            if 0 in st.max() or len(st)!=3: continue
            st = st.detrend('demean').normalize(global_max=global_max_norm) # note: no detrend here
            # write & record out_paths
            if samp_class=='train': train_paths_i.append([])
            if samp_class=='valid': valid_paths_i.append([])
            for ii,tr in enumerate(st):
                out_path = os.path.join(out_dir,'%s.%s.%s'%(aug_idx,samp_name,ii+1))
                tr.write(out_path, format='sac')
                tr = read(out_path, headonly=True)[0]
                tr.stats.sac.t1 = ts-start_time
                tr.write(out_path, format='sac')
                if samp_class=='train': train_paths_i[-1].append(out_path)
                if samp_class=='valid': valid_paths_i[-1].append(out_path)
    return train_paths_i, valid_paths_i

  def __len__(self):
    return len(self.sta_date_items)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--fpha', type=str)
    parser.add_argument('--out_root', type=str)
    args = parser.parse_args()
    # i/o paths
    train_root = os.path.join(args.out_root,'train')
    valid_root = os.path.join(args.out_root,'valid')
    fout_train_paths = os.path.join(args.out_root,'train_neg.npy')
    fout_valid_paths = os.path.join(args.out_root,'valid_neg.npy')
    # read fpha
    event_list = read_fpha(args.fpha)
    sta_date_dict = get_sta_date(event_list)
    sta_date_items = list(sta_date_dict.items())
    # for sta-date pairs
    train_paths, valid_paths = [], []
    dataset = Negative(sta_date_items, args.data_dir, args.out_root)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=None)
    for i,[train_paths_i, valid_paths_i] in enumerate(dataloader): 
        train_paths += train_paths_i
        valid_paths += valid_paths_i
        if i%10==0: print('%s/%s sta-date pairs done/total'%(i+1,len(dataset)))
    np.save(fout_train_paths, train_paths)
    np.save(fout_valid_paths, valid_paths)

