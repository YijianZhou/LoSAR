""" Cut positive samples (include preprocess)
"""
import os, sys, glob, shutil
sys.path.append('/home/zhouyj/software/data_prep')
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from obspy import read, UTCDateTime
from reader import get_data_dict, read_fpha, dtime2str
from signal_lib import preprocess
from sac import obspy_slice
import warnings
warnings.filterwarnings("ignore")

# i/o paths
data_dir = '/data/Continuous_Data'
fpha = 'input/example.pha'
out_root = 'output/pos-example'
train_root = os.path.join(out_root,'train')
valid_root = os.path.join(out_root,'valid')
fout_train_paths = os.path.join(out_root,'train_pos.npy')
fout_valid_paths = os.path.join(out_root,'valid_pos.npy')
# cut params
num_workers = 10
samp_rate = 100
win_len = 20
rand_dt = win_len / 2 # rand before P
train_ratio, valid_ratio = 0.9, 0.1
freq_band = [[2,40],[1,40]][1]
to_filter = [True, False][0]
global_max_norm = True
num_aug = 3
max_noise = [0,0.8][1] # n times P std
noise_win = 5 # sec


def get_sta_date(event_list):
    sta_date_dict = {}
    for i, [event_loc, pha_dict] in enumerate(event_list):
        if i%1e3==0: print('%s/%s events done/total'%(i, len(event_list)))
        # 1. get event info
        ot, lat, lon = event_loc[0:3]
        event_name = dtime2str(ot)
        train_dir = os.path.join(train_root, 'positive', event_name)
        valid_dir = os.path.join(valid_root, 'positive', event_name)
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


def add_noise(tr, tp, ts):
    if tp>ts: return tr
    scale = np.random.rand(1)[0] * max_noise * np.std(tr.slice(tp, ts).data)
    tr.data += np.random.normal(loc=np.mean(tr.data), scale=scale, size=len(tr))
    return tr


class Positive(Dataset):
  """ Dataset for cutting positive samples
  """
  def __init__(self, sta_date_items):
    self.sta_date_items = sta_date_items

  def __getitem__(self, index):
    train_paths_i, valid_paths_i = [], []
    # get one sta-date
    sta_date, samples = self.sta_date_items[index]
    net_sta, date = sta_date.split('_')
    net, sta = net_sta.split('.')
    date = UTCDateTime(date)
    # read & prep one day's data
    print('reading %s %s'%(net_sta, date.date))
    data_dict = get_data_dict(date, data_dir)
    if net_sta not in data_dict: return train_paths_i, valid_paths_i
    st_paths = data_dict[net_sta]
    try:
        stream  = read(st_paths[0])
        stream += read(st_paths[1])
        stream += read(st_paths[2])
    except: return train_paths_i, valid_paths_i
    if to_filter: stream = preprocess(stream, samp_rate, freq_band)
    if len(stream)!=3: return train_paths_i, valid_paths_i
    for [samp_class, event_name, tp, ts] in samples:
        out_dir = os.path.join(out_root, samp_class, 'positive', event_name)
        samp_name = 'pos_%s_%s_%s'%(net,sta,event_name[:-3])
        n_aug = num_aug if samp_class=='train' else 1
        for aug_idx in range(n_aug):
            # rand time shift & prep
            start_time = tp - np.random.rand(1)[0]*rand_dt
            end_time = start_time + win_len
            st = obspy_slice(stream, start_time, end_time)
            if 0 in st.max() or len(st)!=3: continue
            st = st.detrend('demean').normalize(global_max=global_max_norm) # note: no detrend here
            # write & record out_paths
            if samp_class=='train': train_paths_i.append([])
            if samp_class=='valid': valid_paths_i.append([])
            for tr in st:
                if aug_idx>0 and max_noise>0: tr = add_noise(tr, tp, ts)
                out_path = os.path.join(out_dir,'%s.%s.%s'%(aug_idx,samp_name,tr.stats.channel))
                tr.stats.sac.t0, tr.stats.sac.t1 = tp-start_time, ts-start_time
                tr.write(out_path, format='sac')
                if samp_class=='train': train_paths_i[-1].append(out_path)
                if samp_class=='valid': valid_paths_i[-1].append(out_path)
    return train_paths_i, valid_paths_i

  def __len__(self):
    return len(self.sta_date_items)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    # read fpha
    event_list = read_fpha(fpha)
    sta_date_dict = get_sta_date(event_list)
    sta_date_items = list(sta_date_dict.items())
    # for sta-date pairs
    train_paths, valid_paths = [], []
    dataset = Positive(sta_date_items)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=None)
    for i,[train_paths_i, valid_paths_i] in enumerate(dataloader): 
        train_paths += train_paths_i
        valid_paths += valid_paths_i
        if i%10==0: print('%s/%s sta-date pairs done/total'%(i+1,len(dataset)))
    np.save(fout_train_paths, train_paths)
    np.save(fout_valid_paths, valid_paths)

