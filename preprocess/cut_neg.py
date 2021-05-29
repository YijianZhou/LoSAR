""" Cut negative samples (include preprocess)
"""
import os, sys, glob, shutil
sys.path.append('/home/zhouyj/software/data_prep')
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from obspy import read, UTCDateTime
from reader import get_data_dict, read_fpha, read_fsta, dtime2str
from signal_lib import preprocess, calc_dist_km
from sac import obspy_slice
import warnings
warnings.filterwarnings("ignore")

# i/o paths
data_dir = '/data/Continuous_Data'
fsta = 'input/example.sta'
fpha = 'input/example.pha'
out_root = 'output/neg-example'
train_root = os.path.join(out_root,'train')
valid_root = os.path.join(out_root,'valid')
fout_train_paths = os.path.join(out_root,'train_neg.npy')
fout_valid_paths = os.path.join(out_root,'valid_neg.npy')
# cut params
num_workers = 10
samp_rate = 100
win_len = 20
rand_dt = win_len / 2  # rand after P
vp, vs = 6.0, 3.5 # average pha velo
train_ratio, valid_ratio = [0.9, 0.1]
remote_ratio = .7
near_ratio = 0.7
freq_band = [2,40]
global_max_norm = True
to_filter = [True, False][0]
num_aug = 1
lon_min, lon_max = -118, -117.1 # define near-sta
lat_min, lat_max = 35.3, 36.2

def get_sta_date(event_list, sta_dict):
    sta_date_dict = {}
    sta_list = list(sta_dict.keys())
    for i, [event_loc, pha_dict] in enumerate(event_list):
        if i%1e3==0: print('%s/%s events done/total'%(i, len(event_list)))
        # 1. get event info
        ot, lat, lon = event_loc[0:3]
        event_name = dtime2str(ot)
        train_dir = os.path.join(train_root, 'negative', event_name)
        valid_dir = os.path.join(valid_root, 'negative', event_name)
        if not os.path.exists(train_dir): os.makedirs(train_dir)
        if not os.path.exists(valid_dir): os.makedirs(valid_dir)
        for net_sta in sta_list:
            # if is near-source station
            sta_lat, sta_lon = sta_dict[net_sta][0:2]
            if lon_min<sta_lon<lon_max and lat_min<sta_lat<lat_max \
            and net_sta not in pha_dict: continue
            if lon_min<sta_lon<lon_max and lat_min<sta_lat<lat_max:
                if np.random.rand(1)[0]>near_ratio: continue
            else: 
                if np.random.rand(1)[0]>remote_ratio: continue
            # if recorded in fpha
            if net_sta in pha_dict: 
                tp, ts = pha_dict[net_sta]
                is_pick = True
            # if not detected (remote sta)
            else: 
                dist = calc_dist_km([lat,sta_lat],[lon,sta_lon])
                tp, ts = ot+dist/vp, ot+dist/vs
                is_pick = False
            # 2. divide into train / valid
            rand = np.random.rand(1)[0]
            if rand<train_ratio: samp_class = 'train'
            elif rand<train_ratio+valid_ratio: samp_class = 'valid'
            else: continue
            date = str(tp.date)
            sta_date = '%s_%s'%(net_sta, date) # for one day's stream data
            if sta_date not in sta_date_dict: 
                sta_date_dict[sta_date] = [[samp_class, event_name, tp, ts, is_pick]]
            else: sta_date_dict[sta_date].append([samp_class, event_name, tp, ts, is_pick])
    return sta_date_dict


class Negative(Dataset):
  """ Dataset for cutting negative sample
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
    dtype = [('tp','O'),('ts','O')]
    picks = np.array([tuple(sample[-3:-1]) for sample in samples if sample[-1]], dtype=dtype)
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
    for [samp_class, event_name, tp, ts, is_pick] in samples:
        # check if near a real pick
        if sum(abs(picks['tp']-tp)<win_len)>0 and not is_pick: continue
        out_dir = os.path.join(out_root, samp_class, 'negative', event_name)
        samp_name = 'neg_%s_%s_%s'%(net,sta,event_name[:-3])
        n_aug = num_aug if samp_class=='train' else 1
        for aug_idx in range(n_aug):
            # rand time shift
            start_time = tp + np.random.rand(1)[0]*min(rand_dt,2*(ts-tp))
            end_time = start_time + win_len
            # check if tp-ts exists in selected win
            is_tp = (picks['tp']>max(ts, start_time)) * (picks['tp']<end_time)
            is_ts = (picks['ts']>max(ts, start_time)) * (picks['ts']<end_time)
            if sum(is_tp*is_ts)>0: continue
            # slice & prep
            st = obspy_slice(stream, start_time, end_time)
            if 0 in st.max() or len(st)!=3: continue
            st = st.detrend('demean').normalize(global_max=global_max_norm) # note: no detrend here
            # write & record out_paths
            if samp_class=='train': train_paths_i.append([])
            if samp_class=='valid': valid_paths_i.append([])
            for tr in st:
                out_path = os.path.join(out_dir,'%s.%s.%s'%(aug_idx,samp_name,tr.stats.channel))
                tr.stats.sac.t1 = ts-start_time
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
    sta_dict = read_fsta(fsta)
    sta_date_dict = get_sta_date(event_list, sta_dict)
    sta_date_items = list(sta_date_dict.items())
    # for sta-date pair
    train_paths, valid_paths = [], []
    dataset = Negative(sta_date_items)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=None)
    for i, [train_paths_i, valid_paths_i] in enumerate(dataloader): 
        train_paths += train_paths_i
        valid_paths += valid_paths_i
        if i%10==0: print('%s/%s sta-date pairs done/total'%(i+1,len(dataset)))
    np.save(fout_train_paths, train_paths)
    np.save(fout_valid_paths, valid_paths)

