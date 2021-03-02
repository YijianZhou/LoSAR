""" Cut negative samples (include preprocess)
"""
import os, sys, glob, shutil
sys.path.append('/home/zhouyj/software/data_prep')
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from obspy import read, UTCDateTime
from reader import get_rc_data, read_rc_pha, read_pad_fsta, dtime2str
from signal_lib import preprocess, calc_dist_km
from sac import obspy_slice
import warnings
warnings.filterwarnings("ignore")

# i/o paths
data_dir = '/data2/Ridgecrest'
fpha = 'input/rc_scsn.pha'
#fpha = 'input/rc_pad_hyp.pha'
fsta = 'input/rc_scsn_pad.sta'
out_root = '/data3/bigdata/zhouyj/RC_train/neg-scsn_win-20s_freq-2-40hz'
#out_root = '/data3/bigdata/zhouyj/RC_train/neg-pad_win-20s_freq-2-40hz'
train_root = os.path.join(out_root,'train')
valid_root = os.path.join(out_root,'valid')
fout_train_paths = os.path.join(out_root,'train_neg.npy')
fout_valid_paths = os.path.join(out_root,'valid_neg.npy')
# cut params
num_workers = 10
samp_rate = 100
win_len = 20
rand_dt = 10 # rand after P
vp, vs = 6.0, 3.5 # average pha velo
read_fsta = read_pad_fsta
read_fpha = read_rc_pha
get_data_dict = get_rc_data
train_ratio, valid_ratio = 0.9, 0.1
remote_ratio = 0.5
freq_band = [2,40]
to_filter = [True, False][0]
num_aug = 1

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
            # if recorded in fpha
            if net_sta in pha_dict: 
                tp, ts = pha_dict[net_sta]
                is_pick = True
            # if not detected (remote sta)
            else: 
                sta_lat, sta_lon = sta_dict[net_sta][0:2]
                dist = calc_dist_km([lat,sta_lat],[lon,sta_lon])
                tp, ts = ot+dist/vp, ot+dist/vs
                if np.random.rand(1)[0]>remote_ratio: continue
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
    for [samp_class, event_name, tp, ts, _] in samples:
        out_dir = os.path.join(out_root, samp_class, 'negative', event_name)
        samp_name = 'neg_%s_%s_%s'%(net,sta,event_name[:-3])
        n_aug = num_aug if samp_class=='train' else 1
        for aug_idx in range(n_aug):
            # rand time shift
            start_time = tp + np.random.rand(1)[0]*rand_dt
            end_time = start_time + win_len
            # check if tp-ts exists in selected win
            is_tp = (picks['tp']>max(ts, start_time)) * (picks['tp']<end_time)
            is_ts = (picks['ts']>max(ts, start_time)) * (picks['ts']<end_time)
            if sum(is_tp*is_ts)>0: continue
            # slice & prep
            st = obspy_slice(stream, start_time, end_time)
            if 0 in st.max() or len(st)!=3: continue
            st = st.detrend('demean').normalize() # note: no detrend here
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
    for i,[train_paths_i, valid_paths_i] in enumerate(dataloader): 
        train_paths += train_paths_i
        valid_paths += valid_paths_i
        if i%10==0: print('%s/%s sta-date pairs done/total'%(i+1,len(dataset)))
    np.save(fout_train_paths, train_paths)
    np.save(fout_valid_paths, valid_paths)

