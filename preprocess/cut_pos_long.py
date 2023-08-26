""" Cut positive samples for long-term data
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
step_len = cfg.step_len
rand_dt = win_len/2 - step_len  # rand before P
read_fpha = cfg.read_fpha
get_data_dict = cfg.get_data_dict
train_ratio = cfg.train_ratio
valid_ratio = cfg.valid_ratio
freq_band = cfg.freq_band
to_prep = cfg.to_prep
global_max_norm = cfg.global_max_norm
num_aug = cfg.num_aug
max_noise = cfg.max_noise


def add_noise(tr, tp, ts):
    if tp>ts: return tr
    scale = np.random.rand(1)[0] * max_noise * np.std(tr.slice(tp, ts).data)
    tr.data += np.random.normal(loc=np.mean(tr.data), scale=scale, size=len(tr))
    return tr

def cut_event_window(stream_paths, t0, t1, tp, ts, to_aug, out_paths):
    st  = read(stream_paths[0], starttime=t0-win_len/2, endtime=t1+win_len/2)
    st += read(stream_paths[1], starttime=t0-win_len/2, endtime=t1+win_len/2)
    st += read(stream_paths[2], starttime=t0-win_len/2, endtime=t1+win_len/2)
    if 0 in st.max() or len(st)!=3: return False
    if to_prep: st = preprocess(st, samp_rate, freq_band)
    st = st.slice(t0, t1)
    if 0 in st.max() or len(st)!=3: return False
    st = st.detrend('demean').normalize(global_max=global_max_norm)
    st = sac_ch_time(st)
    for ii, tr in enumerate(st): 
        if to_aug: tr = add_noise(tr, tp, ts)
        tr.write(out_paths[ii], format='sac')
        tr = read(out_paths[ii])[0]
        tr.stats.sac.t0, tr.stats.sac.t1 = tp-t0, ts-t0
        tr.write(out_paths[ii], format='sac')
    return True

class Positive(Dataset):
  """ Dataset for cutting positive samples
  """
  def __init__(self, event_list, data_dir, out_root):
    self.event_list = event_list
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
        stream_paths = data_dict[net_sta]
        net, sta = net_sta.split('.')
        # divide into train / valid
        rand = np.random.rand(1)[0]
        if rand<train_ratio: samp_class = 'train'
        elif rand<train_ratio+valid_ratio: samp_class = 'valid'
        else: continue
        out_dir = os.path.join(self.out_root, samp_class, 'positive', event_name)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        samp_name = 'pos_%s_%s_%s'%(net,sta,event_name[:-3])
        # data aug loop
        n_aug = num_aug if samp_class=='train' else 1
        for aug_idx in range(n_aug):
            out_paths = [os.path.join(out_dir,'%s.%s.%s.sac'%(aug_idx,samp_name,ii+1)) for ii in range(3)]
            start_time = tp - step_len - np.random.rand(1)[0] * min(win_len-step_len-(ts-tp), rand_dt)
            end_time = start_time + win_len 
            to_aug = True if aug_idx>0 and max_noise>0 else False
            is_cut = cut_event_window(stream_paths, start_time, end_time, tp, ts, to_aug, out_paths)
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
    fout_train_paths = os.path.join(args.out_root,'train_pos.npy')
    fout_valid_paths = os.path.join(args.out_root,'valid_pos.npy')
    # read fpha
    event_list = read_fpha(args.fpha)
    # for sta-date pairs
    train_paths, valid_paths = [], []
    dataset = Positive(event_list, args.data_dir, args.out_root)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=None)
    for i,[train_paths_i, valid_paths_i] in enumerate(dataloader):
        train_paths += train_paths_i
        valid_paths += valid_paths_i
        if i%50==0: print('%s/%s events done/total'%(i,len(dataset)))
    np.save(fout_train_paths, train_paths)
    np.save(fout_valid_paths, valid_paths)
