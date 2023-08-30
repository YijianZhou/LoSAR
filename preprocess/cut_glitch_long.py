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
rand_dt = win_len/2 # rand before P
read_fpick = cfg.read_fpick
get_data_dict = cfg.get_data_dict
train_ratio = cfg.train_ratio
valid_ratio = cfg.valid_ratio
freq_band = cfg.freq_band
to_prep = cfg.to_prep
global_max_norm = cfg.global_max_norm
glitch_ratio = cfg.glitch_ratio


def cut_event_window(stream_paths, t0, t1, tp, ts, out_paths):
    try:
        st  = read(stream_paths[0], starttime=t0-win_len/2, endtime=t1+win_len/2)
        st += read(stream_paths[1], starttime=t0-win_len/2, endtime=t1+win_len/2)
        st += read(stream_paths[2], starttime=t0-win_len/2, endtime=t1+win_len/2)
    except: return False
    if 0 in st.max() or len(st)!=3: return False
    if to_prep: st = preprocess(st, samp_rate, freq_band)
    st = st.slice(t0, t1)
    if 0 in st.max() or len(st)!=3: return False
    st = st.detrend('demean').normalize(global_max=global_max_norm)
    st = sac_ch_time(st)
    for ii, tr in enumerate(st):
        tr.write(out_paths[ii], format='sac')
        tr = read(out_paths[ii])[0]
        tr.stats.sac.t0, tr.stats.sac.t1 = tp-t0, ts-t0
        tr.write(out_paths[ii], format='sac')
    return True

class Glitch(Dataset):
  """ Dataset for cutting glitch samples
  """
  def __init__(self, pick_items, data_dir, out_root):
    self.pick_items = pick_items
    self.data_dir= data_dir
    self.out_root = out_root

  def __getitem__(self, index):
    train_paths_i, valid_paths_i = [], []
    # get event info
    date, picks = self.pick_items[index]
    data_dict = get_data_dict(UTCDateTime(date), self.data_dir)
    # cut event
    for [net_sta, tp, ts] in picks:
        if net_sta not in data_dict: continue
        stream_paths = data_dict[net_sta]
        net, sta = net_sta.split('.')
        # divide into train / valid
        rand = np.random.rand(1)[0]
        if rand<train_ratio: samp_class = 'train'
        elif rand<train_ratio+valid_ratio: samp_class = 'valid'
        else: continue
        out_dir = os.path.join(self.out_root, samp_class, 'glitch', date)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        samp_name = 'glitch_%s_%s'%(net_sta,dtime2str(tp))
        # cut event window
        out_paths = [os.path.join(out_dir,'0.%s.%s.sac'%(samp_name,ii+1)) for ii in range(3)]
        start_time = tp - np.random.rand(1)[0]*rand_dt
        end_time = start_time + win_len
        is_cut = cut_event_window(stream_paths, start_time, end_time, tp, ts, out_paths)
        if not is_cut: continue
        # record out_paths
        if samp_class=='train': train_paths_i.append(out_paths)
        if samp_class=='valid': valid_paths_i.append(out_paths)
    return train_paths_i, valid_paths_i

  def __len__(self):
    return len(self.pick_items)


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
    fout_train_paths = os.path.join(args.out_root,'train_glitch.npy')
    fout_valid_paths = os.path.join(args.out_root,'valid_glitch.npy')
    # read fpick
    pick_dict = read_fpick(args.fpick, args.fpha, glitch_ratio)
    pick_items = list(pick_dict.items())
    # for dates
    train_paths, valid_paths = [], []
    dataset = Glitch(pick_items, args.data_dir, args.out_root)
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=None)
    for i,[train_paths_i, valid_paths_i] in enumerate(dataloader):
        train_paths += train_paths_i
        valid_paths += valid_paths_i
        if i%500==0: print('%s/%s days done/total'%(i,len(dataset)))
    np.save(fout_train_paths, train_paths)
    np.save(fout_valid_paths, valid_paths)
