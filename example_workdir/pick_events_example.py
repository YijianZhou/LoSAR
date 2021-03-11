""" Pick event data with CERP_Pytorch
"""
import os, shutil, glob, sys
sys.path.append('/home/zhouyj/software/CERP_Pytorch')
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from obspy import read, UTCDateTime
import warnings
warnings.filterwarnings("ignore")

# i/o paths
cdrp_dir = '/home/zhouyj/software/CERP_Pytorch'
shutil.copyfile('config_example.py', os.path.join(cdrp_dir, 'config.py'))
import picker_event as picker
data_root = '/data3/bigdata/zhouyj/RC_events_scsn'
samples = np.load(os.path.join(data_root, 'data_paths.npy'))
fout = open('output/test.picks','w')
ckpt_dir = 'output/rc_ckpt/PpkNet6'
ckpt_step = [None][0]
# picking params
win_len = 20
time_shift = 10 # rand pre-P
freq_band = [2,40]
to_filter = False
batch_size = 10
num_workers = 0
gpu_idx = ['0','1'][0]
picker = picker.CDRP_Picker_Event(ckpt_dir, ckpt_step, gpu_idx=gpu_idx)


def preprocess(st):
    st = st.detrend('demean')
    if to_filter:
        freq_min, freq_max = freq_band
        if freq_min and freq_max:
            st = st.filter('bandpass', freqmin=freq_min, freqmax=freq_max)
        elif freq_min and not freq_max:
            st = st.filter('highpass', freq=freq_min)
        else: print('filter type not supported')
    return st.normalize()


def read_one_stream(st_paths):
    # read data
    stream  = read(st_paths[0])
    stream += read(st_paths[1])
    stream += read(st_paths[2])
    if len(stream)!=3: return [], None
    stream = preprocess(stream)
    if len(stream)!=3: return [], None
    event_dir, fname = os.path.split(st_paths[0])
    event_name = os.path.basename(event_dir)
    net, sta = fname.split('.')[0:2]
    # get header
    header = stream[0].stats
    start_time = header.starttime
    st_name = '%s_%s.%s'%(event_name,net,sta)
    # rand time shift
    dt =  time_shift * np.random.rand(1)[0]
    tp_target = dt
    ts_target = dt + header.sac.t1 - header.sac.t0
    start_time = start_time + header.sac.t0 - dt
    stream = stream.slice(start_time, start_time+win_len)
    return stream, [st_name, start_time, tp_target, ts_target]


def write_pick(picks, headers, fout):
    num_picks = len(picks)
    for i in range(num_picks):
        tp_pred, ts_pred = picks[i]
        st_name, start_time, tp_target, ts_target = headers[i]
        fout.write('{},{},{:.2f},{:.2f},{:.2f},{:.2f}\n'.format(st_name, start_time, tp_pred, ts_pred, tp_target, ts_target))


class Pick_One_Batch(Dataset):

  def __init__(self, samples):
    self.samples = samples
    num_samples = len(samples)
    self.num_batch = 1 + num_samples // batch_size

  def __getitem__(self, index):
    streams, headers = [], []
    for st_paths in self.samples[index*batch_size:(index+1)*batch_size]:
        stream, header = read_one_stream(st_paths)
        if len(stream)==0: continue
        streams.append(stream)
        headers.append(header)
    picks = picker.pick(streams)
    return picks, headers

  def __len__(self):
    return self.num_batch


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # 'spawn' or 'forkserver'
    dataset = Pick_One_Batch(samples)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=num_workers)
    for i, [picks, headers] in enumerate(dataloader):
        if i%100==0: print('{}/{} batch done/total'.format(i, len(dataset)))
        write_pick(picks.numpy(), headers, fout)
    fout.close()

