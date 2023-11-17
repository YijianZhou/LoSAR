import os, glob
import time
import torch
import torch.nn.functional as F
import numpy as np
from obspy import UTCDateTime
import config
from models import SAR
import warnings
warnings.filterwarnings("ignore")

cfg = config.Config()
# model config
samp_rate = cfg.samp_rate
num_chn = cfg.num_chn
win_len = cfg.win_len
win_len_npts = int(win_len * samp_rate)
win_stride = cfg.win_stride
win_stride_npts = int(win_stride * samp_rate)
step_len = cfg.rnn_step_len
step_len_npts = int(step_len * samp_rate)
step_stride = cfg.rnn_step_stride
step_stride_npts = int(step_stride * samp_rate)
num_steps = cfg.rnn_num_steps
freq_band = cfg.freq_band
global_max_norm = cfg.global_max_norm
# picker config
trig_thres = cfg.trig_thres
batch_size = cfg.picker_batch_size
tp_dev = cfg.tp_dev
ts_dev = cfg.ts_dev
amp_win = cfg.amp_win
amp_win_npts = int(sum(amp_win)*samp_rate)


class SAR_Picker(object):
  """ SAR picker for raw stream data
  """
  def __init__(self, ckpt_dir, ckpt_idx=-1, gpu_idx=0):
    if int(ckpt_idx)==-1:
        ckpt_idx = max([int(os.path.basename(ckpt).split('_')[0]) for ckpt in glob.glob(os.path.join(ckpt_dir, '*.ckpt'))])
    ckpt_idx = sorted(glob.glob(os.path.join(ckpt_dir, '%s_*.ckpt'%ckpt_idx)))[0] 
    print('SAR checkpoint: %s'%ckpt_idx)
    # load model
    self.device = torch.device("cuda:%s"%gpu_idx)
    self.model = SAR()
    self.model.load_state_dict(torch.load(ckpt_idx, map_location=self.device))
    self.model.to(self.device)
    self.model.eval()

  def pick(self, stream, fout=None):
    # 1. preprocess stream data 
    print('1. preprocess stream data')
    t = time.time()
    stream, st_raw = self.preprocess(stream)
    if len(stream)!=num_chn: return 
    start_time, end_time = stream[0].stats.starttime+win_stride, stream[0].stats.endtime
    if end_time < start_time + win_len: return
    stream, st_raw = stream.slice(start_time, end_time), st_raw.slice(start_time, end_time)
    net_sta = '%s.%s'%(stream[0].stats.network, stream[0].stats.station)
    num_win = int((end_time - start_time - win_len) / win_stride) + 1
    st_len_npts = min([len(trace) for trace in stream])
    st_data = np.array([trace.data[0:st_len_npts] for trace in stream], dtype=np.float32)
    st_data_cuda = torch.from_numpy(st_data).cuda(device=self.device)
    # find miss chn
    st_raw_npts = min([len(tr) for tr in st_raw])
    st_raw_data = np.array([tr.data[0:st_raw_npts] for tr in st_raw])
    raw_stride = int(st_raw[0].stats.sampling_rate * win_stride)
    raw_win_npts = int(st_raw[0].stats.sampling_rate * win_len)
    miss_chn = np.array([np.sum(st_raw_data[:, i*raw_stride : i*raw_stride+raw_win_npts]==0, axis=1)>win_len_npts/2 for i in range(num_win)])
    # 2. run SAR picker
    picks_raw = self.run_sar(st_data_cuda, start_time, num_win, miss_chn)
    picks_raw = np.sort(picks_raw, order=['tp','ts'])
    num_picks = len(picks_raw)
    # 3.1 select picks
    print('3. select & write picks')
    to_drop = []
    for ii in range(1,num_picks):
        if abs(picks_raw['tp'][ii] - picks_raw['tp'][ii-1]) > tp_dev: continue
        if abs(picks_raw['ts'][ii] - picks_raw['ts'][ii-1]) > ts_dev: continue
        prob_current = np.mean([picks_raw['p_prob'][ii], picks_raw['s_prob'][ii]])
        prob_old = np.mean([picks_raw['p_prob'][ii-1], picks_raw['s_prob'][ii-1]])
        if prob_current>prob_old: to_drop.append(ii-1)
        else: to_drop.append(ii)
    picks_raw = np.delete(picks_raw, to_drop)
    print('  %s picks dropped'%len(to_drop))
    # 3.2 get s_amp & write fout
    print('  get s_amp & write fout')
    picks = []
    for [tp, ts, p_prob, s_prob] in picks_raw:
        st = stream.slice(tp-amp_win[0], ts+amp_win[1]).copy()
        amp_data = np.array([tr.data[0:amp_win_npts] for tr in st])
        s_amp = self.get_amp(amp_data)
        picks.append([net_sta, tp, ts, s_amp, p_prob, s_prob])
        if fout:
            fout.write('{},{},{},{},{:.2f},{:.2f}\n'.format(net_sta, tp, ts, s_amp, p_prob, s_prob))
    print('total run time {:.2f}s'.format(time.time()-t))
    if not fout: return picks

  def run_sar(self, st_data_cuda, start_time, num_win, miss_chn):
    print('2. run SAR for phase picking')
    t = time.time()
    num_batch = int(np.ceil(num_win / batch_size))
    picks_raw = []
    dtype = [('tp','O'),('ts','O'),('p_prob','O'),('s_prob','O')]
    for batch_idx in range(num_batch):
        # get win_data
        n_win = batch_size if batch_idx<num_batch-1 else num_win%batch_size
        if n_win==0: n_win = batch_size
        win_idx_list = [ii + batch_idx*batch_size for ii in range(n_win)]
        win_data_batch = self.st2win(st_data_cuda, win_idx_list, miss_chn)
        pred_logits = self.model(win_data_batch)
        pred_probs = F.softmax(pred_logits, dim=-1).detach().cpu().numpy()
        # decode to sec
        for nn, pred_prob in enumerate(pred_probs):
            win_idx = nn + batch_idx*batch_size
            t0 = start_time + win_idx * win_stride
            if sum(miss_chn[win_idx])==3: continue
            pred_prob_p, pred_prob_s = pred_prob[:,1], pred_prob[:,2]
            pred_prob_p[np.isnan(pred_prob_p)] = 0
            pred_prob_s[np.isnan(pred_prob_s)] = 0
            if min(np.amax(pred_prob_p), np.amax(pred_prob_s)) < trig_thres: continue
            p_idxs = np.where(pred_prob_p>=trig_thres)[0]
            s_idxs = np.where(pred_prob_s>=trig_thres)[0]
            p_dets = np.split(p_idxs, np.where(np.diff(p_idxs)!=1)[0] + 1)
            s_dets = np.split(s_idxs, np.where(np.diff(s_idxs)!=1)[0] + 1)
            p_probs = [np.amax(pred_prob_p[p_det]) for p_det in p_dets]
            s_probs = [np.amax(pred_prob_s[s_det]) for s_det in s_dets]
            p_idxs = [np.median(x) for x in p_dets]
            s_idxs = [np.median(x) for x in s_dets]
            for ii, p_idx in enumerate(p_idxs):
                tp = t0 + step_len/2 + step_stride*p_idx
                p_prob = p_probs[ii]
                for jj, s_idx in enumerate(s_idxs):
                    ts = t0 + step_len/2 + step_stride*s_idx
                    s_prob = s_probs[jj]
                    if ts>tp: picks_raw.append((tp, ts, p_prob, s_prob))
    print('  {} raw P&S picks | SAR run time {:.2f}s'.format(len(picks_raw), time.time()-t))
    return np.array(picks_raw, dtype=dtype)

  def st2win(self, st_data_cuda, win_idx_list, miss_chn):
    num_win = len(win_idx_list)
    win_data_batch = torch.zeros((num_win, num_chn, win_len_npts), dtype=torch.float32, device=self.device)
    for i,win_idx in enumerate(win_idx_list):
        win_data = st_data_cuda[:,win_idx*win_stride_npts : win_idx*win_stride_npts+win_len_npts].clone()
        win_data_batch[i] = self.preprocess_cuda(win_data, miss_chn[win_idx])
    return win_data_batch

  def preprocess(self, st, max_gap=5.):
    # align time
    if len(st)!=num_chn: return [], []
    start_time = max([tr.stats.starttime for tr in st])
    end_time = min([tr.stats.endtime for tr in st])
    if end_time < start_time + win_len: return [], []
    st = st.slice(start_time, end_time, nearest_sample=True)
    if len(st)!=num_chn: return [], []
    # remove nan & inf
    for ii in range(3):
        st[ii].data[np.isnan(st[ii].data)] = 0
        st[ii].data[np.isinf(st[ii].data)] = 0
    if max(st.max())==0: return [], []
    st_raw = st.copy()
    # fill data gap
    max_gap_npts = int(max_gap*samp_rate)
    for tr in st:
        npts = len(tr.data)
        data_diff = np.diff(tr.data)
        gap_idx = np.where(data_diff==0)[0]
        gap_list = np.split(gap_idx, np.where(np.diff(gap_idx)!=1)[0] + 1)
        gap_list = [gap for gap in gap_list if len(gap)>=10]
        num_gap = len(gap_list)
        for ii,gap in enumerate(gap_list):
            idx0, idx1 = max(0, gap[0]-1), min(npts-1, gap[-1]+1)
            if ii<num_gap-1: idx2 = min(idx1+(idx1-idx0), idx1+max_gap_npts, gap_list[ii+1][0])
            else: idx2 = min(idx1+(idx1-idx0), idx1+max_gap_npts, npts-1)
            if idx1==idx2: continue
            if idx2==idx1+(idx1-idx0): tr.data[idx0:idx1] = tr.data[idx1:idx2]
            else:
                num_tile = int(np.ceil((idx1-idx0)/(idx2-idx1)))
                tr.data[idx0:idx1] = np.tile(tr.data[idx1:idx2], num_tile)[0:idx1-idx0]
    # resample 
    st = st.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=5.)
    org_rate = st[0].stats.sampling_rate
    if org_rate!=samp_rate: st.resample(samp_rate)
    # filter
    freq_min, freq_max = freq_band
    if freq_min and freq_max:
        return st.filter('bandpass', freqmin=freq_min, freqmax=freq_max), st_raw
    elif not freq_max and freq_min: 
        return st.filter('highpass', freq=freq_min), st_raw
    elif not freq_min and freq_max: 
        return st.filter('lowpass', freq=freq_max), st_raw
    else:
        print('filter type not supported!'); return [], []

  # preprocess cuda data (in-place)
  def preprocess_cuda(self, data, is_miss):
    # fix missed channel
    if 0<sum(is_miss)<3: data[is_miss] = data[~is_miss][-1]
    # rmean & norm
    data -= torch.mean(data, axis=1).view(num_chn,1)
    if global_max_norm: data /= torch.max(abs(data))
    else: data /= torch.max(abs(data), axis=1).values.view(num_chn,1)
    return data

  # get S amplitide
  def get_amp(self, velo):
    velo -= np.reshape(np.mean(velo, axis=1), [velo.shape[0],1])
    disp = np.cumsum(velo, axis=1)
    disp /= samp_rate
    return np.amax(np.sum(disp**2, axis=0))**0.5
