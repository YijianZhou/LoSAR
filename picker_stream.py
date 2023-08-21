import os, glob
from obspy import UTCDateTime
import time
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import kurtosis
import config
from models import CNN, RNN
import warnings
warnings.filterwarnings("ignore")

# model config
cfg = config.Config()
samp_rate = cfg.samp_rate
num_chn = cfg.num_chn
win_len = cfg.win_len
win_len_npts = int(win_len * samp_rate)
win_stride = cfg.win_stride
win_stride_npts = int(win_stride * samp_rate)
step_len = cfg.step_len
step_len_npts = int(step_len * samp_rate)
step_stride = cfg.step_stride
step_stride_npts = int(step_stride * samp_rate)
num_steps = cfg.num_steps
freq_band = cfg.freq_band
global_max_norm = cfg.global_max_norm
# picker config
to_repick = cfg.to_repick
batch_size = cfg.picker_batch_size
tp_dev = cfg.tp_dev
ts_dev = cfg.ts_dev
p_win_npts = [int(samp_rate*win) for win in cfg.p_win]
s_win_npts = [int(samp_rate*win) for win in cfg.s_win]
win_kurt_npts = [int(win*samp_rate) for win in cfg.win_kurt]
win_sta_npts = [int(win*samp_rate) for win in cfg.win_sta]
win_lta_npts = [int(win*samp_rate) for win in cfg.win_lta]
amp_win = cfg.amp_win
amp_win_npts = int(sum(amp_win)*samp_rate)


class CERP_Picker_Stream(object):
  """ CERP picker for raw stream data
  """
  def __init__(self, ckpt_dir, cnn_ckpt=-1, rnn_ckpt=-1, gpu_idx='0'):
    if cnn_ckpt=='-1':
        cnn_ckpt = max([int(os.path.basename(ckpt).split('_')[0]) for ckpt in glob.glob(os.path.join(ckpt_dir, 'CNN', '*.ckpt'))])
    if rnn_ckpt=='-1':
        rnn_ckpt = max([int(os.path.basename(ckpt).split('_')[0]) for ckpt in glob.glob(os.path.join(ckpt_dir, 'RNN', '*.ckpt'))])
    cnn_ckpt = sorted(glob.glob(os.path.join(ckpt_dir, 'CNN', '%s_*.ckpt'%cnn_ckpt)))[0]
    rnn_ckpt = sorted(glob.glob(os.path.join(ckpt_dir, 'RNN', '%s_*.ckpt'%rnn_ckpt)))[0]
    print('CNN checkpoint: %s'%cnn_ckpt)
    print('RNN checkpoint: %s'%rnn_ckpt)
    # load model
    self.device = torch.device("cuda:%s"%gpu_idx)
    self.model_cnn = CNN()
    self.model_rnn = RNN()
    self.model_cnn.load_state_dict(torch.load(cnn_ckpt, map_location=self.device))
    self.model_rnn.load_state_dict(torch.load(rnn_ckpt, map_location=self.device))
    self.model_cnn.to(self.device)
    self.model_rnn.to(self.device)
    self.model_cnn.eval()
    self.model_rnn.eval()

  def pick(self, stream, fout=None):
    # 1. preprocess stream data & sliding win
    print('1. preprocess stream data & slice into windows')
    t = time.time()
    stream = self.preprocess(stream)
    if len(stream)!=num_chn: return 
    start_time, end_time = stream[0].stats.starttime+win_stride, stream[0].stats.endtime
    if end_time < start_time + win_len: return
    stream = stream.slice(start_time, end_time)
    net, sta = stream[0].stats.network, stream[0].stats.station
    net_sta = '%s.%s'%(net,sta)
    num_win = int((end_time - start_time - win_len) / win_stride) + 1
    st_len_npts = min([len(trace) for trace in stream])
    st_data = np.array([trace.data[0:st_len_npts] for trace in stream], dtype=np.float32)
    st_data_cuda = torch.from_numpy(st_data).cuda(device=self.device)
    # 2. run CERP picker
    det_idx, det_prob = self.run_cnn(st_data_cuda, num_win)
    num_det = len(det_idx)
    picks_raw = self.run_rnn(st_data_cuda, det_idx)
    # 3.1 select picks
    print('3. select & write picks')
    to_drop = []
    for i in range(num_det):
        win_idx = det_idx[i]
        tp, ts = picks_raw[i]
        # if no tp-ts pair, bad
        if tp==-1 or ts==-1: to_drop.append(i); continue
        # if not consecutive det, good
        if i==num_det-1 or det_idx[i+1]!=win_idx+1: continue
        # if not the same p or s det, good
        if abs(tp-win_stride-picks_raw[i+1][0])>tp_dev: continue
        if abs(ts-win_stride-picks_raw[i+1][1])>ts_dev: continue
        # else: use the pick with higher det_prob
        if det_prob[i]<det_prob[i+1]: to_drop.append(i)
        else: to_drop.append(i+1)
    print('  %s picks dropped'%len(to_drop))
    # 3.2 repick & get s_amp
    print('  repick & get s_amp')
    picks = []
    tp_old, ts_old = -1, -1
    for i in range(num_det):
        if i in to_drop: continue
        win_idx = det_idx[i]
        win_t0 = start_time + win_idx*win_stride
        tp, ts = [win_t0+ti for ti in picks_raw[i]]
        if to_repick: 
            tp_sec, ts_sec = self.repick(st_data, tp-start_time, ts-start_time)
            tp, ts = [start_time + t_sec for t_sec in [tp_sec, ts_sec]]
        # get s_amp
        st = stream.slice(tp-amp_win[0], ts+amp_win[1]).copy()
        amp_data = np.array([tr.data[0:amp_win_npts] for tr in st])
        s_amp = self.get_amp(amp_data)
        if tp_old!=-1 and (abs(tp-tp_old)<=tp_dev or abs(ts-ts_old)<=ts_dev): 
            picks[-1] = [net_sta, tp, ts, s_amp, det_prob[i]]
        else: picks.append([net_sta, tp, ts, s_amp, det_prob[i]])
        tp_old, ts_old = tp, ts
    if fout:
        for net_sta, tp, ts, s_amp, det_prob_i in picks:
            fout.write('{},{},{},{},{:.4f}\n'.format(net_sta, tp, ts, s_amp, det_prob_i))
    print('total run time {:.2f}s'.format(time.time()-t))
    return picks

  # 2.1 detect earthquake windows
  def run_cnn(self, st_data_cuda, num_win):
    print('2.1 run CNN for Event detection')
    t = time.time()
    num_batch = int(np.ceil(num_win / batch_size))
    det_idx = torch.tensor([], dtype=torch.int, device=self.device)
    det_prob = torch.tensor([], device=self.device)
    for batch_idx in range(num_batch):
        # get win_data
        n_win = batch_size if batch_idx<num_batch-1 else num_win%batch_size
        if n_win==0: n_win = batch_size
        win_data = torch.zeros([n_win, num_chn, win_len_npts], dtype=torch.float32, device=self.device)
        for i in range(n_win):
            win_idx = i + batch_idx*batch_size
            win_data[i] = st_data_cuda[:,win_idx*win_stride_npts : win_idx*win_stride_npts+win_len_npts].clone()
            win_data[i] = self.preprocess_cuda(win_data[i])
        # run CNN
        pred_logits = self.model_cnn(win_data)
        pred_class = torch.argmax(pred_logits,1)
        pred_prob = F.softmax(pred_logits)[:,1].detach()
        # add det_idx & det_prob
        det_idx_i = torch.where(pred_class==1)[0] + batch_idx*batch_size
        det_prob_i = pred_prob[pred_class==1]
        det_idx = torch.cat((det_idx, det_idx_i.int()))
        det_prob = torch.cat((det_prob, det_prob_i))
    det_idx = det_idx.cpu().numpy()
    det_prob = det_prob.cpu().numpy()
    print('  {} detections | CNN run time {:.2f}s'.format(len(det_idx), time.time()-t))
    return det_idx, det_prob

  # 2.2 pick tp & ts of earthquake window
  def run_rnn(self, st_data_cuda, det_idx):
    print('2.2 run RNN for Phase picking')
    num_win = len(det_idx)
    num_batch = int(np.ceil(num_win / batch_size))
    t = time.time()
    picks = []
    for batch_idx in range(num_batch):
        # get win_data
        win_idx = det_idx[batch_idx*batch_size:(batch_idx+1)*batch_size]
        win_seq = self.st2seq(st_data_cuda, win_idx)
        pred_logits = self.model_rnn(win_seq)
        pred_classes = torch.argmax(pred_logits,2).cpu().numpy()
        # decode to sec
        for pred_class in pred_classes:
            pred_p = np.where(pred_class==1)[0]
            if len(pred_p)>0:
                tp = step_len/2 if pred_p[0]==0 \
                else step_len + step_stride * (pred_p[0]-0.5)
                pred_class[0:pred_p[0]] = 0
            else: tp = -1
            pred_s = np.where(pred_class==2)[0]
            if len(pred_s)>0:
                ts = step_len/2 if pred_s[0]==0 \
                else step_len + step_stride * (pred_s[0]-0.5)
            else: ts = -1
            picks.append([tp, ts])
    print('  {} events picked | RNN run time {:.2f}s'.format(len(picks), time.time()-t))
    return np.array(picks)

  def st2seq(self, st_data_cuda, win_idx):
    num_win = len(win_idx)
    win_seq = torch.zeros((num_win, num_steps, num_chn, step_len_npts), dtype=torch.float32, device=self.device)
    for i,win_i in enumerate(win_idx):
        win_data = st_data_cuda[:,win_i*win_stride_npts : win_i*win_stride_npts+win_len_npts].clone()
        win_data = self.preprocess_cuda(win_data)
        for j in range(num_steps):
            idx0 = j * step_stride_npts
            idx1 = idx0 + step_len_npts
            win_seq[i,j,:] = win_data[:,idx0:idx1]
    return win_seq.view([num_win, num_steps, step_len_npts*num_chn])

  def preprocess(self, st, max_gap=5.):
    # check num_chn
    if len(st)!=num_chn: print('missing trace!'); return []
    # align time
    start_time = max([tr.stats.starttime for tr in st])
    end_time = min([tr.stats.endtime for tr in st])
    if end_time < start_time + win_len: return []
    st = st.slice(start_time, end_time, nearest_sample=True)
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
    # resample data
    st = st.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=5.)
    org_rate = st[0].stats.sampling_rate
    if org_rate!=samp_rate: st.resample(samp_rate)
    for ii in range(3):
        st[ii].data[np.isnan(st[ii].data)] = 0
        st[ii].data[np.isinf(st[ii].data)] = 0
    # filter
    freq_min, freq_max = freq_band
    if freq_min and freq_max:
        return st.filter('bandpass', freqmin=freq_min, freqmax=freq_max)
    elif not freq_max and freq_min: 
        return st.filter('highpass', freq=freq_min)
    elif not freq_min and freq_max: 
        return st.filter('lowpass', freq=freq_max)
    else:
        print('filter type not supported!'); return []

  # preprocess cuda data (in-place)
  def preprocess_cuda(self, data):
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

  def repick(self, st_data, tp0, ts0):
    # extract data
    tp0_idx = int(tp0*samp_rate)
    ts0_idx = int(ts0*samp_rate)
    # 1. repick P
    idx0 = tp0_idx - p_win_npts[0] - win_lta_npts[0]
    idx1 = tp0_idx + min(p_win_npts[1], (ts0_idx-tp0_idx)//2) + win_sta_npts[0]
    if idx0<0 or idx1>st_data.shape[1]: return tp0, ts0
    data_p = st_data[2,idx0:idx1]**2
    cf_p = self.calc_sta_lta(data_p, win_lta_npts[0], win_sta_npts[0])
    tp_idx = np.argmax(cf_p) + idx0
    dt_idx = self.find_first_peak(data_p[0:tp_idx-idx0][::-1])
    tp_idx -= dt_idx
    tp = tp_idx / samp_rate
    # 2. pick S
    # 2.1 get amp_peak
    idx0 = max(tp_idx+(ts0_idx-tp_idx)//2, ts0_idx-s_win_npts[0])
    idx1 = ts0_idx + s_win_npts[1]
    if idx1<=idx0 or idx1>len(st_data[0]): return tp, ts0
    data_s = np.sum(st_data[0:2, idx0:idx1]**2, axis=0)
    ts_min = idx0
    ts_max = min(idx1, idx0 + np.argmax(data_s) + 1)
    if len(st_data[0]) < ts_max + win_sta_npts[1]: return tp, ts0
    # 2.2 long_win kurt --> t_max
    idx0 = ts_min - win_kurt_npts[0]
    idx1 = ts_max
    if min(idx0,idx1)<0 or idx1-idx0<win_kurt_npts[0]: return tp, ts0
    data_s = np.sum(st_data[0:2, idx0:idx1]**2, axis=0)
    data_s /= np.amax(data_s)
    kurt_long = self.calc_kurtosis(data_s, win_kurt_npts[0])
    # 2.3 STA/LTA --> t_min
    idx0 = ts_min - win_lta_npts[1]
    idx1 = ts_max + win_sta_npts[1]
    data_s = np.sum(st_data[0:2, idx0:idx1]**2, axis=0)
    cf_s = self.calc_sta_lta(data_s, win_lta_npts[1], win_sta_npts[1])[win_lta_npts[1]:]
    # 2.4 pick S on short_win kurt
    dt_max = np.argmax(kurt_long)
    dt_max -= self.find_first_peak(kurt_long[0:dt_max+1][::-1])
    dt_min = np.argmax(cf_s) # relative to ts_min
    if dt_min>=dt_max:
        t_data = dt_min + win_lta_npts[1] # time relative to data
        dt_data = self.find_first_peak(data_s[0:t_data][::-1])
        ts_idx = t_data - dt_data + idx0
        ts = ts_idx / samp_rate
        return tp, ts
    idx0 = ts_min + dt_min - win_kurt_npts[1]
    idx1 = ts_min + dt_max
    data_s = np.sum(st_data[0:2, idx0:idx1]**2, axis=0)
    data_s /= np.amax(data_s)
    kurt_short = self.calc_kurtosis(data_s, win_kurt_npts[1])
    kurt_max = np.argmax(kurt_short) if np.argmax(kurt_short)>0 else dt_max-dt_min
    t_data = kurt_max + win_kurt_npts[1] # idx in data domain
    dt_data = self.find_first_peak(data_s[0:t_data][::-1])
    ts_idx = t_data - dt_data + idx0
    ts = ts_idx / samp_rate
    return tp, ts

  # calc STA/LTA for a trace of data (abs or square)
  def calc_sta_lta(self, data, win_lta_npts, win_sta_npts):
    npts = len(data)
    if npts < win_lta_npts + win_sta_npts:
        print('input data too short!')
        return np.zeros(1)
    sta = np.zeros(npts)
    lta = np.ones(npts)
    data_cum = np.cumsum(data)
    sta[:-win_sta_npts] = data_cum[win_sta_npts:] - data_cum[:-win_sta_npts]
    sta /= win_sta_npts
    lta[win_lta_npts:]  = data_cum[win_lta_npts:] - data_cum[:-win_lta_npts]
    lta /= win_lta_npts
    sta_lta = sta/lta
    sta_lta[0:win_lta_npts] = 0.
    sta_lta[np.isinf(sta_lta)] = 0.
    sta_lta[np.isnan(sta_lta)] = 0.
    return sta_lta

  # calc kurtosis trace
  def calc_kurtosis(self, data, win_kurt_npts):
    npts = len(data) - win_kurt_npts + 1
    kurt = np.zeros(npts)
    for i in range(npts):
        kurt[i] = kurtosis(data[i:i+win_kurt_npts])
    return kurt

  def find_first_peak(self, data):
    npts = len(data)
    if npts<2: return 0
    delta_d = data[1:npts] - data[0:npts-1]
    delta_d[np.isnan(delta_d)] = 0
    if min(delta_d)>=0 or max(delta_d)<=0: return 0
    neg_idx = np.where(delta_d<0)[0]
    pos_idx = np.where(delta_d>=0)[0]
    return max(neg_idx[0], pos_idx[0])

  def find_second_peak(self, data):
    npts = len(data)
    if npts<2: return 0
    delta_d = data[1:npts] - data[0:npts-1]
    if min(delta_d)>=0 or max(delta_d)<=0: return 0
    neg_idx = np.where(delta_d<0)[0]
    pos_idx = np.where(delta_d>=0)[0]
    first_peak = max(neg_idx[0], pos_idx[0])
    neg_peak = neg_idx[neg_idx>first_peak]
    pos_peak = pos_idx[pos_idx>first_peak]
    if len(neg_peak)==0 or len(pos_peak)==0: return first_peak
    return max(neg_peak[0], pos_peak[0])

