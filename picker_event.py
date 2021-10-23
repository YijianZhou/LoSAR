import os, glob
import time
import torch
import numpy as np
from scipy.stats import kurtosis
import config
from models import PhaseNet
import warnings
warnings.filterwarnings("ignore")

# model config
cfg = config.Config()
samp_rate = cfg.samp_rate
global_max_norm = cfg.global_max_norm
num_chn = cfg.num_chn
win_len = cfg.win_len
win_len_npts = int(win_len * samp_rate)
win_stride = win_len / 2
win_stride_npts = int(win_stride * samp_rate)
step_len = cfg.step_len
step_len_npts = int(step_len * samp_rate)
step_stride = cfg.step_stride
step_stride_npts = int(step_stride * samp_rate)
num_steps = cfg.num_steps
to_repick = cfg.to_repick
p_win_npts = [int(win*samp_rate) for win in cfg.p_win]
s_win_npts = [int(win*samp_rate) for win in cfg.s_win]
win_kurt_npts = [int(win*samp_rate) for win in cfg.win_kurt]
win_sta_npts = [int(win*samp_rate) for win in cfg.win_sta]
win_lta_npts = [int(win*samp_rate) for win in cfg.win_lta]


class CERP_Picker_Event(object):
  """ CERP picker for event data
  """
  def __init__(self, ckpt_dir, ckpt_step=None, gpu_idx='0'):
    if not ckpt_step:
        ckpt_step = max([int(os.path.basename(ckpt).split('_')[0]) for ckpt in glob.glob(os.path.join(ckpt_dir, '*.ckpt'))])
    ckpt_path = sorted(glob.glob(os.path.join(ckpt_dir, '%s_*.ckpt'%ckpt_step)))[0]
    # load model
    self.device = torch.device("cuda:%s"%gpu_idx)
    self.model = PhaseNet()
    self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
    self.model.to(self.device)
    self.model.eval()


  def pick(self, streams):
    picks = []
    # get data batch
    data_batch = self.st2seq(streams)
    data_batch = torch.from_numpy(data_batch).cuda(device=self.device)
    # prediction
    pred_logits = self.model(data_batch)
    pred_classes = torch.argmax(pred_logits,2).cpu().numpy()
    # decode to sec
    for i,pred_class in enumerate(pred_classes):
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
        if to_repick and tp!=-1 and ts!=-1: tp, ts = self.repick(streams[i], tp, ts)
        picks.append([tp, ts])
    return np.array(picks)


  def st2seq(self, streams):
    batch_size = len(streams)
    data_holder = np.zeros((batch_size, num_steps, step_len_npts*num_chn), dtype=np.float32)
    for i,stream in enumerate(streams):
        # convert to numpy array
        win_len_npts = min([len(tr) for tr in stream])
        st_data = np.array([trace.data[0:win_len_npts] for trace in stream], dtype=np.float32)
        st_data = self.preprocess(st_data)
        # feed into holder
        for j in range(num_steps):
            idx0 = j * step_stride_npts
            idx1 = idx0 + step_len_npts
            if idx1>st_data.shape[1]: continue
            step_data = st_data[:, idx0:idx1]
            data_holder[i, j, :] = np.reshape(step_data, [step_len_npts*num_chn])
    return data_holder


  def preprocess(self, data):
    data -= np.reshape(np.mean(data, axis=1), [num_chn,1])
    if global_max_norm: data /= np.amax(abs(data))
    else: data /= np.reshape(np.amax(abs(data), axis=1), [num_chn,1])
    return data


  def repick(self, stream, tp0, ts0):
    # extract data
    min_npts = min([len(trace) for trace in stream])
    st_data = np.array([trace.data[0:min_npts] for trace in stream])
    tp0_idx = int(samp_rate * tp0)
    ts0_idx = int(samp_rate * ts0)
    # 1. repick P
    p_idx0 = tp0_idx - p_win_npts[0] - win_lta_npts[0]
    p_idx1 = tp0_idx + min(p_win_npts[1], int((ts0_idx-tp0_idx)/2)) + win_sta_npts[0]
    if p_idx0<0 or p_idx1>st_data.shape[1]: return tp0, ts0
    data_p = st_data[2,p_idx0:p_idx1]**2
    cf_p = self.calc_sta_lta(data_p, win_lta_npts[0], win_sta_npts[0])
    p_idx = np.argmax(cf_p) + p_idx0
    dt_idx = self.find_first_peak(data_p[0:p_idx-p_idx0][::-1])
    tp_idx = p_idx - dt_idx
    tp = tp_idx / samp_rate
    # 2. pick S
    # 2.1 get amp_peak
    s_idx0 = max(tp_idx+(ts0_idx-tp_idx)//2, ts0_idx-s_win_npts[0])
    s_idx1 = ts0_idx + s_win_npts[1]
    if s_idx1<=s_idx0 or s_idx1>len(st_data[0]): return tp, ts0
    data_s = np.sum(st_data[0:2, s_idx0:s_idx1]**2, axis=0)
    ts_min = s_idx0
    ts_max = min(s_idx1, s_idx0 + np.argmax(data_s) + 1)
    if len(st_data[0]) < ts_max + win_sta_npts[1]: return tp, ts0
    # 2.2 long_win kurt --> t_max
    s_idx0 = ts_min - win_kurt_npts[0]
    s_idx1 = ts_max
    if min(s_idx0,s_idx1)<0 or s_idx1-s_idx0<win_kurt_npts[0]: return tp, ts0
    data_s = np.sum(st_data[0:2, s_idx0:s_idx1]**2, axis=0)
    data_s /= np.amax(data_s)
    kurt_long = self.calc_kurtosis(data_s, win_kurt_npts[0])
    # 2.3 STA/LTA --> t_min
    s_idx0 = ts_min - win_lta_npts[1]
    s_idx1 = ts_max + win_sta_npts[1]
    data_s = np.sum(st_data[0:2, s_idx0:s_idx1]**2, axis=0)
    cf_s = self.calc_sta_lta(data_s, win_lta_npts[1], win_sta_npts[1])[win_lta_npts[1]:]
    # 2.4 pick S on short_win kurt
    dt_max = np.argmax(kurt_long)
    dt_max -= self.find_first_peak(kurt_long[0:dt_max+1][::-1])
    dt_min = np.argmax(cf_s)
    if dt_min>=dt_max:
        t_data = dt_min + win_lta_npts[1]
        dt_data = self.find_first_peak(data_s[0:t_data][::-1])
        s_idx = t_data - dt_data + s_idx0
        ts = s_idx / samp_rate
        return tp, ts
    s_idx0 = ts_min + dt_min - win_kurt_npts[1]
    s_idx1 = ts_min + dt_max
    data_s = np.sum(st_data[0:2, s_idx0:s_idx1]**2, axis=0)
    data_s /= np.amax(data_s)
    kurt_short = self.calc_kurtosis(data_s, win_kurt_npts[1])
    kurt_max = np.argmax(kurt_short) if np.argmax(kurt_short)>0 else dt_max-dt_min
    t_data = kurt_max + win_kurt_npts[1] # idx in data domain
    dt_data = self.find_first_peak(data_s[0:t_data][::-1])
    s_idx = t_data - dt_data + s_idx0
    ts = s_idx / samp_rate
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

