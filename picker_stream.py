import os, glob
from obspy import UTCDateTime
import time
import torch
import torch.nn.functional as F
import numpy as np
import config
from models import DetNet, PpkNet
import warnings
warnings.filterwarnings("ignore")

# model config
cfg = config.Config()
samp_rate = cfg.samp_rate
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
freq_band = cfg.freq_band
# picker config
batch_size = cfg.picker_batch_size
tp_dev = cfg.tp_dev
ts_dev = cfg.ts_dev
amp_win = cfg.amp_win


class CERP_Picker_Stream(object):
  """ CERP picker for raw stream data
  """
  def __init__(self, cnn_ckpt_dir, rnn_ckpt_dir, cnn_ckpt_step=None, rnn_ckpt_step=None, gpu_idx='0'):
    if not cnn_ckpt_step:
        cnn_ckpt_step = max([int(os.path.basename(ckpt).split('_')[0]) for ckpt in glob.glob(os.path.join(cnn_ckpt_dir, '*.ckpt'))])
    if not rnn_ckpt_step:
        rnn_ckpt_step = max([int(os.path.basename(ckpt).split('_')[0]) for ckpt in glob.glob(os.path.join(rnn_ckpt_dir, '*.ckpt'))])
    cnn_ckpt = sorted(glob.glob(os.path.join(cnn_ckpt_dir, '%s_*.ckpt'%cnn_ckpt_step)))[0]
    rnn_ckpt = sorted(glob.glob(os.path.join(rnn_ckpt_dir, '%s_*.ckpt'%rnn_ckpt_step)))[0]
    print('CNN checkpoint: %s'%cnn_ckpt)
    print('RNN checkpoint: %s'%rnn_ckpt)
    # load model
    self.device = torch.device("cuda:%s"%gpu_idx)
    self.model_det = DetNet()
    self.model_ppk = PpkNet()
    self.model_det.load_state_dict(torch.load(cnn_ckpt, map_location=self.device))
    self.model_ppk.load_state_dict(torch.load(rnn_ckpt, map_location=self.device))
    self.model_det.to(self.device)
    self.model_ppk.to(self.device)
    self.model_det.eval()
    self.model_ppk.eval()


  def pick(self, stream, fout_pick=None, fout_det=None):
    # 1. preprocess stream data & sliding win
    print('1. preprocess stream data & slice into windows')
    t = time.time()
    stream = self.preprocess(stream)
    if len(stream)!=num_chn: return 
    net, sta = stream[0].stats.network, stream[0].stats.station
    net_sta = '%s.%s'%(net,sta)
    start_time, end_time = stream[0].stats.starttime, stream[0].stats.endtime
    if end_time < start_time + win_len: return 
    num_win = int((end_time - start_time - win_len) / win_stride) + 1
    st_len_npts = min([len(trace) for trace in stream])
    st_data = np.array([trace.data for trace in stream], dtype=np.float32)
    st_data = torch.from_numpy(st_data).cuda(device=self.device)
    win_data = torch.zeros([num_win, num_chn, win_len_npts], dtype=torch.float32, device=self.device)
    for win_idx in range(num_win):
        win_data_i = st_data[:,win_idx*win_stride_npts : win_idx*win_stride_npts+win_len_npts]
        win_data[win_idx] = self.preprocess_cuda(win_data_i.clone())
    print('  {} {} | {} windows preprocessed | {:.2f}s'.format(net_sta, start_time.date, num_win, time.time()-t))
    del st_data
    # 2. run DetNet & PpkNet
    det_idx, det_prob = self.run_det(win_data)
    num_det = len(det_idx)
    win_data = win_data[det_idx]
    picks_raw = self.run_ppk(win_data)
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
    # 3.2 get s_amp
    print('  getting s_amp & write picks')
    picks = []
    for i in range(num_det):
        if i in to_drop: continue
        win_idx = det_idx[i]
        win_t0 = start_time + win_idx*win_stride
        tp, ts = [win_t0+ti for ti in picks_raw[i]]
        # get s_amp
        st = stream.slice(tp-amp_win[0], ts+amp_win[1])
        st_data = np.array([tr.data for tr in st])
        s_amp = self.get_amp(st_data)
        picks.append([net_sta, tp, ts, s_amp, det_prob[i]])
        if fout_pick:
            fout_pick.write('{},{},{},{},{:.4f}\n'.format(net_sta, tp, ts, s_amp, det_prob[i]))
        if fout_det:
            fout_det.write('{},{},{},{:.4f}\n'.format(net_sta, win_t0, win_t0+win_len, det_prob[i]))
    print('total run time {:.2f}s'.format(time.time()-t))
    return picks


  # 2.1 detect earthquake windows
  def run_det(self, win_data):
    print('2.1 run DetNet (CNN)')
    t = time.time()
    det_idx = torch.tensor([], dtype=torch.int, device=self.device)
    det_prob = torch.tensor([], device=self.device)
    num_batch = int(np.ceil(win_data.shape[0] / batch_size))
    for batch_idx in range(num_batch):
        # run DetNet
        pred_logits = self.model_det(win_data[batch_idx*batch_size:(batch_idx+1)*batch_size])
        pred_class = torch.argmax(pred_logits,1)
        pred_prob = F.softmax(pred_logits)[:,1].detach()
        # add det_idx & det_prob
        det_idx_i = torch.where(pred_class==1)[0] + batch_idx*batch_size
        det_prob_i = pred_prob[pred_class==1]
        det_idx = torch.cat((det_idx, det_idx_i))
        det_prob = torch.cat((det_prob, det_prob_i))
    det_idx = det_idx.cpu().numpy()
    det_prob = det_prob.cpu().numpy()
    print('  {} detections | CNN run time {:.2f}s'.format(len(det_idx), time.time()-t))
    return det_idx, det_prob


  # 2.2 pick tp & ts of earthquake window
  def run_ppk(self, win_data):
    print('2.2 run PpkNet (RNN)')
    print('  transfer event data into sequences')
    win_seq = self.event2seq(win_data)
    del win_data
    t = time.time()
    picks = []
    num_batch = int(np.ceil(win_seq.shape[0] / batch_size))
    for batch_idx in range(num_batch):
        pred_logits = self.model_ppk(win_seq[batch_idx*batch_size:(batch_idx+1)*batch_size])
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
    del win_seq
    print('  {} events picked | RNN run time {:.2f}s'.format(len(picks), time.time()-t))
    return np.array(picks)

  
  def event2seq(self, win_data):
    t = time.time()
    num_win = win_data.shape[0]
    win_seq = torch.zeros((num_win, num_steps, num_chn, step_len_npts), dtype=torch.float32, device=self.device)
    for i in range(num_win):
      for j in range(num_steps):
        idx0 = j * step_stride_npts
        idx1 = idx0 + step_len_npts
        win_seq[i,j,:] = win_data[i,:,idx0:idx1]
    print('  sequencing time {:.2f}s'.format(time.time()-t))
    return win_seq.view([num_win, num_steps, step_len_npts*num_chn])
  

  def preprocess(self, st):
    # check num_chn
    if len(st)!=num_chn: print('missing trace!'); return []
    # align time
    start_time = max([tr.stats.starttime for tr in st])
    end_time = min([tr.stats.endtime for tr in st])
    if end_time < start_time + win_len: return  []
    st = st.slice(start_time, end_time, nearest_sample=True)
    # resample data
    org_rate = int(st[0].stats.sampling_rate)
    rate = np.gcd(org_rate, int(samp_rate))
    if rate==1: print('warning: bad sampling rate!'); return []
    decim_factor = int(org_rate / rate)
    resamp_factor = int(samp_rate / rate)
    if decim_factor!=1: st = st.decimate(decim_factor)
    if resamp_factor!=1: st = st.interpolate(samp_rate)
    # filter
    st = st.detrend('demean').detrend('linear').taper(max_percentage=0.05, max_length=10.)
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
    data /= torch.max(abs(data), axis=1).values.view(num_chn,1)
    return data


  # get S amplitide
  def get_amp(self, velo):
    # remove mean
    velo -= np.reshape(np.mean(velo, axis=1), [velo.shape[0],1])
    # velocity to displacement
    disp = np.cumsum(velo, axis=1)
    disp /= samp_rate
    return np.amax(np.sum(disp**2, axis=0))**0.5

