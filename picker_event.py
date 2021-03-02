import os, glob
import time
import torch
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


class CERP_Picker_Event(object):
  """ CERP picker for event data
  """
  def __init__(self, ckpt_dir, ckpt_step=None, gpu_idx='0'):
    if not ckpt_step:
        ckpt_step = max([int(os.path.basename(ckpt).split('_')[0]) for ckpt in glob.glob(os.path.join(ckpt_dir, '*.ckpt'))])
    ckpt_path = sorted(glob.glob(os.path.join(ckpt_dir, '%s_*.ckpt'%ckpt_step)))[0]
    # load model
    self.device = torch.device("cuda:%s"%gpu_idx)
    self.model = PpkNet()
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
    return np.array(picks)


  def st2seq(self, streams):
    batch_size = len(streams)
    data_holder = np.zeros((batch_size, num_steps, step_len_npts*num_chn), dtype=np.float32)
    for i,stream in enumerate(streams):
        # convert to numpy array
        win_len_npts = min([len(tr) for tr in stream])
        st_data = np.array([trace.data[0:win_len_npts] for trace in stream], dtype=np.float32)
        # feed into holder
        for j in range(num_steps):
            idx0 = j * step_stride_npts
            idx1 = idx0 + step_len_npts
            if idx1>st_data.shape[1]: continue
            step_data = st_data[:, idx0:idx1]
            data_holder[i, j, :] = np.reshape(step_data, [step_len_npts*num_chn])
    return data_holder

