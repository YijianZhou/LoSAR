""" Configure file for SAR_TED
"""
import sys
sys.path.append('/home/zhouyj/software/SAR_TED/preprocess')
import reader

class Config(object):
  def __init__(self):
    # data prep & training sample cut
    self.samp_rate = 100
    self.win_len = 20  # sec
    self.win_stride = 10  # stride for sliding win, sec
    self.num_chn = 3
    self.freq_band = [1,20]
    self.global_max_norm = False
    self.to_prep = True
    self.train_ratio = 0.9
    self.valid_ratio = 0.1  # ratio of samples to cut for training
    self.max_assoc_ratio = 0.5  # neg_cut_ratio = (max_ratio-assoc_ratio)/max_ratio
    self.num_aug = 2  # whether data augment
    self.max_noise = 0.5  # max noise level in pos aug
    self.read_fpha = reader.read_fpha  # import readers
    self.read_fpick = reader.read_fpick
    self.get_data_dict = reader.get_data_dict
    self.get_sta_dict = reader.get_sta_dict
    self.read_data = reader.read_data 
    # SAR model 
    self.rnn_hidden_size = 128
    self.rnn_num_layers = 2
    self.rnn_step_len = 0.5  # in sec
    self.rnn_step_stride = 0.1
    self.rnn_num_steps = int((self.win_len - self.rnn_step_len) / self.rnn_step_stride) + 1
    self.num_att_heads = 4
    # SAR train
    self.num_epochs = 15
    self.batch_size = 128
    self.lr = 1e-4
    self.ckpt_step = 100
    self.summary_step = 20
    # picking config
    self.trig_thres = 0.3
    self.picker_batch_size = 20
    self.tp_dev = 1.5  # merge picks in different sliding win
    self.ts_dev = 1.5
    self.amp_win = [1,6]  # sec pre-P & post-S for amp calc
    self.rm_glitch = True
    self.win_peak = 1
    self.amp_ratio_thres = [5,8,3]  # Peak rm; P/P_tail; P/S
