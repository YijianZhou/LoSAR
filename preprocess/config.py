import sys
sys.path.append('/home/zhouyj/software/RSeL_TED/preprocess')
import reader

class Config(object):
  """ Configure file for RSeL_TED
  """
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
    self.cut_neg_ratio = 0.1  # ratio of neg (dropped PAL pick) to cut
    self.num_aug = 2  # whether data augment
    self.max_noise = 0.5  # max noise level in pos aug
    self.read_fpha = reader.read_fpha # import readers
    self.read_fpick = reader.read_fpick
    self.get_data_dict = reader.get_data_dict
    self.get_sta_dict = reader.get_sta_dict
    self.read_data = reader.read_data 
    # rsel model (GRU)
    self.rnn_hidden_size = 128
    self.rnn_num_layers = 2
    self.rnn_step_len = 1.  # in sec
    self.rnn_step_stride = 0.1
    self.rnn_num_steps = int((self.win_len - self.rnn_step_len) / self.rnn_step_stride) + 1
    # rsel train
    self.num_epochs = 10
    self.batch_size = 128
    self.lr = 1e-4
    self.neg_ratio = 0.5
    self.ckpt_step = 100
    self.summary_step = 10
    # picking config
    self.trig_thres = 0.5
    self.picker_batch_size = 20
    self.tp_dev = 1.5 # whether same pick in different sliding windows
    self.ts_dev = 1.5
    self.amp_win = [1,5] # sec pre-P & post-S for amp calc
