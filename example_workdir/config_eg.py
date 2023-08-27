import sys
sys.path.append('/home/zhouyj/software/CERP_TED/preprocess')
import reader

class Config(object):
  """ Configure file for CERP_TED
  """
  def __init__(self):

    # data prep & training sample cut
    self.samp_rate = 100
    self.win_len = 20  # sec
    self.win_stride = 8  # CERP step stride, sec
    self.num_chn = 3
    self.step_len = 0.8  # sec
    self.step_stride = 0.2
    self.num_steps = int((self.win_len - self.step_len) / self.step_stride) + 1
    self.freq_band = [1,20]
    self.global_max_norm = False
    self.to_prep = True
    self.train_ratio = 0.9
    self.valid_ratio = 0.1  # ratio of samples to cut 
    self.glitch_ratio = 0.1  # ratio of glitches to cut
    self.num_aug = 2  # whether data augment
    self.max_noise = 0.5  # n time P std
    self.neg_ref = ['P','S'][1] # start time for negative window
    self.read_fpha = reader.read_fpha 
    self.get_data_dict = reader.get_data_dict
    self.get_sta_dict = reader.get_sta_dict
    self.read_data = reader.read_data
    # cnn model
    self.num_cnn_kernels = 64
    self.cnn_stem_kernel_size = [7]
    self.cnn_stem_conv_stride = [2]
    self.num_cnn_res_blocks = [2,2,2]
    # cnn train
    self.cnn_num_epochs = 10
    self.cnn_batch_size = 128
    self.cnn_lr = 1e-4
    self.cnn_ckpt_step = 25
    self.cnn_summary_step = 10
    self.cnn_out_classes = 2  # 2 for P/N; 3 for P/N/G
    # rnn model
    self.rnn_hidden_size = 32
    self.num_rnn_layers = 2
    # rnn train
    self.rnn_num_epochs = 14
    self.rnn_batch_size = 128
    self.rnn_lr = 1e-4
    self.rnn_ckpt_step = 25
    self.rnn_summary_step = 10
    # PAL picker
    self.picker_batch_size = 20
    self.tp_dev = 2.  # whether same pick in different sliding windows
    self.ts_dev = 2.
    self.amp_win = [1,5]  # sec pre-P & post-S for amp calc
    self.to_repick = [False, True][1]
    self.p_win = [1.,1.]
    self.s_win = [1.,1.]
    self.win_kurt = [5.,1.]
    self.win_sta  = [0.4,1.]
    self.win_lta  = [2.,2.]

