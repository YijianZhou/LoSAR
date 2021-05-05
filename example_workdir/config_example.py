class Config(object):
  """ Configure file for CERP_Pytorch
  """
  def __init__(self):

    # data preprocessing
    self.samp_rate = 100
    self.win_len = 20 # sec
    self.num_chn = 3
    self.step_len = 0.8 # in sec
    self.step_stride = 0.2
    self.num_steps = int((self.win_len - self.step_len) / self.step_stride) + 1
    self.freq_band = [2,40]
    self.global_max_norm = True
    # cnn model
    self.num_cnn_layers = 8
    self.num_cnn_kernels = 32
    self.kernel_size = 3
    # cnn train
    self.to_init_cnn = False
    self.cnn_num_epochs = 20
    self.cnn_batch_size = 128
    self.cnn_lr = 1e-4
    self.cnn_ckpt_step = 25
    self.cnn_summary_step = 10
    self.cnn_num_workers = 10 # set to 0 if use lbdm or hdf5
    # rnn model
    self.rnn_hidden_size = 32
    self.num_rnn_layers = 2
    # rnn train
    self.to_init_rnn = False
    self.rnn_num_epochs = 30
    self.rnn_batch_size = 128
    self.rnn_lr = 1e-4
    self.rnn_ckpt_step = 25
    self.rnn_summary_step = 10
    self.rnn_num_workers = 2 # set to 0 if use lbdm or hdf5
    # picker
    self.picker_batch_size = 100
    self.tp_dev = 2.
    self.ts_dev = 2.
    self.amp_win = [1,4] # sec pre-P & post-S for amp calc
    self.to_repick = [False, True][1]
    self.p_win = [1.,1.]
    self.s_win = [1.,1.]
    self.win_kurt = [5.,1.]
    self.win_sta  = [0.4,1.]
    self.win_lta  = [2.,2.]

