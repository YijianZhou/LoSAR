""" PAL Configure file
"""
import data_pipeline as dp
import numpy as np

class Config(object):
  def __init__(self):

    # 1. picker params
    self.win_sta     = [0.8,0.4,1.]   # pick win for STA/LTA
    self.win_lta     = [6.,2.,2.]     # pick win for STA/LTA
    self.trig_thres  = 12.            # threshold to trig picker (by energy)
    self.p_win       = [.5,1.]        # win len for P picking
    self.s_win       = 10.         # win len for S picking
    self.pca_win     = 1.          # win len for PCA filter
    self.pca_range   = [0.,2.]     # time range to apply PCA filter
    self.win_kurt    = [5.,1.]     # win for kurtosis calc
    self.fd_thres    = 2.5         # min value of dominant frequency
    self.amp_win     = [1.,4.]     # time win to get S amplitude
    self.det_gap     = 5.          # time gap between detections
    self.to_prep = True
    self.freq_band   = [2,40]     # frequency band for ppk
    # 2. assoc params
    self.min_sta    = 4             # min num of stations to assoc
    self.ot_dev     = 1.5           # max time deviation for ot assoc
    self.xy_margin  = 0.1          # ratio of lateral margin, relative to sta range
    self.xy_grid    = 0.02          # lateral grid width, in degree
    self.z_grids    = np.arange(1,16,2)           # z (dep) grids
    self.max_res    = 1.            # max P res for loc assoc
    self.vp         = 5.9
    # 3. data interface
    self.get_data_dict = dp.get_rc_data
    self.get_sta_dict = dp.get_sta_dict
    self.get_picks = dp.get_cerp_picks
    self.read_data = dp.read_rc_data

