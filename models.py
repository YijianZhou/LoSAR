""" Defination of RNN Seis Labelling (RSeL) Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
cfg = config.Config()

class RSeL(nn.Module):
  def __init__(self):
    super(RSeL, self).__init__()
    # hyper-params
    self.input_size = int(cfg.rnn_step_len * cfg.num_chn * cfg.samp_rate)
    self.hidden_size = cfg.rnn_hidden_size
    self.num_layers = cfg.rnn_num_layers
    # def layers
    self.gru_layer = nn.GRU(input_size=self.input_size, 
        hidden_size=self.hidden_size, 
        num_layers=self.num_layers, 
        batch_first=True, bidirectional=True)
    self.fc_layer = nn.Linear(2*self.hidden_size, 3)

  def forward(self, x):
    x, _ = self.gru_layer(x)
    return self.fc_layer(x)

