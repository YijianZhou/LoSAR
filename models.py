""" Defination of CNN & RNN Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

# set hyper-params
cfg = config.Config()
# data info
num_chn = cfg.num_chn
win_len = int(cfg.win_len * cfg.samp_rate)
step_len = int(cfg.step_len * cfg.samp_rate)
step_stride = int(cfg.step_stride * cfg.samp_rate)


class EventNet(nn.Module):
  """ CNN for event detection
  """
  def __init__(self):
    super(EventNet, self).__init__()
    # hyper-params
    self.in_channels = num_chn
    self.num_layers = cfg.num_cnn_layers
    self.num_kernels = cfg.num_cnn_kernels
    self.kernel_size = cfg.kernel_size
    self.out_channels = int(win_len / 2**(self.num_layers+1))
    # def layers
    self.conv_layers = self._make_conv_layers()
    self.avgpool = nn.AdaptiveAvgPool1d(self.out_channels)
    self.fc_layer = nn.Linear(self.num_kernels*self.out_channels, 2)

  def forward(self, x):
    x = self.conv_layers(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return self.fc_layer(x)

  def _make_conv_layers(self):
    layers = []
    # input size: batch_size * num_chn * win_len
    layers += [nn.Conv1d(self.in_channels, self.num_kernels, self.kernel_size), 
               nn.BatchNorm1d(self.num_kernels),
               nn.ReLU(inplace=True), 
               nn.MaxPool1d(2)]
    for i in range(self.num_layers-1):
        layers += [nn.Conv1d(self.num_kernels, self.num_kernels, self.kernel_size),
                   nn.BatchNorm1d(self.num_kernels), 
                   nn.ReLU(inplace=True), 
                   nn.MaxPool1d(2)]
    return nn.Sequential(*layers)



class PhaseNet(nn.Module):
  """ RNN for phase picking
  """
  def __init__(self):
    super(PhaseNet, self).__init__()
    # hyper-params
    self.input_size = int(step_len * num_chn)
    self.hidden_size = cfg.rnn_hidden_size
    self.num_layers = cfg.num_rnn_layers
    # def layers
    self.gru_layer = nn.GRU(input_size=self.input_size, 
        hidden_size=self.hidden_size, 
        num_layers=self.num_layers, 
        batch_first=True, bidirectional=True)
    self.fc_layer = nn.Linear(2*self.hidden_size, 3)

  def forward(self, x):
    x, _ = self.gru_layer(x)
    return self.fc_layer(x)

